use std::cmp::Ordering;

use crate::{
    AgenticPolicy, Endpoint, InferenceRequest, ProviderError, ReasoningEffort, ResponseSchema,
    RouteCandidate, RoutePlan, RoutingStrategy, StrictOutputTransport, ToolChoice,
};

/// Cheap cross-model token estimate used only for routing/cost prediction.
/// Callers can set `InferenceRequest::prompt_tokens` when they have a model tokenizer.
#[must_use]
pub fn approximate_prompt_tokens(prompt: &str) -> u64 {
    let bytes = u64::try_from(prompt.len()).unwrap_or(u64::MAX);
    bytes.div_ceil(4).max(1)
}

/// Resolve the prompt-token estimate used consistently by routing and quota admission.
#[must_use]
pub fn request_prompt_tokens(request: &InferenceRequest) -> u64 {
    request.prompt_tokens.unwrap_or_else(|| {
        let system_tokens = request
            .system_prompt
            .as_deref()
            .map_or(0, approximate_prompt_tokens);
        system_tokens.saturating_add(approximate_prompt_tokens(&request.prompt))
    })
}

pub(crate) fn supports_request(
    endpoint: &Endpoint,
    request: &InferenceRequest,
    prompt_tokens: u64,
) -> bool {
    if endpoint.status < 0
        || !request
            .requirements
            .allowed_route_classes
            .contains(&endpoint.route_class)
    {
        return false;
    }
    match endpoint.workload_policy.agentic {
        AgenticPolicy::Required if !request.agentic => return false,
        AgenticPolicy::Forbidden if request.agentic => return false,
        AgenticPolicy::Any | AgenticPolicy::Required | AgenticPolicy::Forbidden => {}
    }
    if endpoint.workload_policy.requires_stable_session
        && request
            .session_id
            .as_deref()
            .is_none_or(|session| session.trim().is_empty())
    {
        return false;
    }
    let output_capacity = request
        .max_output_tokens
        .unwrap_or(request.expected_output_tokens);
    let total = prompt_tokens.saturating_add(output_capacity);
    if endpoint.context_length.is_some_and(|limit| total > limit)
        || endpoint
            .max_prompt_tokens
            .is_some_and(|limit| prompt_tokens > limit)
        || endpoint
            .max_completion_tokens
            .is_some_and(|limit| output_capacity > limit)
    {
        return false;
    }
    if request
        .requirements
        .minimum_uptime_30m
        .zip(endpoint.stats.uptime_30m)
        .is_some_and(|(minimum, actual)| actual < minimum)
    {
        return false;
    }
    if !supports_behavioral_request(endpoint, request) {
        return false;
    }
    supports_reasoning_request(endpoint, request)
}

fn supports_behavioral_request(endpoint: &Endpoint, request: &InferenceRequest) -> bool {
    if !request
        .requirements
        .required_parameters
        .iter()
        .all(|parameter| {
            if matches!(
                parameter.as_str(),
                "response_format" | "structured_outputs" | "tools" | "tool_choice"
            ) {
                endpoint.capabilities.verified_parameter(parameter)
            } else {
                endpoint.capabilities.supports_parameter(parameter)
            }
        })
    {
        return false;
    }
    if request.temperature.is_some() && !endpoint.capabilities.supports_parameter("temperature") {
        return false;
    }
    if request.max_output_tokens.is_some()
        && !endpoint.capabilities.supports_parameter("max_tokens")
        && !endpoint
            .capabilities
            .supports_parameter("max_completion_tokens")
    {
        return false;
    }
    if request.tools.is_empty() {
        if request.tool_choice.is_some() && !matches!(request.tool_choice, Some(ToolChoice::None)) {
            return false;
        }
    } else if !matches!(request.tool_choice, Some(ToolChoice::None))
        && endpoint.capabilities.tool_transport(request).is_none()
    {
        return false;
    }
    match request.requirements.response_schema.as_ref() {
        None => {}
        Some(ResponseSchema::JsonObject) => {
            if !endpoint.capabilities.verified_parameter("response_format") {
                return false;
            }
        }
        Some(ResponseSchema::JsonSchema { strict, .. }) => {
            if *strict {
                let transport = endpoint.capabilities.strict_output_transport();
                if transport.is_none()
                    || (!request.tools.is_empty()
                        && transport == Some(StrictOutputTransport::ForcedTool))
                {
                    return false;
                }
            } else if !endpoint.capabilities.verified_parameter("response_format") {
                return false;
            }
        }
    }
    true
}

fn supports_reasoning_request(endpoint: &Endpoint, request: &InferenceRequest) -> bool {
    if request.include_reasoning
        && !endpoint
            .capabilities
            .supports_parameter("include_reasoning")
    {
        return false;
    }
    let explicit_reasoning = request.requirements.reasoning != ReasoningEffort::None
        || request.requirements.reasoning_max_tokens.is_some()
        || request
            .requirements
            .required_parameters
            .contains("reasoning");
    if !explicit_reasoning {
        return true;
    }
    let parameter_supported = endpoint.capabilities.supports_parameter("reasoning")
        || endpoint.capabilities.supports_parameter("reasoning_effort");
    if !parameter_supported {
        return false;
    }
    if request.requirements.reasoning_max_tokens.is_some() {
        if !endpoint
            .capabilities
            .supports_parameter("reasoning_max_tokens")
        {
            return false;
        }
        if request.requirements.reasoning == ReasoningEffort::None {
            return true;
        }
    }
    endpoint.capabilities.supports_reasoning_effort(
        request.requirements.reasoning,
        request.requirements.require_verified_reasoning,
    )
}

fn expected_time(endpoint: &Endpoint, output_tokens: u64, tail: bool) -> Option<f64> {
    let latency = if tail {
        endpoint.stats.latency_seconds.p95_or_interpolate()
    } else {
        endpoint.stats.latency_seconds.p75_or_interpolate()
    }?;
    // Throughput percentiles run in the opposite desirability direction from
    // latency: OpenRouter's p75/p95 throughput values are progressively faster
    // (more optimistic) samples. The catalog does not expose p25/p05, so use
    // median throughput and put the requested service percentile on TTFT.
    let throughput = endpoint.stats.throughput_tokens_per_second.p50?;
    if throughput <= 0.0 || !throughput.is_finite() || latency < 0.0 || !latency.is_finite() {
        return None;
    }
    Some(latency + tokens_as_f64(output_tokens) / throughput)
}

fn optional_f64_cmp(left: Option<f64>, right: Option<f64>) -> Ordering {
    match (left, right) {
        (Some(a), Some(b)) => a.total_cmp(&b),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

#[allow(clippy::cast_precision_loss)]
fn tokens_as_f64(tokens: u64) -> f64 {
    tokens as f64
}

pub(crate) fn validate_strategy(strategy: RoutingStrategy) -> Result<f64, ProviderError> {
    match strategy {
        RoutingStrategy::FastestResponseCheap(multiplier)
            if multiplier.is_finite() && multiplier >= 1.0 =>
        {
            Ok(multiplier)
        }
        RoutingStrategy::FastestResponseCheap(multiplier) => Err(ProviderError::InvalidStrategy(
            format!("cost multiplier must be finite and >= 1.0, got {multiplier}"),
        )),
    }
}

/// Filter and rank provider endpoints for a request shape.
///
/// # Errors
/// Returns an error when the strategy is invalid or no endpoint satisfies the request.
pub fn build_route_plan(
    strategy: RoutingStrategy,
    request: &InferenceRequest,
    endpoints: Vec<Endpoint>,
) -> Result<RoutePlan, ProviderError> {
    let prompt_tokens = request_prompt_tokens(request);
    if request
        .max_output_tokens
        .is_some_and(|maximum| maximum < request.expected_output_tokens)
    {
        return Err(ProviderError::InvalidRequest(format!(
            "max_output_tokens is smaller than expected_output_tokens for {}",
            request.model
        )));
    }

    let multiplier = validate_strategy(strategy)?;

    let mut eligible: Vec<RouteCandidate> = endpoints
        .into_iter()
        .filter(|endpoint| supports_request(endpoint, request, prompt_tokens))
        .map(|endpoint| RouteCandidate {
            expected_cost_usd: endpoint
                .pricing
                .expected_cost(prompt_tokens, request.expected_output_tokens),
            expected_response_p75_seconds: expected_time(
                &endpoint,
                request.expected_output_tokens,
                false,
            ),
            expected_response_p95_seconds: expected_time(
                &endpoint,
                request.expected_output_tokens,
                true,
            ),
            endpoint,
        })
        .filter(|candidate| {
            candidate.expected_cost_usd.is_finite() && candidate.expected_cost_usd >= 0.0
        })
        .collect();

    let cheapest = eligible
        .iter()
        .map(|candidate| candidate.expected_cost_usd)
        .min_by(f64::total_cmp)
        .ok_or_else(|| ProviderError::NoEligibleEndpoint(request.model.clone()))?;
    let ceiling = cheapest * multiplier;
    eligible.retain(|candidate| candidate.expected_cost_usd <= ceiling + f64::EPSILON);
    eligible.sort_by(|left, right| {
        optional_f64_cmp(
            left.expected_response_p75_seconds,
            right.expected_response_p75_seconds,
        )
        .then_with(|| {
            optional_f64_cmp(
                left.expected_response_p95_seconds,
                right.expected_response_p95_seconds,
            )
        })
        .then_with(|| left.expected_cost_usd.total_cmp(&right.expected_cost_usd))
        .then_with(|| left.endpoint.id.cmp(&right.endpoint.id))
    });

    Ok(RoutePlan {
        prompt_tokens,
        expected_output_tokens: request.expected_output_tokens,
        cheapest_cost_usd: cheapest,
        cost_ceiling_usd: ceiling,
        candidates: eligible,
    })
}
