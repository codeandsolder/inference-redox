use std::cmp::Ordering;

use crate::{
    Endpoint, InferenceRequest, ProviderError, ReasoningEffort, ResponseSchema, RouteCandidate,
    RoutePlan, RoutingStrategy,
};

/// Cheap cross-model token estimate used only for routing/cost prediction.
/// Callers can set `InferenceRequest::prompt_tokens` when they have a model tokenizer.
#[must_use]
pub fn approximate_prompt_tokens(prompt: &str) -> u64 {
    let bytes = u64::try_from(prompt.len()).unwrap_or(u64::MAX);
    bytes.div_ceil(4).max(1)
}

fn supports_request(endpoint: &Endpoint, request: &InferenceRequest, prompt_tokens: u64) -> bool {
    if endpoint.status < 0 {
        return false;
    }
    let total = prompt_tokens.saturating_add(request.expected_output_tokens);
    if endpoint.context_length.is_some_and(|limit| total > limit)
        || endpoint
            .max_prompt_tokens
            .is_some_and(|limit| prompt_tokens > limit)
        || endpoint
            .max_completion_tokens
            .is_some_and(|limit| request.expected_output_tokens > limit)
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
    if !request
        .requirements
        .required_parameters
        .is_subset(&endpoint.capabilities.supported_parameters)
    {
        return false;
    }
    match request.requirements.response_schema.as_ref() {
        None => {}
        Some(ResponseSchema::JsonObject) => {
            if !endpoint
                .capabilities
                .supported_parameters
                .contains("response_format")
            {
                return false;
            }
        }
        Some(ResponseSchema::JsonSchema { strict, .. }) => {
            if !endpoint
                .capabilities
                .supported_parameters
                .contains("response_format")
            {
                return false;
            }
            if *strict
                && !endpoint
                    .capabilities
                    .supported_parameters
                    .contains("structured_outputs")
            {
                return false;
            }
        }
    }
    if request.requirements.reasoning != ReasoningEffort::None
        && !endpoint
            .capabilities
            .reasoning
            .supports(request.requirements.reasoning)
    {
        return false;
    }
    true
}

fn expected_time(endpoint: &Endpoint, output_tokens: u64, tail: bool) -> Option<f64> {
    let latency = if tail {
        endpoint.stats.latency_seconds.p95_or_interpolate()
    } else {
        endpoint.stats.latency_seconds.p75_or_interpolate()
    }?;
    // Higher throughput percentiles are faster, so using p75 throughput as a
    // tail estimate would be backwards. Until a provider publishes lower
    // throughput percentiles, use median throughput and put the conservative
    // percentile on TTFT/latency instead.
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

/// Filter and rank provider endpoints for a request shape.
///
/// # Errors
/// Returns an error when the strategy is invalid or no endpoint satisfies the request.
pub fn build_route_plan(
    strategy: RoutingStrategy,
    request: &InferenceRequest,
    endpoints: Vec<Endpoint>,
) -> Result<RoutePlan, ProviderError> {
    let prompt_tokens = request
        .prompt_tokens
        .unwrap_or_else(|| approximate_prompt_tokens(&request.prompt));

    let multiplier = match strategy {
        RoutingStrategy::FastestResponseCheap(multiplier)
            if multiplier.is_finite() && multiplier >= 1.0 =>
        {
            multiplier
        }
        RoutingStrategy::FastestResponseCheap(multiplier) => {
            return Err(ProviderError::InvalidStrategy(format!(
                "cost multiplier must be finite and >= 1.0, got {multiplier}"
            )));
        }
    };

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
