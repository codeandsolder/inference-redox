use std::collections::{BTreeMap, BTreeSet};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::{
    AgenticPolicy, Endpoint, EndpointCapabilities, InferenceProvider, InferenceRequest,
    InferenceResponse, ProviderError, ReasoningEffort, RequestRequirements, ResponseSchema,
    RetryPolicy, RouteCandidate, RoutePlan, approximate_prompt_tokens,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConformanceOutcome {
    Passed,
    Failed,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConformanceProbe {
    BasicChat,
    SystemRole,
    JsonObject,
    StrictJsonSchema,
    ReasoningOff,
    ReasoningMinimal,
    ReasoningLow,
    ReasoningMedium,
    ReasoningHigh,
    ReasoningXhigh,
    ReasoningMax,
    ReasoningBudget256,
}

impl ConformanceProbe {
    #[must_use]
    pub const fn test_id(self) -> &'static str {
        match self {
            Self::BasicChat => "basic-chat",
            Self::SystemRole => "system-role",
            Self::JsonObject => "json-object",
            Self::StrictJsonSchema => "strict-json-schema",
            Self::ReasoningOff => "reasoning-off",
            Self::ReasoningMinimal => "reasoning-minimal",
            Self::ReasoningLow => "reasoning-low",
            Self::ReasoningMedium => "reasoning-medium",
            Self::ReasoningHigh => "reasoning-high",
            Self::ReasoningXhigh => "reasoning-xhigh",
            Self::ReasoningMax => "reasoning-max",
            Self::ReasoningBudget256 => "reasoning-budget-256",
        }
    }

    #[must_use]
    pub const fn test_version(self) -> u32 {
        match self {
            Self::ReasoningOff
            | Self::ReasoningMinimal
            | Self::ReasoningLow
            | Self::ReasoningMedium
            | Self::ReasoningHigh
            | Self::ReasoningXhigh
            | Self::ReasoningMax => 6,
            Self::ReasoningBudget256
            | Self::BasicChat
            | Self::SystemRole
            | Self::JsonObject
            | Self::StrictJsonSchema => 1,
        }
    }

    #[must_use]
    pub const fn parameter(self) -> &'static str {
        match self {
            Self::BasicChat => "chat_completions",
            Self::SystemRole => "system_role",
            Self::JsonObject => "response_format",
            Self::StrictJsonSchema => "structured_outputs",
            Self::ReasoningOff
            | Self::ReasoningMinimal
            | Self::ReasoningLow
            | Self::ReasoningMedium
            | Self::ReasoningHigh
            | Self::ReasoningXhigh
            | Self::ReasoningMax => "reasoning",
            Self::ReasoningBudget256 => "reasoning_max_tokens",
        }
    }

    #[must_use]
    pub const fn reasoning_effort(self) -> Option<ReasoningEffort> {
        match self {
            Self::ReasoningOff => Some(ReasoningEffort::None),
            Self::ReasoningMinimal => Some(ReasoningEffort::Minimal),
            Self::ReasoningLow => Some(ReasoningEffort::Low),
            Self::ReasoningMedium => Some(ReasoningEffort::Medium),
            Self::ReasoningHigh => Some(ReasoningEffort::High),
            Self::ReasoningXhigh => Some(ReasoningEffort::Xhigh),
            Self::ReasoningMax => Some(ReasoningEffort::Max),
            Self::BasicChat
            | Self::SystemRole
            | Self::JsonObject
            | Self::StrictJsonSchema
            | Self::ReasoningBudget256 => None,
        }
    }

    #[must_use]
    pub const fn is_reasoning_probe(self) -> bool {
        matches!(
            self,
            Self::ReasoningOff
                | Self::ReasoningMinimal
                | Self::ReasoningLow
                | Self::ReasoningMedium
                | Self::ReasoningHigh
                | Self::ReasoningXhigh
                | Self::ReasoningMax
                | Self::ReasoningBudget256
        )
    }

    #[must_use]
    pub fn request(self, model: impl Into<String>) -> InferenceRequest {
        let model = model.into();
        match self {
            Self::BasicChat => {
                let mut request = InferenceRequest::new(model, "Reply with PROBE_OK.", 16);
                request.max_output_tokens = Some(256);
                request.temperature = Some(0.0);
                request
            }
            Self::SystemRole => {
                let mut request = InferenceRequest::new(model, "Reply briefly.", 16);
                request.system_prompt = Some("This is a protocol conformance probe.".to_owned());
                request.max_output_tokens = Some(256);
                request.temperature = Some(0.0);
                request
            }
            Self::JsonObject => {
                let mut request = InferenceRequest::new(
                    model,
                    "Return a JSON object whose only key is ok and whose value is true.",
                    16,
                );
                request.max_output_tokens = Some(256);
                request.temperature = Some(0.0);
                request.requirements = RequestRequirements {
                    response_schema: Some(ResponseSchema::JsonObject),
                    ..RequestRequirements::default()
                };
                request
            }
            Self::StrictJsonSchema => {
                let mut request = InferenceRequest::new(
                    model,
                    "Return the requested object with ok set to true.",
                    16,
                );
                request.max_output_tokens = Some(256);
                request.temperature = Some(0.0);
                request.requirements = RequestRequirements {
                    response_schema: Some(ResponseSchema::JsonSchema {
                        name: "conformance".to_owned(),
                        schema: serde_json::json!({
                            "type": "object",
                            "properties": {"ok": {"type": "boolean", "const": true}},
                            "required": ["ok"],
                            "additionalProperties": false
                        }),
                        strict: true,
                    }),
                    ..RequestRequirements::default()
                };
                request
            }
            Self::ReasoningOff
            | Self::ReasoningMinimal
            | Self::ReasoningLow
            | Self::ReasoningMedium
            | Self::ReasoningHigh
            | Self::ReasoningXhigh
            | Self::ReasoningMax => {
                let mut request = InferenceRequest::new(
                    model,
                    "A,B,C,D,E,F are a permutation of 1..6. A+B=7; B+C=8; C+D=9; D+E=7; E+F=5; |A-F|=2. Find ABCDEF. Reply with the 6 digits only.",
                    16,
                );
                request.max_output_tokens = Some(2048);
                request.include_reasoning = true;
                request.requirements.reasoning =
                    self.reasoning_effort().unwrap_or(ReasoningEffort::None);
                request
                    .requirements
                    .required_parameters
                    .insert("reasoning".to_owned());
                request
            }
            Self::ReasoningBudget256 => {
                let mut request = InferenceRequest::new(
                    model,
                    "A,B,C,D,E,F are a permutation of 1..6. A+B=7; B+C=8; C+D=9; D+E=7; E+F=5; |A-F|=2. Find ABCDEF. Reply with the 6 digits only.",
                    16,
                );
                request.max_output_tokens = Some(2048);
                request.include_reasoning = true;
                request.requirements.reasoning_max_tokens = Some(256);
                request
            }
        }
    }

    /// Run one probe against a single provider/model scope. The supplied endpoint is used only
    /// as the exact route candidate; evidence is provider-scoped unless a future adapter can
    /// independently address a narrower endpoint.
    pub async fn run_provider_model<P: InferenceProvider + ?Sized>(
        self,
        provider: &P,
        endpoint: Endpoint,
        canonical_id: Option<String>,
    ) -> ConformanceObservation {
        let source = endpoint.source.clone();
        let credential = endpoint.credential.clone();
        let provider_key = endpoint.provider_key.clone();
        let endpoint_id = Some(endpoint.id.clone());
        let native_model_id = endpoint.model.clone();
        let mut request = self.request(native_model_id.clone());
        if !endpoint
            .capabilities
            .adapter_parameters
            .contains("include_reasoning")
        {
            request.include_reasoning = false;
        }
        if endpoint.workload_policy.agentic == AgenticPolicy::Required {
            request.agentic = true;
        }
        if endpoint.workload_policy.requires_stable_session {
            request.session_id = Some("inference-redox-conformance".to_owned());
        }
        let prompt_tokens = request.prompt_tokens.unwrap_or_else(|| {
            request
                .system_prompt
                .as_deref()
                .map_or(0, approximate_prompt_tokens)
                .saturating_add(approximate_prompt_tokens(&request.prompt))
        });
        let expected_cost_usd = endpoint
            .pricing
            .expected_cost(prompt_tokens, request.expected_output_tokens);
        let plan = RoutePlan {
            prompt_tokens,
            expected_output_tokens: request.expected_output_tokens,
            cheapest_cost_usd: expected_cost_usd,
            cost_ceiling_usd: expected_cost_usd,
            candidates: vec![RouteCandidate {
                endpoint,
                expected_cost_usd,
                expected_response_p75_seconds: None,
                expected_response_p95_seconds: None,
            }],
        };
        let retry = RetryPolicy {
            max_attempts: 1,
            ..RetryPolicy::default()
        };
        let mut reasoning_tokens = None;
        let mut completion_tokens = None;
        let mut reasoning_chars = None;
        let mut puzzle_correct = None;
        let mut finish_reason = None;
        let (outcome, detail) = match provider.execute_plan(&request, &plan, retry).await {
            Ok(response) => {
                reasoning_tokens = response.usage.reasoning_tokens;
                completion_tokens = response.usage.output_tokens;
                reasoning_chars = response
                    .raw
                    .pointer("/choices/0/message/reasoning")
                    .and_then(serde_json::Value::as_str)
                    .and_then(|value| u64::try_from(value.chars().count()).ok());
                finish_reason = response
                    .raw
                    .pointer("/choices/0/finish_reason")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned);
                if self.is_reasoning_probe() {
                    puzzle_correct = Some(response.content.trim() == "253614");
                }
                match self.validate_response(&response) {
                    Ok(()) => (ConformanceOutcome::Passed, None),
                    Err(detail) => (ConformanceOutcome::Failed, Some(detail)),
                }
            }
            Err(error) => (
                classify_probe_error(&error),
                Some(probe_error_detail(&error)),
            ),
        };
        ConformanceObservation {
            source,
            credential,
            provider_key,
            endpoint_id,
            native_model_id,
            canonical_id,
            parameter: self.parameter().to_owned(),
            reasoning_effort: self.reasoning_effort(),
            reasoning_tokens,
            completion_tokens,
            reasoning_chars,
            puzzle_correct,
            finish_reason,
            test_id: self.test_id().to_owned(),
            test_version: self.test_version(),
            outcome,
            observed_unix_ms: unix_time_ms(),
            detail,
        }
    }

    fn validate_response(self, response: &InferenceResponse) -> Result<(), String> {
        match self {
            Self::JsonObject | Self::StrictJsonSchema => {
                let value: serde_json::Value = serde_json::from_str(&response.content)
                    .map_err(|error| format!("response was not JSON: {error}"))?;
                if value.get("ok").and_then(serde_json::Value::as_bool) == Some(true) {
                    Ok(())
                } else {
                    Err("response JSON did not contain boolean ok=true".to_owned())
                }
            }
            Self::BasicChat | Self::SystemRole => {
                if response.content.trim().is_empty() {
                    Err("response content was empty".to_owned())
                } else {
                    Ok(())
                }
            }
            Self::ReasoningOff
            | Self::ReasoningMinimal
            | Self::ReasoningLow
            | Self::ReasoningMedium
            | Self::ReasoningHigh
            | Self::ReasoningXhigh
            | Self::ReasoningMax
            | Self::ReasoningBudget256 => Ok(()),
        }
    }
}

fn classify_probe_error(error: &ProviderError) -> ConformanceOutcome {
    match probe_http_status(error) {
        Some(400 | 422) => ConformanceOutcome::Failed,
        _ => ConformanceOutcome::Inconclusive,
    }
}

fn probe_error_detail(error: &ProviderError) -> String {
    probe_http_status(error).map_or_else(
        || match error {
            ProviderError::NoEligibleEndpoint(_) => "no eligible endpoint".to_owned(),
            ProviderError::InvalidRequest(_) => "invalid request".to_owned(),
            ProviderError::InvalidStrategy(_) => "invalid strategy".to_owned(),
            ProviderError::Catalog(_) => "catalog error".to_owned(),
            ProviderError::Request(_) => "request error".to_owned(),
            ProviderError::Execution { .. } => "execution error".to_owned(),
            ProviderError::Response(_) => "response error".to_owned(),
        },
        |status| format!("HTTP {status}"),
    )
}

fn probe_http_status(error: &ProviderError) -> Option<u16> {
    let text = match error {
        ProviderError::Catalog(text)
        | ProviderError::Request(text)
        | ProviderError::Response(text) => text,
        ProviderError::Execution { detail, .. } => detail,
        ProviderError::NoEligibleEndpoint(_)
        | ProviderError::InvalidRequest(_)
        | ProviderError::InvalidStrategy(_) => return None,
    };
    let tail = text.split_once("HTTP ")?.1;
    let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
        .unwrap_or(0)
}

/// Versioned evidence from one provider/model conformance probe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceObservation {
    /// Execution surface, e.g. `openrouter` or direct `nvidia`.
    pub source: String,
    /// Opaque logical credential/account label. Never contains the secret itself.
    pub credential: Option<String>,
    /// Stable provider selector within the execution surface.
    pub provider_key: String,
    /// Optional exact endpoint refinement when that endpoint is independently addressable.
    pub endpoint_id: Option<String>,
    /// Model identifier spoken to this provider route.
    pub native_model_id: String,
    pub canonical_id: Option<String>,
    pub parameter: String,
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    #[serde(default)]
    pub reasoning_chars: Option<u64>,
    #[serde(default)]
    pub puzzle_correct: Option<bool>,
    #[serde(default)]
    pub finish_reason: Option<String>,
    pub test_id: String,
    pub test_version: u32,
    pub outcome: ConformanceOutcome,
    pub observed_unix_ms: u64,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceProfile {
    pub observations: Vec<ConformanceObservation>,
}

impl ConformanceProfile {
    /// Return a current-version cached observation for one exact probe scope.
    #[must_use]
    pub fn cached_probe(
        &self,
        endpoint: &Endpoint,
        probe: ConformanceProbe,
    ) -> Option<&ConformanceObservation> {
        self.observations
            .iter()
            .filter(|observation| observation.matches_endpoint(endpoint))
            .filter(|observation| {
                observation.parameter == probe.parameter()
                    && observation.test_id == probe.test_id()
                    && observation.test_version == probe.test_version()
            })
            .max_by_key(|observation| observation.observed_unix_ms)
    }

    /// Record one observation, replacing an older observation for the same exact probe/version.
    pub fn record(&mut self, observation: ConformanceObservation) {
        if let Some(existing) = self.observations.iter_mut().find(|existing| {
            existing.same_probe_scope(&observation)
                && existing.test_version == observation.test_version
        }) {
            if observation.observed_unix_ms >= existing.observed_unix_ms {
                *existing = observation;
            }
            return;
        }
        self.observations.push(observation);
    }

    /// Apply only observations scoped to this exact endpoint/model route.
    pub fn apply_to_endpoint(&self, endpoint: &mut Endpoint) {
        let source = endpoint.source.clone();
        let credential = endpoint.credential.clone();
        let provider_key = endpoint.provider_key.clone();
        let endpoint_id = endpoint.id.clone();
        let native_model_id = endpoint.model.clone();
        let observations = self.observations.iter().filter(|observation| {
            observation.source == source
                && observation.credential == credential
                && observation.provider_key == provider_key
                && observation.native_model_id == native_model_id
                && observation
                    .endpoint_id
                    .as_ref()
                    .is_none_or(|id| id == &endpoint_id)
        });
        apply_observations(&mut endpoint.capabilities, observations);
    }
}

impl ConformanceObservation {
    fn effective_reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.reasoning_effort.or(match self.test_id.as_str() {
            "reasoning-off" => Some(ReasoningEffort::None),
            "reasoning-minimal" => Some(ReasoningEffort::Minimal),
            "reasoning-low" => Some(ReasoningEffort::Low),
            "reasoning-medium" => Some(ReasoningEffort::Medium),
            "reasoning-high" => Some(ReasoningEffort::High),
            "reasoning-xhigh" => Some(ReasoningEffort::Xhigh),
            "reasoning-max" => Some(ReasoningEffort::Max),
            _ => None,
        })
    }

    fn matches_endpoint(&self, endpoint: &Endpoint) -> bool {
        self.source == endpoint.source
            && self.credential == endpoint.credential
            && self.provider_key == endpoint.provider_key
            && self.native_model_id == endpoint.model
            && self
                .endpoint_id
                .as_ref()
                .is_none_or(|endpoint_id| endpoint_id == &endpoint.id)
    }

    fn same_probe_scope(&self, other: &Self) -> bool {
        self.source == other.source
            && self.credential == other.credential
            && self.provider_key == other.provider_key
            && self.endpoint_id == other.endpoint_id
            && self.native_model_id == other.native_model_id
            && self.parameter == other.parameter
            && self.test_id == other.test_id
    }
}

fn apply_observations<'a>(
    capabilities: &mut EndpointCapabilities,
    observations: impl Iterator<Item = &'a ConformanceObservation>,
) {
    // Newer test definitions supersede older definitions. Within one definition,
    // the newest observation wins.
    let mut latest: BTreeMap<(&str, &str), &ConformanceObservation> = BTreeMap::new();
    for observation in observations {
        let key = (observation.parameter.as_str(), observation.test_id.as_str());
        let replace = latest.get(&key).is_none_or(|current| {
            (observation.test_version, observation.observed_unix_ms)
                > (current.test_version, current.observed_unix_ms)
        });
        if replace {
            latest.insert(key, observation);
        }
    }

    let mut passed = BTreeSet::new();
    let mut failed = BTreeSet::new();
    let mut reasoning_passed = BTreeSet::new();
    let mut reasoning_failed = BTreeSet::new();
    for observation in latest.values() {
        if let Some(effort) = observation.effective_reasoning_effort() {
            match observation.outcome {
                ConformanceOutcome::Passed => {
                    reasoning_passed.insert(effort);
                    // A successful level proves that the normalized reasoning control itself works.
                    passed.insert("reasoning".to_owned());
                }
                ConformanceOutcome::Failed => {
                    reasoning_failed.insert(effort);
                }
                ConformanceOutcome::Inconclusive => {}
            }
            continue;
        }
        match observation.outcome {
            ConformanceOutcome::Passed => {
                passed.insert(observation.parameter.clone());
            }
            ConformanceOutcome::Failed => {
                failed.insert(observation.parameter.clone());
            }
            ConformanceOutcome::Inconclusive => {}
        }
    }

    for parameter in passed.difference(&failed) {
        capabilities.rejected_parameters.remove(parameter);
        capabilities.verified_parameters.insert(parameter.clone());
    }
    for parameter in failed {
        capabilities.verified_parameters.remove(&parameter);
        capabilities.rejected_parameters.insert(parameter);
    }
    for effort in reasoning_passed.difference(&reasoning_failed) {
        capabilities.rejected_reasoning_efforts.remove(effort);
        capabilities.verified_reasoning_efforts.insert(*effort);
    }
    for effort in reasoning_failed {
        capabilities.verified_reasoning_efforts.remove(&effort);
        capabilities.rejected_reasoning_efforts.insert(effort);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EndpointStats, Pricing};

    fn endpoint(id: &str, model: &str) -> Endpoint {
        Endpoint {
            source: "openrouter".to_owned(),
            credential: None,
            provider_key: id.split('/').next().unwrap_or(id).to_owned(),
            id: id.to_owned(),
            provider: "provider".to_owned(),
            model: model.to_owned(),
            route_class: crate::ModelRouteClass::Standard,
            display_name: id.to_owned(),
            quantization: None,
            status: 0,
            context_length: None,
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            stats: EndpointStats::default(),
            workload_policy: crate::WorkloadPolicy::default(),
            capabilities: EndpointCapabilities {
                adapter_parameters: [
                    "chat_completions",
                    "system_role",
                    "response_format",
                    "structured_outputs",
                    "reasoning",
                    "reasoning_effort",
                    "seed",
                    "tools",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
                ..EndpointCapabilities::default()
            },
        }
    }

    fn observation(
        endpoint_id: &str,
        model: &str,
        parameter: &str,
        test_id: &str,
        version: u32,
        observed_unix_ms: u64,
        outcome: ConformanceOutcome,
    ) -> ConformanceObservation {
        ConformanceObservation {
            source: "openrouter".to_owned(),
            credential: None,
            provider_key: endpoint_id
                .split('/')
                .next()
                .unwrap_or(endpoint_id)
                .to_owned(),
            endpoint_id: Some(endpoint_id.to_owned()),
            native_model_id: model.to_owned(),
            canonical_id: None,
            parameter: parameter.to_owned(),
            reasoning_effort: None,
            reasoning_tokens: None,
            completion_tokens: None,
            reasoning_chars: None,
            puzzle_correct: None,
            finish_reason: None,
            test_id: test_id.to_owned(),
            test_version: version,
            outcome,
            observed_unix_ms,
            detail: None,
        }
    }

    #[test]
    fn newer_probe_result_replaces_older_result_for_same_test() {
        let mut route = endpoint("endpoint", "model");
        route
            .capabilities
            .supported_parameters
            .insert("structured_outputs".to_owned());
        ConformanceProfile {
            observations: vec![
                observation(
                    "endpoint",
                    "model",
                    "structured_outputs",
                    "strict-json-schema",
                    1,
                    10,
                    ConformanceOutcome::Failed,
                ),
                observation(
                    "endpoint",
                    "model",
                    "structured_outputs",
                    "strict-json-schema",
                    1,
                    20,
                    ConformanceOutcome::Passed,
                ),
            ],
        }
        .apply_to_endpoint(&mut route);
        assert!(route.capabilities.supports_parameter("structured_outputs"));
        assert!(
            route
                .capabilities
                .verified_parameters
                .contains("structured_outputs")
        );
        assert!(route.capabilities.rejected_parameters.is_empty());
    }

    #[test]
    fn failed_test_rejects_parameter_only_on_matching_route() {
        let mut target = endpoint("endpoint-a", "model");
        target
            .capabilities
            .supported_parameters
            .insert("tools".to_owned());
        let mut other = endpoint("endpoint-b", "model");
        other
            .capabilities
            .supported_parameters
            .insert("tools".to_owned());
        let profile = ConformanceProfile {
            observations: vec![
                observation(
                    "endpoint-a",
                    "model",
                    "tools",
                    "single-tool",
                    1,
                    10,
                    ConformanceOutcome::Passed,
                ),
                observation(
                    "endpoint-a",
                    "model",
                    "tools",
                    "streamed-tool-args",
                    1,
                    10,
                    ConformanceOutcome::Failed,
                ),
            ],
        };
        profile.apply_to_endpoint(&mut target);
        profile.apply_to_endpoint(&mut other);
        assert!(!target.capabilities.supports_parameter("tools"));
        assert!(other.capabilities.supports_parameter("tools"));
    }

    #[test]
    fn evidence_does_not_cross_execution_surfaces() {
        let mut route = endpoint("deepinfra/fp8", "model");
        route
            .capabilities
            .supported_parameters
            .insert("tools".to_owned());
        let mut observation = observation(
            "deepinfra/fp8",
            "model",
            "tools",
            "single-tool",
            1,
            10,
            ConformanceOutcome::Failed,
        );
        observation.source = "direct-deepinfra".to_owned();
        ConformanceProfile {
            observations: vec![observation],
        }
        .apply_to_endpoint(&mut route);
        assert!(route.capabilities.supports_parameter("tools"));
    }

    #[test]
    fn probe_error_detail_drops_provider_response_body() {
        let error = ProviderError::Execution {
            phase: crate::ExecutionFailurePhase::ResponseReceived,
            detail: "HTTP 429 Too Many Requests: {\"user_id\":\"secret-ish\"}".to_owned(),
            quota_observations: Vec::new(),
        };
        assert_eq!(probe_error_detail(&error), "HTTP 429");
        assert_eq!(
            classify_probe_error(&error),
            ConformanceOutcome::Inconclusive
        );
    }

    #[test]
    fn inconclusive_result_does_not_override_existing_evidence() {
        let mut route = endpoint("endpoint", "model");
        route
            .capabilities
            .verified_parameters
            .insert("seed".to_owned());
        ConformanceProfile {
            observations: vec![observation(
                "endpoint",
                "model",
                "seed",
                "determinism",
                1,
                10,
                ConformanceOutcome::Inconclusive,
            )],
        }
        .apply_to_endpoint(&mut route);
        assert!(route.capabilities.verified_parameters.contains("seed"));
        assert!(route.capabilities.rejected_parameters.is_empty());
    }

    #[test]
    fn reasoning_off_probe_is_explicit_and_requests_trace() {
        let request = ConformanceProbe::ReasoningOff.request("model");
        assert_eq!(request.requirements.reasoning, ReasoningEffort::None);
        assert!(
            request
                .requirements
                .required_parameters
                .contains("reasoning")
        );
        assert!(request.include_reasoning);
    }

    #[test]
    fn reasoning_budget_probe_is_independent_from_effort() {
        let request = ConformanceProbe::ReasoningBudget256.request("model");
        assert_eq!(request.requirements.reasoning, ReasoningEffort::None);
        assert_eq!(request.requirements.reasoning_max_tokens, Some(256));
        assert!(request.include_reasoning);
        assert_eq!(
            ConformanceProbe::ReasoningBudget256.parameter(),
            "reasoning_max_tokens"
        );
    }

    #[test]
    fn reasoning_evidence_is_scoped_to_exact_effort() {
        let mut route = endpoint("endpoint", "model");
        route
            .capabilities
            .supported_parameters
            .insert("reasoning".to_owned());
        route.capabilities.reasoning = crate::ReasoningSupport::GatewayNormalized;

        let mut low = observation(
            "endpoint",
            "model",
            "reasoning",
            "reasoning-low",
            3,
            10,
            ConformanceOutcome::Passed,
        );
        low.reasoning_effort = Some(ReasoningEffort::Low);
        let mut high = observation(
            "endpoint",
            "model",
            "reasoning",
            "reasoning-high",
            3,
            10,
            ConformanceOutcome::Failed,
        );
        high.reasoning_effort = Some(ReasoningEffort::High);

        ConformanceProfile {
            observations: vec![low, high],
        }
        .apply_to_endpoint(&mut route);

        assert!(
            route
                .capabilities
                .verified_reasoning_efforts
                .contains(&ReasoningEffort::Low)
        );
        assert!(
            route
                .capabilities
                .rejected_reasoning_efforts
                .contains(&ReasoningEffort::High)
        );
        assert!(
            route
                .capabilities
                .supports_reasoning_effort(ReasoningEffort::Low, true)
        );
        assert!(
            !route
                .capabilities
                .supports_reasoning_effort(ReasoningEffort::High, false)
        );
        assert!(
            route
                .capabilities
                .supports_reasoning_effort(ReasoningEffort::Medium, false)
        );
    }

    #[test]
    fn historical_reasoning_probe_ids_recover_effort_scope() {
        let mut route = endpoint("endpoint", "model");
        route
            .capabilities
            .supported_parameters
            .insert("reasoning".to_owned());
        route.capabilities.reasoning = crate::ReasoningSupport::GatewayNormalized;
        ConformanceProfile {
            observations: vec![observation(
                "endpoint",
                "model",
                "reasoning",
                "reasoning-low",
                1,
                10,
                ConformanceOutcome::Failed,
            )],
        }
        .apply_to_endpoint(&mut route);
        assert!(
            route
                .capabilities
                .rejected_reasoning_efforts
                .contains(&ReasoningEffort::Low)
        );
        assert!(!route.capabilities.rejected_parameters.contains("reasoning"));
    }

    #[test]
    fn cached_probe_is_exactly_credential_and_endpoint_scoped() {
        let mut route = endpoint("endpoint", "model");
        route.credential = Some("account-a".to_owned());
        let mut cached = observation(
            "endpoint",
            "model",
            "chat_completions",
            "basic-chat",
            ConformanceProbe::BasicChat.test_version(),
            10,
            ConformanceOutcome::Passed,
        );
        cached.credential = Some("account-a".to_owned());
        let profile = ConformanceProfile {
            observations: vec![cached],
        };
        assert!(
            profile
                .cached_probe(&route, ConformanceProbe::BasicChat)
                .is_some()
        );
        route.credential = Some("account-b".to_owned());
        assert!(
            profile
                .cached_probe(&route, ConformanceProbe::BasicChat)
                .is_none()
        );
    }

    #[test]
    fn record_replaces_same_probe_version_instead_of_growing_forever() {
        let first = observation(
            "endpoint",
            "model",
            "chat_completions",
            "basic-chat",
            1,
            10,
            ConformanceOutcome::Inconclusive,
        );
        let mut second = first.clone();
        second.observed_unix_ms = 20;
        second.outcome = ConformanceOutcome::Passed;
        let mut profile = ConformanceProfile::default();
        profile.record(first);
        profile.record(second);
        assert_eq!(profile.observations.len(), 1);
        assert_eq!(profile.observations[0].outcome, ConformanceOutcome::Passed);
    }
}
