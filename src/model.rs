use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ModelRouteClass;

#[allow(clippy::cast_precision_loss)]
fn tokens_as_f64(tokens: u64) -> f64 {
    tokens as f64
}

/// Percentile samples for an endpoint metric.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Percentiles {
    pub p50: Option<f64>,
    pub p75: Option<f64>,
    pub p90: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
}

impl Percentiles {
    /// Returns p75, interpolating from nearby published samples when needed.
    #[must_use]
    pub fn p75_or_interpolate(self) -> Option<f64> {
        self.p75.or_else(|| match (self.p50, self.p90) {
            (Some(p50), Some(p90)) => Some(p50 + (p90 - p50) * 0.625),
            (Some(p50), None) => Some(p50),
            (None, Some(p90)) => Some(p90),
            (None, None) => self.p99.or(self.p95),
        })
    }

    /// Returns a p95 value, interpolating between p90 and p99 when the source omits p95.
    #[must_use]
    pub fn p95_or_interpolate(self) -> Option<f64> {
        self.p95.or_else(|| match (self.p90, self.p99) {
            (Some(p90), Some(p99)) => Some(p90 + (p99 - p90) * (5.0 / 9.0)),
            (Some(p90), None) => Some(p90),
            (None, Some(p99)) => Some(p99),
            (None, None) => self.p75.or(self.p50),
        })
    }
}

/// Effective token rates after request-shape pricing overrides are applied.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct EffectivePricing {
    pub prompt_per_token: f64,
    pub completion_per_token: f64,
    pub cache_read_per_token: Option<f64>,
    pub cache_write_per_token: Option<f64>,
}

/// A pricing tier that activates at a prompt-token threshold.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct PricingOverride {
    pub min_prompt_tokens: u64,
    pub prompt_per_token: Option<f64>,
    pub completion_per_token: Option<f64>,
    pub cache_read_per_token: Option<f64>,
    pub cache_write_per_token: Option<f64>,
}

/// Normalized endpoint pricing in USD.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Pricing {
    pub prompt_per_token: f64,
    pub completion_per_token: f64,
    pub request: f64,
    pub discount: f64,
    pub cache_read_per_token: Option<f64>,
    pub cache_write_per_token: Option<f64>,
    pub overrides: Vec<PricingOverride>,
}

impl Pricing {
    /// Resolve prompt-length-dependent token rates.
    #[must_use]
    pub fn effective(&self, prompt_tokens: u64) -> EffectivePricing {
        let mut effective = EffectivePricing {
            prompt_per_token: self.prompt_per_token,
            completion_per_token: self.completion_per_token,
            cache_read_per_token: self.cache_read_per_token,
            cache_write_per_token: self.cache_write_per_token,
        };
        if let Some(override_) = self
            .overrides
            .iter()
            .filter(|override_| override_.min_prompt_tokens <= prompt_tokens)
            .max_by_key(|override_| override_.min_prompt_tokens)
        {
            effective.prompt_per_token = override_
                .prompt_per_token
                .unwrap_or(effective.prompt_per_token);
            effective.completion_per_token = override_
                .completion_per_token
                .unwrap_or(effective.completion_per_token);
            effective.cache_read_per_token = override_
                .cache_read_per_token
                .or(effective.cache_read_per_token);
            effective.cache_write_per_token = override_
                .cache_write_per_token
                .or(effective.cache_write_per_token);
        }
        effective
    }

    /// Estimate uncached request cost for a request shape.
    #[must_use]
    pub fn expected_cost(&self, prompt_tokens: u64, output_tokens: u64) -> f64 {
        let effective = self.effective(prompt_tokens);
        let token_cost = effective.prompt_per_token * tokens_as_f64(prompt_tokens)
            + effective.completion_per_token * tokens_as_f64(output_tokens);
        self.request + token_cost * (1.0 - self.discount.clamp(0.0, 1.0))
    }
}

/// Cross-provider reasoning effort levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ReasoningEffort {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

/// How precisely an endpoint exposes reasoning-level support.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningSupport {
    None,
    /// The provider gateway accepts normalized reasoning levels, but the upstream catalog
    /// does not expose a narrower endpoint-specific set.
    GatewayNormalized,
    Exact(BTreeSet<ReasoningEffort>),
}

impl ReasoningSupport {
    #[must_use]
    pub fn supports(&self, effort: ReasoningEffort) -> bool {
        match self {
            Self::None => effort == ReasoningEffort::None,
            Self::GatewayNormalized => true,
            Self::Exact(levels) => levels.contains(&effort),
        }
    }
}

/// Empirically observed strength of a schema-like control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintEnforcement {
    /// The control was accepted on the wire but lost even an easy adversarial constraint.
    Ineffective,
    /// Easy constraints hold, but a stronger adversarial stress constraint can still be violated.
    HintOnly,
    /// The route survived the stress probe; behavior is consistent with constrained decoding.
    Constrained,
}

/// Provider-independent function tool definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub parameters: Value,
    #[serde(default)]
    pub strict: bool,
}

/// Provider-independent tool selection policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    Required,
    None,
    Function { name: String },
}

/// Requested output schema mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResponseSchema {
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        strict: bool,
    },
}

/// Normalized endpoint capabilities used before routing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EndpointCapabilities {
    /// Parameters the adapter can actually encode on the wire.
    pub adapter_parameters: BTreeSet<String>,
    /// Provider/catalog claims. These are nominal until conformance-tested.
    pub supported_parameters: BTreeSet<String>,
    /// Parameters observed to work on this provider/model route.
    #[serde(default)]
    pub verified_parameters: BTreeSet<String>,
    /// Parameters observed to fail despite provider/catalog claims.
    #[serde(default)]
    pub rejected_parameters: BTreeSet<String>,
    /// Reasoning effort levels observed to work on this exact provider/model route.
    #[serde(default)]
    pub verified_reasoning_efforts: BTreeSet<ReasoningEffort>,
    /// Reasoning effort levels observed to fail on this exact provider/model route.
    #[serde(default)]
    pub rejected_reasoning_efforts: BTreeSet<ReasoningEffort>,
    /// Empirical strength of schema-like controls such as strict structured output and tools.
    #[serde(default)]
    pub constraint_enforcement: BTreeMap<String, ConstraintEnforcement>,
    pub reasoning: ReasoningSupport,
    pub supports_implicit_caching: bool,
}

impl EndpointCapabilities {
    /// Effective parameter support, with observations overriding nominal claims.
    #[must_use]
    pub fn supports_parameter(&self, parameter: &str) -> bool {
        if self.rejected_parameters.contains(parameter)
            || !self.adapter_parameters.contains(parameter)
        {
            return false;
        }
        self.verified_parameters.contains(parameter)
            || self.supported_parameters.contains(parameter)
    }

    /// Whether every requested parameter is currently believed to work.
    #[must_use]
    pub fn supports_parameters(&self, parameters: &BTreeSet<String>) -> bool {
        parameters
            .iter()
            .all(|parameter| self.supports_parameter(parameter))
    }

    /// Positive route-local conformance evidence for a wire parameter. Provider/catalog claims
    /// deliberately do not satisfy this check.
    #[must_use]
    pub fn verified_parameter(&self, parameter: &str) -> bool {
        self.adapter_parameters.contains(parameter)
            && self.verified_parameters.contains(parameter)
            && !self.rejected_parameters.contains(parameter)
    }

    #[must_use]
    pub fn constraint_enforcement(&self, parameter: &str) -> Option<ConstraintEnforcement> {
        self.constraint_enforcement.get(parameter).copied()
    }

    #[must_use]
    pub fn strict_output_transport(&self) -> Option<StrictOutputTransport> {
        if self.supports_parameter("structured_outputs")
            && self.constraint_enforcement("structured_outputs")
                == Some(ConstraintEnforcement::Constrained)
        {
            return Some(StrictOutputTransport::Native);
        }
        if self.supports_parameter("tools")
            && self.supports_parameter("tool_choice")
            && self.constraint_enforcement("tools") == Some(ConstraintEnforcement::Constrained)
        {
            return Some(StrictOutputTransport::ForcedTool);
        }
        None
    }

    /// Effective support for one normalized reasoning effort. Exact conformance evidence
    /// overrides nominal provider metadata. When `require_verified` is true, nominal metadata
    /// alone is insufficient.
    #[must_use]
    pub fn supports_reasoning_effort(
        &self,
        effort: ReasoningEffort,
        require_verified: bool,
    ) -> bool {
        if self.rejected_reasoning_efforts.contains(&effort) {
            return false;
        }
        if self.verified_reasoning_efforts.contains(&effort) {
            return true;
        }
        !require_verified && self.reasoning.supports(effort)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrictOutputTransport {
    Native,
    ForcedTool,
}

impl Default for EndpointCapabilities {
    fn default() -> Self {
        Self {
            adapter_parameters: BTreeSet::new(),
            supported_parameters: BTreeSet::new(),
            verified_parameters: BTreeSet::new(),
            rejected_parameters: BTreeSet::new(),
            verified_reasoning_efforts: BTreeSet::new(),
            rejected_reasoning_efforts: BTreeSet::new(),
            constraint_enforcement: BTreeMap::new(),
            reasoning: ReasoningSupport::None,
            supports_implicit_caching: false,
        }
    }
}

/// Recent endpoint performance and availability.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct EndpointStats {
    /// Provider-published first-token/gateway latency samples, seconds.
    pub latency_seconds: Percentiles,
    /// Provider-published output throughput samples, tokens/second.
    pub throughput_tokens_per_second: Percentiles,
    pub uptime_5m: Option<f64>,
    pub uptime_30m: Option<f64>,
    pub uptime_1d: Option<f64>,
}

/// Whether a route is valid for agentic workloads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgenticPolicy {
    #[default]
    Any,
    Required,
    Forbidden,
}

/// Semantic workload restrictions attached to one execution route.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadPolicy {
    #[serde(default)]
    pub agentic: AgenticPolicy,
    /// The provider requires a stable conversation/session identifier on every request.
    #[serde(default)]
    pub requires_stable_session: bool,
}

/// A normalized endpoint from any inference provider/gateway.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Endpoint {
    /// Execution surface, e.g. `openrouter` or direct `nvidia`.
    pub source: String,
    /// Opaque logical credential/account label. Never contains the secret itself.
    #[serde(default)]
    pub credential: Option<String>,
    /// Stable provider selector within the execution surface.
    pub provider_key: String,
    pub id: String,
    pub provider: String,
    pub model: String,
    pub route_class: ModelRouteClass,
    pub display_name: String,
    pub quantization: Option<String>,
    pub status: i32,
    pub context_length: Option<u64>,
    pub max_prompt_tokens: Option<u64>,
    pub max_completion_tokens: Option<u64>,
    pub pricing: Pricing,
    pub stats: EndpointStats,
    #[serde(default)]
    pub workload_policy: WorkloadPolicy,
    pub capabilities: EndpointCapabilities,
}

/// Feature constraints applied before route ranking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestRequirements {
    #[serde(default = "default_allowed_route_classes")]
    pub allowed_route_classes: BTreeSet<ModelRouteClass>,
    pub response_schema: Option<ResponseSchema>,
    pub reasoning: ReasoningEffort,
    /// Optional hard cap on reasoning tokens when the provider exposes one.
    #[serde(default)]
    pub reasoning_max_tokens: Option<u64>,
    /// Require positive conformance evidence for the requested reasoning effort.
    #[serde(default)]
    pub require_verified_reasoning: bool,
    pub required_parameters: BTreeSet<String>,
    pub minimum_uptime_30m: Option<f64>,
}

fn default_allowed_route_classes() -> BTreeSet<ModelRouteClass> {
    BTreeSet::from([
        ModelRouteClass::Standard,
        ModelRouteClass::Free,
        ModelRouteClass::Subscription,
    ])
}

impl Default for RequestRequirements {
    fn default() -> Self {
        Self {
            allowed_route_classes: default_allowed_route_classes(),
            response_schema: None,
            reasoning: ReasoningEffort::None,
            reasoning_max_tokens: None,
            require_verified_reasoning: false,
            required_parameters: BTreeSet::new(),
            minimum_uptime_30m: Some(95.0),
        }
    }
}

/// Provider-independent inference request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub model: String,
    #[serde(default)]
    pub system_prompt: Option<String>,
    pub prompt: String,
    pub expected_output_tokens: u64,
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    #[serde(default)]
    pub temperature: Option<f64>,
    /// Ask the provider to return its reasoning trace when that surface supports it.
    #[serde(default)]
    pub include_reasoning: bool,
    #[serde(default)]
    pub tools: Vec<FunctionTool>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    /// True only when the caller is actually running a coding/agent workflow.
    #[serde(default)]
    pub agentic: bool,
    /// Stable opaque conversation identifier. Required by routes whose terms/session policy demand it.
    #[serde(default)]
    pub session_id: Option<String>,
    pub requirements: RequestRequirements,
}

impl InferenceRequest {
    #[must_use]
    pub fn new(
        model: impl Into<String>,
        prompt: impl Into<String>,
        expected_output_tokens: u64,
    ) -> Self {
        Self {
            model: model.into(),
            system_prompt: None,
            prompt: prompt.into(),
            expected_output_tokens,
            prompt_tokens: None,
            max_output_tokens: None,
            temperature: None,
            include_reasoning: false,
            tools: Vec::new(),
            tool_choice: None,
            agentic: false,
            session_id: None,
            requirements: RequestRequirements::default(),
        }
    }
}

/// Routing policies independent of provider grammar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoutingStrategy {
    /// Admit endpoints costing no more than `multiplier × cheapest`, then rank by
    /// expected response time for the supplied request shape.
    FastestResponseCheap(f64),
}

/// One endpoint in a ranked route plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteCandidate {
    pub endpoint: Endpoint,
    pub expected_cost_usd: f64,
    pub expected_response_p75_seconds: Option<f64>,
    pub expected_response_p95_seconds: Option<f64>,
}

/// Ordered eligible endpoints with selection metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutePlan {
    pub prompt_tokens: u64,
    pub expected_output_tokens: u64,
    pub cheapest_cost_usd: f64,
    pub cost_ceiling_usd: f64,
    pub candidates: Vec<RouteCandidate>,
}

/// Automatic retry behavior for transport/gateway failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(150),
            max_backoff: Duration::from_secs(2),
        }
    }
}

/// Normalized token accounting reported by an upstream response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
}

impl TokenUsage {
    pub(crate) fn from_openai_compatible(raw: &Value) -> Self {
        let usage = raw.get("usage").unwrap_or(&Value::Null);
        Self {
            input_tokens: usage
                .get("prompt_tokens")
                .or_else(|| usage.get("input_tokens"))
                .and_then(Value::as_u64),
            output_tokens: usage
                .get("completion_tokens")
                .or_else(|| usage.get("output_tokens"))
                .and_then(Value::as_u64),
            total_tokens: usage.get("total_tokens").and_then(Value::as_u64),
            reasoning_tokens: usage
                .pointer("/completion_tokens_details/reasoning_tokens")
                .or_else(|| usage.pointer("/output_tokens_details/reasoning_tokens"))
                .and_then(Value::as_u64),
        }
    }
}

/// Normalized inference response. `raw` preserves provider-specific response metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub content: String,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub usage_cost_usd: Option<f64>,
    #[serde(default)]
    pub usage: TokenUsage,
    /// Recognized, normalized quota measurements from response headers. No raw headers are kept.
    #[serde(default)]
    pub quota_observations: Vec<crate::QuotaObservation>,
    pub raw: Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalizes_openai_style_usage_and_reasoning_tokens() {
        let raw = json!({
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 33,
                "total_tokens": 44,
                "completion_tokens_details": {"reasoning_tokens": 29}
            }
        });
        let usage = TokenUsage::from_openai_compatible(&raw);
        assert_eq!(usage.input_tokens, Some(11));
        assert_eq!(usage.output_tokens, Some(33));
        assert_eq!(usage.total_tokens, Some(44));
        assert_eq!(usage.reasoning_tokens, Some(29));
    }

    #[test]
    fn normalizes_responses_style_token_names() {
        let raw = json!({
            "usage": {
                "input_tokens": 5,
                "output_tokens": 7,
                "total_tokens": 12,
                "output_tokens_details": {"reasoning_tokens": 3}
            }
        });
        let usage = TokenUsage::from_openai_compatible(&raw);
        assert_eq!(usage.input_tokens, Some(5));
        assert_eq!(usage.output_tokens, Some(7));
        assert_eq!(usage.total_tokens, Some(12));
        assert_eq!(usage.reasoning_tokens, Some(3));
    }
}
