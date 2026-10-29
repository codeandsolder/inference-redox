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
    /// The route survived the stress probe. This establishes behavioral enforcement, not the
    /// mechanism: grammar masking, provider-side prompt injection, or post-processing may differ.
    #[serde(rename = "enforced", alias = "constrained")]
    Enforced,
}

/// How much empirical JSON-Schema evidence routing requires before trying an endpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchemaEvidencePolicy {
    /// Require positive endpoint-local evidence for every required schema feature.
    #[default]
    Proven,
    /// Allow unknown schema behavior, but never retry features already observed to fail.
    /// Intended for retryable low-priority work where useful inference can double as conformance
    /// sampling.
    Learn,
}

/// JSON-Schema features whose support is probed independently after base enforcement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchemaFeature {
    OneOf,
    AnyOf,
    AllOf,
    Enum,
    NestedObject,
    ArrayItems,
    ArrayBounds,
    TuplePrefixItems,
    NumericBounds,
    MultipleOf,
    StringPattern,
    StringBounds,
    NullableType,
    DefsRef,
    RecursiveRef,
    Contains,
    UniqueItems,
    Not,
    Conditional,
    DependentRequired,
    PropertyNames,
    ObjectBounds,
    AdditionalPropertiesSchema,
    PatternProperties,
    DependentSchemas,
    UnevaluatedProperties,
    Format,
    BooleanType,
    NumberType,
    NullType,
    ArrayType,
    TypeUnion,
    OptionalProperties,
    OpenObject,
    RootArray,
    RootPrimitive,
    RootComposition,
    RootRef,
    UnevaluatedItems,
    LegacyTupleItems,
    /// Synthetic combined schema used to emulate a complete assistant tool turn.
    ToolTurnEnvelope,
    /// Synthetic behavior probe for multiple calls in one schema-emulated assistant turn.
    ToolTurnMultiple,
    UnknownKeyword,
}

impl SchemaFeature {
    pub const ALL: [Self; 40] = [
        Self::OneOf,
        Self::AnyOf,
        Self::AllOf,
        Self::Enum,
        Self::NestedObject,
        Self::ArrayItems,
        Self::ArrayBounds,
        Self::TuplePrefixItems,
        Self::NumericBounds,
        Self::MultipleOf,
        Self::StringPattern,
        Self::StringBounds,
        Self::NullableType,
        Self::DefsRef,
        Self::RecursiveRef,
        Self::Contains,
        Self::UniqueItems,
        Self::Not,
        Self::Conditional,
        Self::DependentRequired,
        Self::PropertyNames,
        Self::ObjectBounds,
        Self::AdditionalPropertiesSchema,
        Self::PatternProperties,
        Self::DependentSchemas,
        Self::UnevaluatedProperties,
        Self::Format,
        Self::BooleanType,
        Self::NumberType,
        Self::NullType,
        Self::ArrayType,
        Self::TypeUnion,
        Self::OptionalProperties,
        Self::OpenObject,
        Self::RootArray,
        Self::RootPrimitive,
        Self::RootComposition,
        Self::RootRef,
        Self::UnevaluatedItems,
        Self::LegacyTupleItems,
    ];
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OneOf => "one_of",
            Self::AnyOf => "any_of",
            Self::AllOf => "all_of",
            Self::Enum => "enum",
            Self::NestedObject => "nested_object",
            Self::ArrayItems => "array_items",
            Self::ArrayBounds => "array_bounds",
            Self::TuplePrefixItems => "tuple_prefix_items",
            Self::NumericBounds => "numeric_bounds",
            Self::MultipleOf => "multiple_of",
            Self::StringPattern => "string_pattern",
            Self::StringBounds => "string_bounds",
            Self::NullableType => "nullable_type",
            Self::DefsRef => "defs_ref",
            Self::RecursiveRef => "recursive_ref",
            Self::Contains => "contains",
            Self::UniqueItems => "unique_items",
            Self::Not => "not",
            Self::Conditional => "conditional",
            Self::DependentRequired => "dependent_required",
            Self::PropertyNames => "property_names",
            Self::ObjectBounds => "object_bounds",
            Self::AdditionalPropertiesSchema => "additional_properties_schema",
            Self::PatternProperties => "pattern_properties",
            Self::DependentSchemas => "dependent_schemas",
            Self::UnevaluatedProperties => "unevaluated_properties",
            Self::Format => "format",
            Self::BooleanType => "boolean_type",
            Self::NumberType => "number_type",
            Self::NullType => "null_type",
            Self::ArrayType => "array_type",
            Self::TypeUnion => "type_union",
            Self::OptionalProperties => "optional_properties",
            Self::OpenObject => "open_object",
            Self::RootArray => "root_array",
            Self::RootPrimitive => "root_primitive",
            Self::RootComposition => "root_composition",
            Self::RootRef => "root_ref",
            Self::UnevaluatedItems => "unevaluated_items",
            Self::LegacyTupleItems => "legacy_tuple_items",
            Self::ToolTurnEnvelope => "tool_turn_envelope",
            Self::ToolTurnMultiple => "tool_turn_multiple",
            Self::UnknownKeyword => "unknown_keyword",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseSchemaFeatureError;

impl std::fmt::Display for ParseSchemaFeatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("unknown JSON-Schema feature")
    }
}

impl std::error::Error for ParseSchemaFeatureError {}

impl std::str::FromStr for SchemaFeature {
    type Err = ParseSchemaFeatureError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value == Self::ToolTurnEnvelope.as_str() {
            return Ok(Self::ToolTurnEnvelope);
        }
        if value == Self::ToolTurnMultiple.as_str() {
            return Ok(Self::ToolTurnMultiple);
        }
        Self::ALL
            .into_iter()
            .find(|feature| feature.as_str() == value)
            .ok_or(ParseSchemaFeatureError)
    }
}

pub(crate) const STRUCTURED_TOOL_TRANSPORT_INSTRUCTION: &str = "Represent all tool calls only through the structured response contract using text_response and tool_calls; do not use any native tool-calling channel.";

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

/// One normalized function call produced by a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: Option<String>,
    pub name: String,
    pub arguments: Value,
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
    /// JSON-Schema keywords/compositions observed to work for each schema-bearing transport.
    #[serde(default)]
    pub verified_schema_features: BTreeMap<String, BTreeSet<SchemaFeature>>,
    /// JSON-Schema keywords/compositions observed to fail for each schema-bearing transport.
    #[serde(default)]
    pub rejected_schema_features: BTreeMap<String, BTreeSet<SchemaFeature>>,
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

    /// Whether every non-baseline schema feature needed by a generated/application schema has
    /// positive evidence on this exact transport. Unknown and explicitly rejected features fail
    /// closed. Basic object/type/required/const/additionalProperties behavior is established by
    /// the base enforcement probe and therefore is not represented here.
    #[must_use]
    pub fn verified_schema_feature_set(
        &self,
        parameter: &str,
        features: &BTreeSet<SchemaFeature>,
    ) -> bool {
        let verified = self.verified_schema_features.get(parameter);
        let rejected = self.rejected_schema_features.get(parameter);
        features.iter().all(|feature| {
            verified.is_some_and(|set| set.contains(feature))
                && !rejected.is_some_and(|set| set.contains(feature))
        })
    }

    #[must_use]
    pub fn strict_output_transport_for_schema(
        &self,
        schema: &Value,
    ) -> Option<StrictOutputTransport> {
        self.strict_output_transport_for_schema_with_policy(schema, SchemaEvidencePolicy::Proven)
    }

    /// Select strict structured-output transport under the caller's empirical-evidence policy.
    #[must_use]
    pub fn strict_output_transport_for_schema_with_policy(
        &self,
        schema: &Value,
        policy: SchemaEvidencePolicy,
    ) -> Option<StrictOutputTransport> {
        let lowered = crate::lower_json_schema(schema);
        self.strict_output_transport_for_lowered_schema_with_policy(&lowered, policy)
    }

    pub(crate) fn strict_output_transport_for_lowered_schema_with_policy(
        &self,
        lowered_schema: &Value,
        policy: SchemaEvidencePolicy,
    ) -> Option<StrictOutputTransport> {
        let native_features = schema_feature_requirements(lowered_schema);
        if self.supports_parameter("structured_outputs")
            && self.schema_transport_evidence_allows("structured_outputs", &native_features, policy)
        {
            return Some(StrictOutputTransport::Native);
        }
        let (tool_schema, _) = forced_tool_parameter_schema_from_lowered(lowered_schema);
        let tool_features = schema_feature_requirements(&tool_schema);
        if self.supports_parameter("tools")
            && self.supports_parameter("tool_choice")
            && self.schema_transport_evidence_allows("tools", &tool_features, policy)
        {
            return Some(StrictOutputTransport::ForcedTool);
        }
        None
    }

    fn schema_transport_evidence_allows(
        &self,
        parameter: &str,
        features: &BTreeSet<SchemaFeature>,
        policy: SchemaEvidencePolicy,
    ) -> bool {
        match policy {
            SchemaEvidencePolicy::Proven => {
                self.constraint_enforcement(parameter) == Some(ConstraintEnforcement::Enforced)
                    && self.verified_schema_feature_set(parameter, features)
            }
            SchemaEvidencePolicy::Learn => {
                if matches!(
                    self.constraint_enforcement(parameter),
                    Some(ConstraintEnforcement::Ineffective | ConstraintEnforcement::HintOnly)
                ) {
                    return false;
                }
                let rejected = self.rejected_schema_features.get(parameter);
                !features
                    .iter()
                    .any(|feature| rejected.is_some_and(|set| set.contains(feature)))
            }
        }
    }

    /// Select a semantics-preserving transport for a tool request. Native tools are preferred.
    /// Structured output may emulate exact, required, and auto choices only when the endpoint has
    /// enough evidence under the request's schema policy for the generated envelope.
    #[must_use]
    pub fn tool_transport(&self, request: &InferenceRequest) -> Option<ToolTransport> {
        if request.tools.is_empty() || matches!(request.tool_choice, Some(ToolChoice::None)) {
            return None;
        }
        let policy = request.requirements.schema_evidence;
        let probe_native = request.requirements.required_parameters.contains("tools");
        let parameter_allows = |parameter: &str| match policy {
            SchemaEvidencePolicy::Proven => self.verified_parameter(parameter),
            SchemaEvidencePolicy::Learn => self.supports_parameter(parameter),
        };
        let native_choice_ok = match request.tool_choice.as_ref() {
            Some(ToolChoice::Required | ToolChoice::Function { .. }) => {
                parameter_allows("tool_choice")
                    || (probe_native && self.adapter_parameters.contains("tool_choice"))
            }
            Some(ToolChoice::Auto) | None => true,
            Some(ToolChoice::None) => return None,
        };
        let native_parallel_ok = request.parallel_tool_calls.is_none()
            || parameter_allows("parallel_tool_calls")
            || (probe_native && self.adapter_parameters.contains("parallel_tool_calls"));
        let strict_features = request.tools.iter().filter(|tool| tool.strict).fold(
            BTreeSet::new(),
            |mut features, tool| {
                let lowered = crate::lower_json_schema(&tool.parameters);
                features.extend(schema_feature_requirements(&lowered));
                features
            },
        );
        let native_schema_ok = strict_features.is_empty()
            || probe_native
            || self.schema_transport_evidence_allows("tools", &strict_features, policy);
        if (parameter_allows("tools")
            || (probe_native && self.adapter_parameters.contains("tools")))
            && native_choice_ok
            && native_parallel_ok
            && native_schema_ok
        {
            return Some(ToolTransport::Native);
        }
        if request.requirements.response_schema.is_none()
            && request
                .lowered_structured_tool_response_schema()
                .is_some_and(|(_, features)| {
                    self.supports_parameter("structured_outputs")
                        && self.schema_transport_evidence_allows(
                            "structured_outputs",
                            &features,
                            policy,
                        )
                })
        {
            return Some(ToolTransport::StructuredOutput);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolTransport {
    Native,
    StructuredOutput,
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
            verified_schema_features: BTreeMap::new(),
            rejected_schema_features: BTreeMap::new(),
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
    /// Whether strict structured output must be pre-proven or may learn from useful traffic.
    #[serde(default)]
    pub schema_evidence: SchemaEvidencePolicy,
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
            schema_evidence: SchemaEvidencePolicy::Proven,
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
    /// Explicit parallel-call policy. `None` preserves the provider/default tool semantics.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
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
            parallel_tool_calls: None,
            agentic: false,
            session_id: None,
            requirements: RequestRequirements::default(),
        }
    }

    /// Return the single function that the request requires unconditionally.
    #[must_use]
    pub fn exact_required_tool(&self) -> Option<&FunctionTool> {
        match self.tool_choice.as_ref() {
            Some(ToolChoice::Function { name }) => {
                self.tools.iter().find(|tool| tool.name == *name)
            }
            Some(ToolChoice::Required) if self.tools.len() == 1 => self.tools.first(),
            _ => None,
        }
    }

    /// Build the schema used to emulate native tool turns through structured output. The envelope
    /// keeps text and calls independent, so `auto` can return text, calls, or both; an array keeps
    /// parallel/multiple calls representable.
    #[must_use]
    pub fn structured_tool_response_schema(&self) -> Option<(Value, BTreeSet<SchemaFeature>)> {
        if self.tools.is_empty() || matches!(self.tool_choice, Some(ToolChoice::None)) {
            return None;
        }
        let selected: Vec<&FunctionTool> = match self.tool_choice.as_ref() {
            Some(ToolChoice::Function { name }) => {
                vec![self.tools.iter().find(|tool| tool.name == *name)?]
            }
            Some(ToolChoice::Auto | ToolChoice::Required) | None => self.tools.iter().collect(),
            Some(ToolChoice::None) => return None,
        };
        if selected.is_empty() {
            return None;
        }
        let branches: Vec<Value> = selected
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": {"const": tool.name},
                        "arguments": tool.parameters
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": false
                })
            })
            .collect();
        let items = if branches.len() == 1 {
            branches.into_iter().next()?
        } else {
            serde_json::json!({"oneOf": branches})
        };
        let mut calls = serde_json::json!({"type": "array", "items": items});
        let calls_object = calls.as_object_mut()?;
        if matches!(
            self.tool_choice,
            Some(ToolChoice::Required | ToolChoice::Function { .. })
        ) {
            calls_object.insert("minItems".to_owned(), Value::from(1_u64));
        }
        if self.parallel_tool_calls == Some(false) {
            calls_object.insert("maxItems".to_owned(), Value::from(1_u64));
        }
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "text_response": {"type": "string"},
                "tool_calls": calls
            },
            "required": ["text_response", "tool_calls"],
            "additionalProperties": false
        });
        let mut features = schema_feature_requirements(&schema);
        features.insert(SchemaFeature::NestedObject);
        features.insert(SchemaFeature::ToolTurnEnvelope);
        if self.parallel_tool_calls == Some(true) {
            features.insert(SchemaFeature::ToolTurnMultiple);
        }
        Some((schema, features))
    }

    /// Canonicalize a generated tool-turn envelope without introducing new feature classes.
    #[must_use]
    pub fn lowered_structured_tool_response_schema(
        &self,
    ) -> Option<(Value, BTreeSet<SchemaFeature>)> {
        let (schema, _) = self.structured_tool_response_schema()?;
        let lowered = crate::lower_json_schema(&schema);
        let mut features = schema_feature_requirements(&lowered);
        features.insert(SchemaFeature::NestedObject);
        features.insert(SchemaFeature::ToolTurnEnvelope);
        if self.parallel_tool_calls == Some(true) {
            features.insert(SchemaFeature::ToolTurnMultiple);
        }
        Some((lowered, features))
    }
}

/// Conservatively collect non-baseline JSON-Schema features used by an application/generated
/// schema. Basic object/type/required/const/additionalProperties(bool) behavior is established by
/// the base enforcement probe. Any unrecognized validation keyword inserts `UnknownKeyword`, which
/// can never be verified and therefore makes cross-transport translation fail closed.
#[must_use]
pub fn schema_feature_requirements(schema: &Value) -> BTreeSet<SchemaFeature> {
    let mut features = BTreeSet::new();
    collect_root_schema_features(schema, &mut features);
    collect_schema_features(schema, &mut features);
    if schema_contains_recursive_ref(schema) {
        features.insert(SchemaFeature::RecursiveRef);
    }
    features
}

fn collect_root_schema_features(schema: &Value, features: &mut BTreeSet<SchemaFeature>) {
    let Some(map) = schema.as_object() else {
        features.insert(SchemaFeature::RootPrimitive);
        return;
    };
    if map.contains_key("$ref") {
        features.insert(SchemaFeature::RootRef);
    }
    if ["oneOf", "anyOf", "allOf"]
        .iter()
        .any(|keyword| map.contains_key(*keyword))
    {
        features.insert(SchemaFeature::RootComposition);
    }
    match map.get("type").and_then(Value::as_str) {
        Some("array") => {
            features.insert(SchemaFeature::RootArray);
        }
        Some("object") | None => {}
        Some(_) => {
            features.insert(SchemaFeature::RootPrimitive);
        }
    }
}

fn collect_object_shape_features(
    map: &serde_json::Map<String, Value>,
    features: &mut BTreeSet<SchemaFeature>,
) {
    let is_object = map.get("type").and_then(Value::as_str) == Some("object")
        || map.contains_key("properties")
        || map.contains_key("patternProperties")
        || map.contains_key("dependentSchemas");
    if !is_object {
        return;
    }
    if map
        .get("additionalProperties")
        .is_none_or(|value| value == &Value::Bool(true))
    {
        features.insert(SchemaFeature::OpenObject);
    }
    let Some(properties) = map.get("properties").and_then(Value::as_object) else {
        return;
    };
    let required: BTreeSet<&str> = map
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    if properties
        .keys()
        .any(|property| !required.contains(property.as_str()))
    {
        features.insert(SchemaFeature::OptionalProperties);
    }
}

fn collect_schema_features(schema: &Value, features: &mut BTreeSet<SchemaFeature>) {
    let Some(map) = schema.as_object() else {
        // Boolean schemas are a normal JSON-Schema construct. Their enclosing keyword determines
        // whether additional feature evidence is needed.
        return;
    };
    collect_object_shape_features(map, features);
    for (keyword, value) in map {
        collect_schema_keyword(keyword, value, features);
    }
}

fn collect_schema_keyword(keyword: &str, value: &Value, features: &mut BTreeSet<SchemaFeature>) {
    match keyword {
        // Baseline validation vocabulary proved by the base strict probe.
        "type" => collect_type_features(value, features),
        "required" | "const" | "additionalProperties" if !value.is_object() => {}
        // Non-validating annotations/meta keywords do not need behavioral evidence.
        "$schema" | "$id" | "$comment" | "title" | "description" | "default" | "examples"
        | "deprecated" | "readOnly" | "writeOnly" => {}
        "properties" => collect_schema_map(value, features, true),
        "$defs" => {
            features.insert(SchemaFeature::DefsRef);
            collect_schema_map(value, features, false);
        }
        "$ref" => {
            features.insert(SchemaFeature::DefsRef);
        }
        "oneOf" => collect_schema_array(value, features, SchemaFeature::OneOf),
        "anyOf" => collect_schema_array(value, features, SchemaFeature::AnyOf),
        "allOf" => collect_schema_array(value, features, SchemaFeature::AllOf),
        "enum" => {
            features.insert(SchemaFeature::Enum);
        }
        "items" if value.is_array() => {
            features.insert(SchemaFeature::LegacyTupleItems);
            if let Some(items) = value.as_array() {
                for item in items {
                    collect_schema_features(item, features);
                }
            }
        }
        "items" => collect_schema_child(value, features, SchemaFeature::ArrayItems),
        "additionalItems" => {
            features.insert(SchemaFeature::LegacyTupleItems);
            collect_schema_features(value, features);
        }
        "prefixItems" => {
            collect_schema_array(value, features, SchemaFeature::TuplePrefixItems);
        }
        "minItems" | "maxItems" => {
            features.insert(SchemaFeature::ArrayBounds);
        }
        "contains" => collect_schema_child(value, features, SchemaFeature::Contains),
        "minContains" | "maxContains" => {
            features.insert(SchemaFeature::Contains);
        }
        "uniqueItems" => {
            features.insert(SchemaFeature::UniqueItems);
        }
        "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" => {
            features.insert(SchemaFeature::NumericBounds);
        }
        "multipleOf" => {
            features.insert(SchemaFeature::MultipleOf);
        }
        "pattern" => {
            features.insert(SchemaFeature::StringPattern);
        }
        "minLength" | "maxLength" => {
            features.insert(SchemaFeature::StringBounds);
        }
        "format" => {
            features.insert(SchemaFeature::Format);
        }
        "not" => collect_schema_child(value, features, SchemaFeature::Not),
        "if" | "then" | "else" => {
            collect_schema_child(value, features, SchemaFeature::Conditional);
        }
        "dependentRequired" => {
            features.insert(SchemaFeature::DependentRequired);
        }
        "dependentSchemas" => {
            features.insert(SchemaFeature::DependentSchemas);
            collect_schema_map(value, features, false);
        }
        "propertyNames" => collect_schema_child(value, features, SchemaFeature::PropertyNames),
        "minProperties" | "maxProperties" => {
            features.insert(SchemaFeature::ObjectBounds);
        }
        "patternProperties" => {
            features.insert(SchemaFeature::PatternProperties);
            collect_schema_map(value, features, false);
        }
        "unevaluatedProperties" => {
            features.insert(SchemaFeature::UnevaluatedProperties);
            collect_schema_features(value, features);
        }
        "unevaluatedItems" => {
            features.insert(SchemaFeature::UnevaluatedItems);
            collect_schema_features(value, features);
        }
        "additionalProperties" => {
            features.insert(SchemaFeature::AdditionalPropertiesSchema);
            collect_schema_features(value, features);
        }
        _ => {
            features.insert(SchemaFeature::UnknownKeyword);
        }
    }
}

fn collect_type_features(value: &Value, features: &mut BTreeSet<SchemaFeature>) {
    if let Some(kind) = value.as_str() {
        match kind {
            "object" | "string" | "integer" => {}
            "boolean" => {
                features.insert(SchemaFeature::BooleanType);
            }
            "number" => {
                features.insert(SchemaFeature::NumberType);
            }
            "null" => {
                features.insert(SchemaFeature::NullType);
            }
            "array" => {
                features.insert(SchemaFeature::ArrayType);
            }
            _ => {
                features.insert(SchemaFeature::UnknownKeyword);
            }
        }
        return;
    }
    let Some(types) = value.as_array() else {
        features.insert(SchemaFeature::UnknownKeyword);
        return;
    };
    let names: BTreeSet<&str> = types.iter().filter_map(Value::as_str).collect();
    if names.len() == 2 && names.contains("null") {
        features.insert(SchemaFeature::NullableType);
    } else {
        features.insert(SchemaFeature::TypeUnion);
    }
    for kind in names {
        collect_type_features(&Value::String(kind.to_owned()), features);
    }
}

fn collect_schema_child(
    value: &Value,
    features: &mut BTreeSet<SchemaFeature>,
    feature: SchemaFeature,
) {
    features.insert(feature);
    collect_schema_features(value, features);
}

fn collect_schema_array(
    value: &Value,
    features: &mut BTreeSet<SchemaFeature>,
    feature: SchemaFeature,
) {
    features.insert(feature);
    if let Some(items) = value.as_array() {
        for item in items {
            collect_schema_features(item, features);
        }
    }
}

fn collect_schema_map(value: &Value, features: &mut BTreeSet<SchemaFeature>, nested: bool) {
    let Some(map) = value.as_object() else {
        return;
    };
    if nested
        && map
            .values()
            .any(|child| child.get("type").and_then(Value::as_str) == Some("object"))
    {
        features.insert(SchemaFeature::NestedObject);
    }
    for child in map.values() {
        collect_schema_features(child, features);
    }
}

fn schema_contains_recursive_ref(root: &Value) -> bool {
    let Some(defs) = root.get("$defs").and_then(Value::as_object) else {
        return false;
    };
    defs.iter().any(|(name, definition)| {
        let target = format!("#/$defs/{name}");
        schema_has_ref(definition, &target)
    })
}

fn schema_has_ref(value: &Value, target: &str) -> bool {
    match value {
        Value::Object(map) => {
            map.get("$ref").and_then(Value::as_str) == Some(target)
                || map.values().any(|child| schema_has_ref(child, target))
        }
        Value::Array(items) => items.iter().any(|child| schema_has_ref(child, target)),
        _ => false,
    }
}

/// Function-tool parameters must be rooted at an object on the normalized tool surfaces. Preserve
/// object-root response schemas directly; wrap every other response shape under `value` and unwrap
/// it after execution so the application-visible response keeps its original schema semantics.
pub(crate) fn forced_tool_parameter_schema(schema: &Value) -> (Value, bool) {
    let lowered = crate::lower_json_schema(schema);
    forced_tool_parameter_schema_from_lowered(&lowered)
}

pub(crate) fn forced_tool_parameter_schema_from_lowered(lowered: &Value) -> (Value, bool) {
    if lowered
        .as_object()
        .is_some_and(|map| map.get("type").and_then(Value::as_str) == Some("object"))
    {
        return (lowered.clone(), false);
    }
    (
        serde_json::json!({
            "type": "object",
            "properties": {"value": lowered},
            "required": ["value"],
            "additionalProperties": false
        }),
        true,
    )
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
    pub attempt_timeout: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(150),
            max_backoff: Duration::from_secs(2),
            attempt_timeout: Duration::from_secs(120),
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
    /// Provider-independent normalized function calls. Provider-specific raw payload is retained.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Narrow local coercions applied after provider output and before returning the response.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_repairs: Vec<crate::OutputRepair>,
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
    fn legacy_constrained_enforcement_cache_deserializes_as_enforced()
    -> Result<(), serde_json::Error> {
        assert_eq!(
            serde_json::from_str::<ConstraintEnforcement>(r#""constrained""#)?,
            ConstraintEnforcement::Enforced
        );
        assert_eq!(
            serde_json::to_string(&ConstraintEnforcement::Enforced)?,
            r#""enforced""#
        );
        Ok(())
    }

    #[test]
    fn schema_feature_requirements_find_generated_tool_envelope_features()
    -> Result<(), &'static str> {
        let mut request = InferenceRequest::new("model", "prompt", 8);
        request.tools.extend([
            FunctionTool {
                name: "a".to_owned(),
                description: None,
                parameters: json!({"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}),
                strict: true,
            },
            FunctionTool {
                name: "b".to_owned(),
                description: None,
                parameters: json!({"type":"object","properties":{"y":{"type":"string"}},"required":["y"],"additionalProperties":false}),
                strict: true,
            },
        ]);
        request.tool_choice = Some(ToolChoice::Required);
        request.parallel_tool_calls = Some(false);
        let (_, features) = request
            .structured_tool_response_schema()
            .ok_or("tool envelope should compile")?;
        assert!(features.contains(&SchemaFeature::OneOf));
        assert!(features.contains(&SchemaFeature::NestedObject));
        assert!(features.contains(&SchemaFeature::ArrayItems));
        assert!(features.contains(&SchemaFeature::ArrayBounds));
        Ok(())
    }

    #[test]
    fn schema_feature_requirements_detect_recursive_refs() {
        let schema = json!({
            "$defs": {
                "node": {
                    "oneOf": [
                        {"type":"null"},
                        {"type":"object","properties":{"child":{"$ref":"#/$defs/node"}},"required":["child"]}
                    ]
                }
            },
            "$ref":"#/$defs/node"
        });
        let features = schema_feature_requirements(&schema);
        assert!(features.contains(&SchemaFeature::DefsRef));
        assert!(features.contains(&SchemaFeature::RecursiveRef));
        assert!(features.contains(&SchemaFeature::OneOf));
    }

    #[test]
    fn schema_feature_requirements_fail_closed_on_unknown_validation_keyword() {
        let schema = json!({
            "type":"object",
            "properties":{"value":{"type":"string","contentEncoding":"base64"}},
            "required":["value"],
            "additionalProperties":false
        });
        assert!(schema_feature_requirements(&schema).contains(&SchemaFeature::UnknownKeyword));
    }

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
