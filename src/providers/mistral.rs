use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use super::openai_compatible::{base_chat_body, execute_json_with_retry};
use crate::{
    Endpoint, EndpointCapabilities, EndpointStats, FunctionTool, InferenceProvider,
    InferenceRequest, InferenceResponse, ModelCatalog, ModelRouteClass, NativeModelRecord, Pricing,
    ProviderError, ProviderInventory, QuotaLedger, QuotaMetric, QuotaResource, QuotaResourceId,
    QuotaRouteKey, QuotaScope, ReasoningEffort, ReasoningSupport, ResponseSchema, RetryPolicy,
    RoutePlan, TokenUsage, ToolCall, ToolChoice,
};

const MISTRAL_BASE_URL: &str = "https://api.mistral.ai/v1";
const CODESTRAL_BASE_URL: &str = "https://codestral.mistral.ai/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MistralSurface {
    Studio,
    Codestral,
}

impl MistralSurface {
    const fn source(self) -> &'static str {
        match self {
            Self::Studio => "mistral",
            Self::Codestral => "codestral",
        }
    }

    const fn provider_name(self) -> &'static str {
        match self {
            Self::Studio => "Mistral Studio",
            Self::Codestral => "Codestral Dedicated",
        }
    }

    const fn route_class(self) -> ModelRouteClass {
        match self {
            Self::Studio => ModelRouteClass::Standard,
            Self::Codestral => ModelRouteClass::Free,
        }
    }
}

/// Native Mistral chat-completions adapter. `new` targets Studio; `codestral` targets the
/// dedicated Codestral endpoint and keeps its independent credential/quota surface distinct.
#[derive(Debug, Clone)]
pub struct MistralCloud {
    client: Client,
    api_key: String,
    base_url: String,
    surface: MistralSurface,
    model_cache: Arc<Mutex<Option<CachedMistralModels>>>,
    catalog_ttl: Duration,
    quota_pool: Option<String>,
}

#[derive(Debug, Clone)]
struct CachedMistralModels {
    fetched_at: Instant,
    models: Vec<NativeModelRecord>,
}

impl MistralCloud {
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::for_surface(api_key.into(), MistralSurface::Studio, MISTRAL_BASE_URL)
    }

    #[must_use]
    pub fn codestral(api_key: impl Into<String>) -> Self {
        Self::for_surface(
            api_key.into(),
            MistralSurface::Codestral,
            CODESTRAL_BASE_URL,
        )
    }

    fn for_surface(api_key: String, surface: MistralSurface, base_url: &str) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: base_url.to_owned(),
            surface,
            model_cache: Arc::new(Mutex::new(None)),
            catalog_ttl: Duration::from_secs(60),
            quota_pool: None,
        }
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        base_url
            .into()
            .trim_end_matches('/')
            .clone_into(&mut self.base_url);
        self
    }

    #[must_use]
    pub fn with_catalog_ttl(mut self, ttl: Duration) -> Self {
        self.catalog_ttl = ttl;
        self
    }

    #[must_use]
    pub fn with_quota_pool(mut self, quota_pool: impl Into<String>) -> Self {
        self.quota_pool = Some(quota_pool.into());
        self
    }

    /// Fetch and reconcile the provider-native model inventory.
    ///
    /// # Errors
    /// Returns a catalog error when Mistral inventory cannot be fetched or decoded.
    pub async fn inventory(
        &self,
        catalog: &ModelCatalog,
    ) -> Result<ProviderInventory, ProviderError> {
        Ok(ProviderInventory::reconcile(
            self.surface.source(),
            self.native_models().await?,
            catalog,
        ))
    }

    async fn native_models(&self) -> Result<Vec<NativeModelRecord>, ProviderError> {
        let mut cache = self.model_cache.lock().await;
        if let Some(cached) = cache.as_ref()
            && cached.fetched_at.elapsed() <= self.catalog_ttl
        {
            return Ok(cached.models.clone());
        }
        let stale = cache.as_ref().map(|cached| cached.models.clone());
        drop(cache);

        match self.fetch_native_models().await {
            Ok(models) => {
                cache = self.model_cache.lock().await;
                *cache = Some(CachedMistralModels {
                    fetched_at: Instant::now(),
                    models: models.clone(),
                });
                Ok(models)
            }
            Err(_) if self.surface == MistralSurface::Codestral => {
                Ok(stale.unwrap_or_else(|| vec![Self::codestral_fallback_record()]))
            }
            Err(error) => stale.ok_or(error),
        }
    }

    async fn fetch_native_models(&self) -> Result<Vec<NativeModelRecord>, ProviderError> {
        let response = self
            .client
            .get(format!("{}/models", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|error| ProviderError::Catalog(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            return Err(ProviderError::Catalog(format!("HTTP {status}")));
        }
        let envelope: MistralModelEnvelope = response
            .json()
            .await
            .map_err(|error| ProviderError::Catalog(error.to_string()))?;
        let models = envelope
            .data
            .into_iter()
            .filter_map(|model| model.normalize(self.surface))
            .collect::<Vec<_>>();
        if self.surface == MistralSurface::Codestral && models.is_empty() {
            Ok(vec![Self::codestral_fallback_record()])
        } else {
            Ok(models)
        }
    }

    fn codestral_fallback_record() -> NativeModelRecord {
        let capabilities = MistralModelCapabilities {
            completion_chat: Some(true),
            function_calling: Some(true),
            reasoning: Some(false),
            completion_fim: Some(true),
        };
        NativeModelRecord {
            id: "codestral-latest".to_owned(),
            display_name: Some("Codestral".to_owned()),
            owned_by: Some("mistralai".to_owned()),
            created_unix: None,
            context_length: Some(131_072),
            max_completion_tokens: None,
            claimed_parameters: Self::claimed_parameters(&capabilities),
            pricing: Some(Pricing::default()),
            route_class: Some(ModelRouteClass::Free),
            canonical_hint: Some("mistralai/codestral-2508".to_owned()),
            virtual_route: false,
        }
    }

    fn claimed_parameters(capabilities: &MistralModelCapabilities) -> BTreeSet<String> {
        let mut parameters = BTreeSet::new();
        if capabilities.completion_chat.unwrap_or(false) {
            parameters.extend(
                [
                    "chat_completions",
                    "system_role",
                    "max_tokens",
                    "temperature",
                    "response_format",
                    "structured_outputs",
                ]
                .into_iter()
                .map(str::to_owned),
            );
        }
        if capabilities.function_calling.unwrap_or(false) {
            parameters.extend(
                ["tools", "tool_choice", "parallel_tool_calls"]
                    .into_iter()
                    .map(str::to_owned),
            );
        }
        if capabilities.reasoning.unwrap_or(false) {
            parameters.extend(
                ["reasoning", "reasoning_effort"]
                    .into_iter()
                    .map(str::to_owned),
            );
        }
        if capabilities.completion_fim.unwrap_or(false) {
            parameters.insert("fim".to_owned());
        }
        parameters
    }

    fn studio_pricing(model: &str) -> Option<Pricing> {
        let lower = model.to_ascii_lowercase();
        let (input, cached, output) = if lower.contains("mistral-large-4") {
            (0.68, 0.07, 2.09)
        } else if lower.contains("mistral-large-2512") || lower.contains("mistral-large-3") {
            (0.5, 0.05, 1.5)
        } else if lower.contains("mistral-medium") {
            (1.5, 0.15, 7.5)
        } else if lower.contains("mistral-small-2603") || lower.contains("mistral-small-4") {
            (0.15, 0.015, 0.6)
        } else if lower.contains("ministral-14b") {
            (0.2, 0.02, 0.2)
        } else if lower.contains("ministral-8b") {
            (0.15, 0.015, 0.15)
        } else if lower.contains("ministral-3b") {
            (0.1, 0.01, 0.1)
        } else if lower.contains("codestral") && !lower.contains("embed") {
            (0.3, 0.03, 0.9)
        } else {
            return None;
        };
        Some(Pricing {
            prompt_per_token: input / 1_000_000.0,
            completion_per_token: output / 1_000_000.0,
            cache_read_per_token: Some(cached / 1_000_000.0),
            ..Pricing::default()
        })
    }

    fn canonical_hint(model: &str) -> Option<String> {
        let lower = model.to_ascii_lowercase();
        let hint = if lower.contains("codestral") && !lower.contains("embed") {
            "mistralai/codestral-2508"
        } else if lower.contains("mistral-large-4") {
            "mistralai/mistral-large-4-0"
        } else if lower.contains("mistral-medium") {
            "mistralai/mistral-medium-3-5"
        } else if lower.contains("mistral-small-2603") || lower.contains("mistral-small-4") {
            "mistralai/mistral-small-2603"
        } else if lower.contains("ministral-14b") {
            "mistralai/ministral-14b-2512"
        } else if lower.contains("ministral-8b") {
            "mistralai/ministral-8b-2512"
        } else if lower.contains("ministral-3b") {
            "mistralai/ministral-3b-2512"
        } else if lower.contains("mistral-large-2512") || lower.contains("mistral-large-3") {
            "mistralai/mistral-large-2512"
        } else {
            return None;
        };
        Some(hint.to_owned())
    }

    fn mistral_response_format(schema: &ResponseSchema) -> Value {
        match schema {
            ResponseSchema::JsonObject => json!({"type":"json_object"}),
            ResponseSchema::JsonSchema {
                name,
                schema,
                strict,
            } => json!({
                "type":"json_schema",
                "json_schema":{
                    "name":name,
                    "schema_definition":crate::lower_json_schema(schema),
                    "strict":strict,
                }
            }),
        }
    }

    fn function_tool(tool: &FunctionTool) -> Value {
        let mut function = serde_json::Map::new();
        function.insert("name".to_owned(), json!(tool.name));
        if let Some(description) = tool.description.as_ref() {
            function.insert("description".to_owned(), json!(description));
        }
        function.insert(
            "parameters".to_owned(),
            crate::lower_json_schema(&tool.parameters),
        );
        json!({"type":"function", "function":function})
    }

    fn tool_choice(choice: &ToolChoice) -> Value {
        match choice {
            ToolChoice::Auto => json!("auto"),
            ToolChoice::Required => json!("any"),
            ToolChoice::None => json!("none"),
            ToolChoice::Function { name } => json!({
                "type":"function",
                "function":{"name":name},
            }),
        }
    }

    fn compile_body(request: &InferenceRequest) -> Result<Value, ProviderError> {
        if request.requirements.reasoning == ReasoningEffort::Max {
            return Err(ProviderError::InvalidRequest(
                "Mistral reasoning_effort supports through xhigh, not max".to_owned(),
            ));
        }
        let mut body = base_chat_body(request, "max_tokens");
        if let Some(schema) = request.requirements.response_schema.as_ref() {
            body.insert(
                "response_format".to_owned(),
                Self::mistral_response_format(schema),
            );
        }
        if !request.tools.is_empty() {
            body.insert(
                "tools".to_owned(),
                Value::Array(request.tools.iter().map(Self::function_tool).collect()),
            );
        }
        if let Some(choice) = request.tool_choice.as_ref() {
            body.insert("tool_choice".to_owned(), Self::tool_choice(choice));
        }
        if let Some(parallel) = request.parallel_tool_calls {
            body.insert("parallel_tool_calls".to_owned(), json!(parallel));
        }
        if request.requirements.reasoning != ReasoningEffort::None {
            body.insert(
                "reasoning_effort".to_owned(),
                json!(request.requirements.reasoning.as_str()),
            );
        }
        Ok(Value::Object(body))
    }

    fn response_content(raw: &Value) -> String {
        let Some(content) = raw.pointer("/choices/0/message/content") else {
            return String::new();
        };
        if let Some(text) = content.as_str() {
            return text.to_owned();
        }
        content
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|chunk| {
                if chunk.get("type").and_then(Value::as_str) == Some("text") {
                    chunk.get("text").and_then(Value::as_str)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }

    fn response_tool_calls(raw: &Value) -> Vec<ToolCall> {
        raw.pointer("/choices/0/message/tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|call| {
                let function = call.get("function")?;
                let name = function.get("name")?.as_str()?.to_owned();
                let arguments = match function.get("arguments") {
                    Some(Value::String(arguments)) => serde_json::from_str(arguments)
                        .unwrap_or_else(|_| Value::String(arguments.to_owned())),
                    Some(arguments) => arguments.clone(),
                    None => Value::Null,
                };
                Some(ToolCall {
                    id: call.get("id").and_then(Value::as_str).map(str::to_owned),
                    name,
                    arguments,
                })
            })
            .collect()
    }

    fn bind_unknown_quota(
        &self,
        ledger: &mut QuotaLedger,
        route: &QuotaRouteKey,
        suffix: &str,
        metric: QuotaMetric,
    ) {
        let pool = self.quota_pool.as_deref().unwrap_or("default");
        let scope_model = if self.surface == MistralSurface::Studio {
            format!("{}/{}/{}", self.surface.source(), pool, route.model)
        } else {
            format!("{}/{pool}", self.surface.source())
        };
        let id = QuotaResourceId(format!("{scope_model}/{suffix}"));
        if ledger.graph().resource(&id).is_none() {
            ledger.graph_mut().insert_resource(QuotaResource::unknown(
                id.0.clone(),
                if self.surface == MistralSurface::Studio {
                    QuotaScope::Model
                } else {
                    QuotaScope::Provider
                },
                metric,
            ));
        }
        let _ = ledger.graph_mut().bind(route.clone(), &id);
    }
}

#[async_trait]
impl InferenceProvider for MistralCloud {
    async fn endpoints(&self, model: &str) -> Result<Vec<Endpoint>, ProviderError> {
        let models = self.native_models().await?;
        let Some(native) = models.iter().find(|candidate| candidate.id == model) else {
            return Err(ProviderError::NoEligibleEndpoint(model.to_owned()));
        };
        let pricing = native.pricing.clone().ok_or_else(|| {
            ProviderError::NoEligibleEndpoint(format!(
                "{model}: Mistral direct pricing is not modeled"
            ))
        })?;
        let reasoning = if native.claimed_parameters.contains("reasoning_effort") {
            ReasoningSupport::Exact(
                [
                    ReasoningEffort::Minimal,
                    ReasoningEffort::Low,
                    ReasoningEffort::Medium,
                    ReasoningEffort::High,
                    ReasoningEffort::Xhigh,
                ]
                .into_iter()
                .collect(),
            )
        } else {
            ReasoningSupport::None
        };
        Ok(vec![Endpoint {
            source: self.surface.source().to_owned(),
            credential: None,
            provider_key: self.surface.source().to_owned(),
            id: self.surface.source().to_owned(),
            provider: self.surface.provider_name().to_owned(),
            model: model.to_owned(),
            route_class: self.surface.route_class(),
            display_name: native
                .display_name
                .clone()
                .unwrap_or_else(|| model.to_owned()),
            quantization: None,
            status: 0,
            context_length: native.context_length,
            max_prompt_tokens: None,
            max_completion_tokens: native.max_completion_tokens,
            pricing,
            stats: EndpointStats::default(),
            workload_policy: crate::WorkloadPolicy::default(),
            capabilities: EndpointCapabilities {
                adapter_parameters: [
                    "chat_completions",
                    "system_role",
                    "max_tokens",
                    "temperature",
                    "response_format",
                    "structured_outputs",
                    "tools",
                    "tool_choice",
                    "parallel_tool_calls",
                    "reasoning",
                    "reasoning_effort",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
                supported_parameters: native.claimed_parameters.clone(),
                verified_parameters: BTreeSet::new(),
                rejected_parameters: BTreeSet::new(),
                verified_reasoning_efforts: BTreeSet::new(),
                rejected_reasoning_efforts: BTreeSet::new(),
                constraint_enforcement: BTreeMap::new(),
                verified_schema_features: BTreeMap::new(),
                rejected_schema_features: BTreeMap::new(),
                reasoning,
                supports_implicit_caching: false,
            },
        }])
    }

    fn configure_quota_route(
        &self,
        route: &QuotaRouteKey,
        ledger: &mut QuotaLedger,
    ) -> Result<(), ProviderError> {
        if route.source != self.surface.source() || route.provider != self.surface.source() {
            return Err(ProviderError::InvalidRequest(format!(
                "{} quota topology received a foreign route",
                self.surface.provider_name()
            )));
        }
        self.bind_unknown_quota(ledger, route, "requests", QuotaMetric::Requests);
        self.bind_unknown_quota(ledger, route, "tokens", QuotaMetric::TotalTokens);
        Ok(())
    }

    async fn execute_plan(
        &self,
        request: &InferenceRequest,
        plan: &RoutePlan,
        retry: RetryPolicy,
    ) -> Result<InferenceResponse, ProviderError> {
        let Some(candidate) = plan.candidates.first() else {
            return Err(ProviderError::NoEligibleEndpoint(request.model.clone()));
        };
        if candidate.endpoint.source != self.surface.source()
            || candidate.endpoint.provider_key != self.surface.source()
        {
            return Err(ProviderError::InvalidRequest(format!(
                "{} adapter received a foreign route",
                self.surface.provider_name()
            )));
        }
        let body = Self::compile_body(request)?;
        let response = execute_json_with_retry(
            &self.client,
            Some(&self.api_key),
            &format!("{}/chat/completions", self.base_url),
            &body,
            retry,
        )
        .await?;
        let raw = response.raw;
        let content = Self::response_content(&raw);
        let tool_calls = Self::response_tool_calls(&raw);
        if content.is_empty() && tool_calls.is_empty() {
            return Err(ProviderError::Response(format!(
                "{} response had neither text nor tool calls",
                self.surface.provider_name()
            )));
        }
        let usage = TokenUsage::from_openai_compatible(&raw);
        let usage_cost_usd = if self.surface == MistralSurface::Codestral {
            Some(0.0)
        } else {
            let input = usage.input_tokens.unwrap_or(plan.prompt_tokens);
            let output = usage.output_tokens.unwrap_or(plan.expected_output_tokens);
            Some(candidate.endpoint.pricing.expected_cost(input, output))
        };
        Ok(InferenceResponse {
            content,
            model: raw.get("model").and_then(Value::as_str).map(str::to_owned),
            provider: Some(self.surface.source().to_owned()),
            usage_cost_usd,
            usage,
            tool_calls,
            output_repairs: Vec::new(),
            quota_observations: response.quota_observations,
            raw,
        })
    }
}

#[derive(Debug, Deserialize)]
struct MistralModelEnvelope {
    data: Vec<MistralModel>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct MistralModelCapabilities {
    #[serde(default)]
    completion_chat: Option<bool>,
    #[serde(default)]
    function_calling: Option<bool>,
    #[serde(default)]
    reasoning: Option<bool>,
    #[serde(default)]
    completion_fim: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct MistralModel {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    owned_by: Option<String>,
    #[serde(default)]
    max_context_length: Option<u64>,
    #[serde(default)]
    capabilities: MistralModelCapabilities,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    billing_model_name: Option<String>,
}

impl MistralModel {
    fn normalize(self, surface: MistralSurface) -> Option<NativeModelRecord> {
        if !self.capabilities.completion_chat.unwrap_or(false) {
            return None;
        }
        if surface == MistralSurface::Codestral
            && !self.id.to_ascii_lowercase().contains("codestral")
        {
            return None;
        }
        let pricing = if surface == MistralSurface::Codestral {
            Some(Pricing::default())
        } else {
            let pricing_key = self
                .billing_model_name
                .as_deref()
                .unwrap_or(self.id.as_str());
            MistralCloud::studio_pricing(pricing_key)
                .or_else(|| MistralCloud::studio_pricing(&self.id))
        }?;
        let canonical_hint = self
            .billing_model_name
            .as_deref()
            .or_else(|| self.aliases.first().map(String::as_str))
            .and_then(MistralCloud::canonical_hint)
            .or_else(|| MistralCloud::canonical_hint(&self.id));
        Some(NativeModelRecord {
            id: self.id,
            display_name: self.name,
            owned_by: self.owned_by,
            created_unix: self.created,
            context_length: self.max_context_length,
            max_completion_tokens: None,
            claimed_parameters: MistralCloud::claimed_parameters(&self.capabilities),
            pricing: Some(pricing),
            route_class: Some(surface.route_class()),
            canonical_hint,
            virtual_route: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_card() -> MistralModel {
        MistralModel {
            id: "mistral-medium-latest".to_owned(),
            name: Some("Mistral Medium 3.5".to_owned()),
            created: Some(1_790_000_000),
            owned_by: Some("mistralai".to_owned()),
            max_context_length: Some(262_144),
            capabilities: MistralModelCapabilities {
                completion_chat: Some(true),
                function_calling: Some(true),
                reasoning: Some(true),
                completion_fim: Some(false),
            },
            aliases: vec!["mistral-medium-latest".to_owned()],
            billing_model_name: Some("mistral-medium-3-5".to_owned()),
        }
    }

    #[test]
    fn normalizes_rich_model_card_and_pricing() -> Result<(), Box<dyn std::error::Error>> {
        let native = model_card()
            .normalize(MistralSurface::Studio)
            .ok_or("modeled Mistral model")?;
        assert_eq!(native.context_length, Some(262_144));
        assert_eq!(native.route_class, Some(ModelRouteClass::Standard));
        assert!(native.claimed_parameters.contains("structured_outputs"));
        assert!(native.claimed_parameters.contains("tool_choice"));
        assert!(native.claimed_parameters.contains("reasoning_effort"));
        let pricing = native.pricing.ok_or("pricing")?;
        assert_eq!(pricing.prompt_per_token, 1.5 / 1_000_000.0);
        assert_eq!(pricing.completion_per_token, 7.5 / 1_000_000.0);
        Ok(())
    }

    #[test]
    fn codestral_is_free_and_canonicalized() {
        let native = MistralCloud::codestral_fallback_record();
        assert_eq!(native.id, "codestral-latest");
        assert_eq!(native.route_class, Some(ModelRouteClass::Free));
        assert_eq!(
            native.canonical_hint.as_deref(),
            Some("mistralai/codestral-2508")
        );
        assert_eq!(native.context_length, Some(131_072));
    }

    #[test]
    fn compiles_mistral_specific_schema_and_tools() -> Result<(), ProviderError> {
        let mut request = InferenceRequest::new("mistral-medium-latest", "lookup", 32);
        request.requirements.response_schema = Some(ResponseSchema::JsonSchema {
            name: "answer".to_owned(),
            schema: json!({
                "type":"object",
                "properties":{"n":{"type":"integer"}},
                "required":["n"],
                "additionalProperties":false
            }),
            strict: true,
        });
        request.tools.push(FunctionTool {
            name: "lookup".to_owned(),
            description: Some("lookup".to_owned()),
            parameters: json!({
                "type":"object",
                "properties":{"id":{"type":"integer"}},
                "required":["id"],
                "additionalProperties":false
            }),
            strict: true,
        });
        request.tool_choice = Some(ToolChoice::Required);
        request.parallel_tool_calls = Some(false);
        let body = MistralCloud::compile_body(&request)?;
        assert_eq!(
            body.pointer("/response_format/json_schema/schema_definition/type")
                .and_then(Value::as_str),
            Some("object")
        );
        assert!(
            body.pointer("/response_format/json_schema/schema")
                .is_none()
        );
        assert_eq!(body.get("tool_choice").and_then(Value::as_str), Some("any"));
        assert_eq!(
            body.get("parallel_tool_calls").and_then(Value::as_bool),
            Some(false)
        );
        assert!(body.pointer("/tools/0/function/strict").is_none());
        Ok(())
    }

    #[test]
    fn parses_chunked_text_and_object_tool_arguments() {
        let raw = json!({
            "choices":[{"message":{
                "content":[{"type":"text","text":"hello "},{"type":"text","text":"world"}],
                "tool_calls":[{"id":"x","function":{"name":"f","arguments":{"n":1}}}]
            }}]
        });
        assert_eq!(MistralCloud::response_content(&raw), "hello world");
        let calls = MistralCloud::response_tool_calls(&raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, json!({"n":1}));
    }

    #[test]
    fn rejects_non_mistral_max_effort() {
        let mut request = InferenceRequest::new("mistral-medium-latest", "hi", 8);
        request.requirements.reasoning = ReasoningEffort::Max;
        assert!(MistralCloud::compile_body(&request).is_err());
    }

    #[test]
    fn dedicated_codestral_quota_is_unknown_not_infinite() -> Result<(), ProviderError> {
        let provider = MistralCloud::codestral("test").with_quota_pool("personal");
        let route = QuotaRouteKey {
            source: "codestral".to_owned(),
            credential: None,
            provider: "codestral".to_owned(),
            model: "codestral-latest".to_owned(),
        };
        let mut ledger = QuotaLedger::new(crate::QuotaGraph::new());
        provider.configure_quota_route(&route, &mut ledger)?;
        let report = ledger.evaluate(&route, crate::QuotaDemand::default(), 1);
        assert_eq!(report.checks.len(), 2);
        assert!(
            report
                .checks
                .iter()
                .all(|check| matches!(check.available, crate::QuotaAmount::Unknown))
        );
        Ok(())
    }
}
