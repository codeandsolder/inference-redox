use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use super::openai_compatible::{base_chat_body, execute_json_with_retry, response_format};
use crate::{
    Endpoint, EndpointCapabilities, EndpointStats, FunctionTool, InferenceProvider,
    InferenceRequest, InferenceResponse, ModelCatalog, ModelRouteClass, NativeModelRecord, Pricing,
    ProviderError, ProviderInventory, QuotaLedger, QuotaMetric, QuotaResource, QuotaResourceId,
    QuotaRouteKey, QuotaScope, ReasoningSupport, ResponseSchema, RetryPolicy, RoutePlan,
    TokenUsage, ToolCall, ToolChoice,
};

const DEFAULT_BASE_URL: &str = "https://ollama.com/v1";
const FREE_STARTER_GEMMA4: &str = "gemma4:31b";

/// Ollama Cloud free-tier adapter.
///
/// This intentionally exposes only models explicitly documented as available on Ollama's free
/// starter allowance. Other cloud models are omitted rather than risking paid credit usage.
#[derive(Debug, Clone)]
pub struct OllamaCloud {
    client: Client,
    api_key: String,
    base_url: String,
    model_cache: Arc<Mutex<Option<CachedOllamaModels>>>,
    catalog_ttl: Duration,
}

#[derive(Debug, Clone)]
struct CachedOllamaModels {
    fetched_at: Instant,
    models: Vec<NativeModelRecord>,
}

impl OllamaCloud {
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_owned(),
            model_cache: Arc::new(Mutex::new(None)),
            catalog_ttl: Duration::from_secs(60),
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

    /// Fetch Ollama's provider-native model list and retain only documented free starter models.
    ///
    /// # Errors
    /// Returns a catalog error when the Ollama inventory cannot be fetched or decoded.
    pub async fn inventory(
        &self,
        catalog: &ModelCatalog,
    ) -> Result<ProviderInventory, ProviderError> {
        Ok(ProviderInventory::reconcile(
            "ollama-cloud",
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

        let fetched = async {
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
            let envelope: OllamaModelEnvelope = response
                .json()
                .await
                .map_err(|error| ProviderError::Catalog(error.to_string()))?;
            Ok::<Vec<NativeModelRecord>, ProviderError>(
                envelope
                    .data
                    .into_iter()
                    .filter(OllamaModel::is_documented_free_starter)
                    .map(OllamaModel::normalize)
                    .collect(),
            )
        }
        .await;

        match fetched {
            Ok(models) => {
                cache = self.model_cache.lock().await;
                *cache = Some(CachedOllamaModels {
                    fetched_at: Instant::now(),
                    models: models.clone(),
                });
                Ok(models)
            }
            Err(error) => stale.ok_or(error),
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
        json!({"type": "function", "function": function})
    }

    fn compile_body(request: &InferenceRequest) -> Result<Value, ProviderError> {
        if request.parallel_tool_calls.is_some() {
            return Err(ProviderError::InvalidRequest(
                "Ollama OpenAI compatibility does not document parallel_tool_calls".to_owned(),
            ));
        }
        if !matches!(request.requirements.reasoning, crate::ReasoningEffort::None) {
            return Err(ProviderError::InvalidRequest(
                "Ollama model-specific reasoning metadata is not normalized yet".to_owned(),
            ));
        }

        let mut body = base_chat_body(request, "max_tokens");
        if let Some(schema) = request.requirements.response_schema.as_ref() {
            match schema {
                ResponseSchema::JsonObject => {
                    body.insert("response_format".to_owned(), response_format(schema));
                }
                ResponseSchema::JsonSchema { .. } => {
                    return Err(ProviderError::InvalidRequest(
                        "Ollama Cloud does not currently support schema-constrained structured outputs"
                            .to_owned(),
                    ));
                }
            }
        }
        if request
            .tool_choice
            .as_ref()
            .is_some_and(|choice| !matches!(choice, ToolChoice::Auto))
        {
            return Err(ProviderError::InvalidRequest(
                "Ollama OpenAI compatibility does not support tool_choice".to_owned(),
            ));
        }
        if !request.tools.is_empty() {
            body.insert(
                "tools".to_owned(),
                Value::Array(request.tools.iter().map(Self::function_tool).collect()),
            );
        }
        Ok(Value::Object(body))
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
}

#[async_trait]
impl InferenceProvider for OllamaCloud {
    async fn endpoints(&self, model: &str) -> Result<Vec<Endpoint>, ProviderError> {
        let models = self.native_models().await?;
        let Some(native) = models.iter().find(|candidate| candidate.id == model) else {
            return Err(ProviderError::NoEligibleEndpoint(model.to_owned()));
        };
        Ok(vec![Endpoint {
            source: "ollama-cloud".to_owned(),
            credential: None,
            provider_key: "ollama-cloud".to_owned(),
            id: "ollama-cloud".to_owned(),
            provider: "Ollama Cloud".to_owned(),
            model: model.to_owned(),
            route_class: ModelRouteClass::Free,
            display_name: native
                .display_name
                .clone()
                .unwrap_or_else(|| model.to_owned()),
            quantization: None,
            status: 0,
            context_length: native.context_length,
            max_prompt_tokens: None,
            max_completion_tokens: native.max_completion_tokens,
            pricing: Pricing::default(),
            stats: EndpointStats::default(),
            workload_policy: crate::WorkloadPolicy::default(),
            capabilities: EndpointCapabilities {
                adapter_parameters: [
                    "chat_completions",
                    "system_role",
                    "max_tokens",
                    "temperature",
                    "response_format",
                    "tools",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
                supported_parameters: [
                    "chat_completions",
                    "system_role",
                    "max_tokens",
                    "temperature",
                    "response_format",
                    "tools",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
                verified_parameters: BTreeSet::new(),
                rejected_parameters: BTreeSet::new(),
                verified_reasoning_efforts: BTreeSet::new(),
                rejected_reasoning_efforts: BTreeSet::new(),
                constraint_enforcement: BTreeMap::new(),
                verified_schema_features: BTreeMap::new(),
                rejected_schema_features: BTreeMap::new(),
                reasoning: ReasoningSupport::None,
                supports_implicit_caching: false,
            },
        }])
    }

    fn configure_quota_route(
        &self,
        route: &QuotaRouteKey,
        ledger: &mut QuotaLedger,
    ) -> Result<(), ProviderError> {
        if route.source != "ollama-cloud" || route.provider != "ollama-cloud" {
            return Err(ProviderError::InvalidRequest(
                "Ollama quota topology received a non-Ollama route".to_owned(),
            ));
        }
        let id = QuotaResourceId("ollama-cloud/free/concurrency".to_owned());
        if ledger.graph().resource(&id).is_none() {
            ledger.graph_mut().insert_resource(QuotaResource::finite(
                id.0.clone(),
                QuotaScope::Provider,
                QuotaMetric::ConcurrentRequests,
                1,
                1,
            ));
        }
        let _ = ledger.graph_mut().bind(route.clone(), &id);

        let credits = QuotaResourceId("ollama-cloud/free/starter-credits".to_owned());
        if ledger.graph().resource(&credits).is_none() {
            ledger.graph_mut().insert_resource(QuotaResource::unknown(
                credits.0.clone(),
                QuotaScope::Account,
                QuotaMetric::Credits,
            ));
        }
        let _ = ledger.graph_mut().bind(route.clone(), &credits);
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
        if candidate.endpoint.source != "ollama-cloud"
            || candidate.endpoint.provider_key != "ollama-cloud"
        {
            return Err(ProviderError::InvalidRequest(
                "Ollama adapter received a non-Ollama route".to_owned(),
            ));
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
        let tool_calls = Self::response_tool_calls(&raw);
        let content = raw
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if content.is_empty() && tool_calls.is_empty() {
            return Err(ProviderError::Response(
                "Ollama response had neither text nor tool calls".to_owned(),
            ));
        }
        Ok(InferenceResponse {
            content,
            model: raw.get("model").and_then(Value::as_str).map(str::to_owned),
            provider: Some("ollama-cloud".to_owned()),
            usage_cost_usd: Some(0.0),
            usage: TokenUsage::from_openai_compatible(&raw),
            tool_calls,
            output_repairs: Vec::new(),
            quota_observations: response.quota_observations,
            raw,
        })
    }
}

#[derive(Debug, Deserialize)]
struct OllamaModelEnvelope {
    data: Vec<OllamaModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaModel {
    id: String,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    owned_by: Option<String>,
}

impl OllamaModel {
    fn is_documented_free_starter(&self) -> bool {
        self.id == FREE_STARTER_GEMMA4
    }

    fn normalize(self) -> NativeModelRecord {
        NativeModelRecord {
            canonical_hint: (self.id == FREE_STARTER_GEMMA4)
                .then(|| "google/gemma-4-31b-it".to_owned()),
            id: self.id.clone(),
            display_name: Some("Gemma 4 31B".to_owned()),
            owned_by: self.owned_by.or_else(|| Some("ollama".to_owned())),
            created_unix: self.created,
            context_length: Some(262_144),
            max_completion_tokens: None,
            claimed_parameters: [
                "chat_completions",
                "system_role",
                "max_tokens",
                "temperature",
                "response_format",
                "tools",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            pricing: Some(Pricing::default()),
            route_class: Some(ModelRouteClass::Free),
            virtual_route: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_documented_free_starter_is_retained() -> Result<(), Box<dyn std::error::Error>> {
        let envelope: OllamaModelEnvelope = serde_json::from_value(json!({
            "object": "list",
            "data": [
                {"id": "gemma4:31b", "object": "model", "owned_by": "library"},
                {"id": "gpt-oss:120b", "object": "model", "owned_by": "library"}
            ]
        }))?;
        let models = envelope
            .data
            .into_iter()
            .filter(OllamaModel::is_documented_free_starter)
            .map(OllamaModel::normalize)
            .collect::<Vec<_>>();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "gemma4:31b");
        assert_eq!(
            models[0].canonical_hint.as_deref(),
            Some("google/gemma-4-31b-it")
        );
        assert_eq!(models[0].route_class, Some(ModelRouteClass::Free));
        assert_eq!(models[0].context_length, Some(262_144));
        Ok(())
    }

    #[test]
    fn compiles_json_mode_and_native_tools() -> Result<(), ProviderError> {
        let mut request = InferenceRequest::new("gemma4:31b", "lookup", 16);
        request.requirements.response_schema = Some(crate::ResponseSchema::JsonObject);
        request.tools.push(FunctionTool {
            name: "lookup".to_owned(),
            description: Some("Lookup a value".to_owned()),
            parameters: json!({
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
                "additionalProperties": false
            }),
            strict: true,
        });
        request.tool_choice = Some(ToolChoice::Auto);
        let body = OllamaCloud::compile_body(&request)?;
        assert_eq!(
            body.pointer("/response_format/type")
                .and_then(Value::as_str),
            Some("json_object")
        );
        assert_eq!(
            body.pointer("/tools/0/function/name")
                .and_then(Value::as_str),
            Some("lookup")
        );
        assert!(body.get("tool_choice").is_none());
        Ok(())
    }

    #[test]
    fn rejects_required_tool_choice() {
        let mut request = InferenceRequest::new("gemma4:31b", "lookup", 16);
        request.tools.push(FunctionTool {
            name: "lookup".to_owned(),
            description: None,
            parameters: json!({"type":"object"}),
            strict: false,
        });
        request.tool_choice = Some(ToolChoice::Required);
        assert!(OllamaCloud::compile_body(&request).is_err());
    }

    #[test]
    fn rejects_native_json_schema() {
        let mut request = InferenceRequest::new("gemma4:31b", "json", 16);
        request.requirements.response_schema = Some(ResponseSchema::JsonSchema {
            name: "answer".to_owned(),
            schema: json!({"type":"object"}),
            strict: true,
        });
        assert!(OllamaCloud::compile_body(&request).is_err());
    }

    #[test]
    fn parses_string_and_object_tool_arguments() {
        let raw = json!({
            "choices": [{
                "message": {
                    "tool_calls": [
                        {
                            "id": "a",
                            "function": {"name": "one", "arguments": "{\"x\":1}"}
                        },
                        {
                            "id": "b",
                            "function": {"name": "two", "arguments": {"y": 2}}
                        }
                    ]
                }
            }]
        });
        let calls = OllamaCloud::response_tool_calls(&raw);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments, json!({"x": 1}));
        assert_eq!(calls[1].arguments, json!({"y": 2}));
    }

    #[test]
    fn free_tier_concurrency_is_shared() -> Result<(), ProviderError> {
        let provider = OllamaCloud::new("test");
        let route = |model: &str| QuotaRouteKey {
            source: "ollama-cloud".to_owned(),
            credential: None,
            provider: "ollama-cloud".to_owned(),
            model: model.to_owned(),
        };
        let mut ledger = QuotaLedger::new(crate::QuotaGraph::new());
        provider.configure_quota_route(&route("gemma4:31b"), &mut ledger)?;
        let report = ledger.evaluate(&route("gemma4:31b"), crate::QuotaDemand::default(), 1);
        assert_eq!(report.checks.len(), 2);
        assert!(report.checks.iter().any(|check| {
            check.metric == QuotaMetric::ConcurrentRequests
                && check.available == crate::QuotaAmount::Finite(1)
        }));
        assert!(report.checks.iter().any(|check| {
            check.metric == QuotaMetric::Credits && check.available == crate::QuotaAmount::Unknown
        }));
        Ok(())
    }
}
