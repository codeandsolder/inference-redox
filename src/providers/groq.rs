use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use super::openai_compatible::{base_chat_body, execute_chat_with_retry, response_format};
use crate::{
    Endpoint, EndpointCapabilities, EndpointStats, InferenceProvider, InferenceRequest,
    InferenceResponse, ModelCatalog, ModelRouteClass, NativeModelRecord, Pricing, ProviderError,
    ProviderInventory, QuotaLedger, QuotaMetric, QuotaResource, QuotaResourceId, QuotaRouteKey,
    QuotaScope, ReasoningEffort, ReasoningSupport, ResponseSchema, RetryPolicy, RoutePlan,
};

const DEFAULT_BASE_URL: &str = "https://api.groq.com/openai/v1";

/// Direct `GroqCloud` adapter for recurring free-tier capacity.
#[derive(Debug, Clone)]
pub struct GroqCloud {
    client: Client,
    api_key: String,
    base_url: String,
    model_cache: Arc<Mutex<Option<CachedGroqModels>>>,
    catalog_ttl: Duration,
    quota_pool: Option<String>,
}

#[derive(Debug, Clone)]
struct CachedGroqModels {
    fetched_at: Instant,
    models: Vec<NativeModelRecord>,
}

impl GroqCloud {
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_owned(),
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

    /// Assign an opaque logical upstream quota pool (organization/project). Multiple credentials
    /// with the same label share model-scoped Groq resources; the label must never be a secret.
    #[must_use]
    pub fn with_quota_pool(mut self, quota_pool: impl Into<String>) -> Self {
        self.quota_pool = Some(quota_pool.into());
        self
    }

    fn bind_quota_resource(
        ledger: &mut QuotaLedger,
        route: &QuotaRouteKey,
        id: QuotaResourceId,
        metric: QuotaMetric,
        bucket: &str,
    ) {
        if ledger.graph().resource(&id).is_none() {
            ledger.graph_mut().insert_resource(QuotaResource::unknown(
                id.0.clone(),
                QuotaScope::Model,
                metric,
            ));
        }
        let _ = ledger.graph_mut().bind(route.clone(), &id);
        ledger.observation_map_mut().bind(route.clone(), bucket, id);
    }

    /// Fetch Groq's native model inventory and reconcile it against the canonical catalog.
    ///
    /// # Errors
    /// Returns a catalog error when the Groq inventory cannot be fetched or decoded.
    pub async fn inventory(
        &self,
        catalog: &ModelCatalog,
    ) -> Result<ProviderInventory, ProviderError> {
        Ok(ProviderInventory::reconcile(
            "groq",
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
            let envelope: GroqModelEnvelope = response
                .json()
                .await
                .map_err(|error| ProviderError::Catalog(error.to_string()))?;
            Ok::<Vec<NativeModelRecord>, ProviderError>(
                envelope
                    .data
                    .into_iter()
                    .filter(|model| model.active.unwrap_or(true))
                    .map(GroqModel::normalize)
                    .collect(),
            )
        }
        .await;

        match fetched {
            Ok(models) => {
                cache = self.model_cache.lock().await;
                *cache = Some(CachedGroqModels {
                    fetched_at: Instant::now(),
                    models: models.clone(),
                });
                Ok(models)
            }
            Err(error) => stale.ok_or(error),
        }
    }

    fn reasoning_support(model: &str) -> ReasoningSupport {
        let efforts = match model {
            "openai/gpt-oss-20b" | "openai/gpt-oss-120b" | "qwen/qwen3.8-27b" => [
                ReasoningEffort::Low,
                ReasoningEffort::Medium,
                ReasoningEffort::High,
            ]
            .into_iter()
            .collect(),
            "meta-models/muse-spark-open" => [
                ReasoningEffort::Low,
                ReasoningEffort::Medium,
                ReasoningEffort::High,
                ReasoningEffort::Xhigh,
            ]
            .into_iter()
            .collect(),
            _ => return ReasoningSupport::None,
        };
        ReasoningSupport::Exact(efforts)
    }

    fn supports_strict_schema(model: &str) -> bool {
        matches!(
            model,
            "openai/gpt-oss-20b"
                | "openai/gpt-oss-120b"
                | "qwen/qwen3.8-27b"
                | "meta-models/muse-spark-open"
        )
    }

    fn claimed_parameters(model: &str) -> BTreeSet<String> {
        let mut parameters = [
            "chat_completions",
            "system_role",
            "max_completion_tokens",
            "temperature",
            "response_format",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
        if Self::supports_strict_schema(model) {
            parameters.insert("structured_outputs".to_owned());
        }
        if !matches!(Self::reasoning_support(model), ReasoningSupport::None) {
            parameters.insert("reasoning".to_owned());
            parameters.insert("reasoning_effort".to_owned());
        }
        parameters
    }

    fn compile_body(request: &InferenceRequest) -> Result<Value, ProviderError> {
        let reasoning = Self::reasoning_support(&request.model);
        if request.requirements.reasoning != ReasoningEffort::None
            && !reasoning.supports(request.requirements.reasoning)
        {
            return Err(ProviderError::InvalidRequest(format!(
                "Groq model {} does not support reasoning effort {}",
                request.model,
                request.requirements.reasoning.as_str()
            )));
        }
        if matches!(
            request.requirements.response_schema,
            Some(ResponseSchema::JsonSchema { strict: true, .. })
        ) && !Self::supports_strict_schema(&request.model)
        {
            return Err(ProviderError::InvalidRequest(format!(
                "Groq model {} does not support strict structured outputs",
                request.model
            )));
        }

        let mut body = base_chat_body(request, "max_completion_tokens");
        if let Some(schema) = request.requirements.response_schema.as_ref() {
            body.insert("response_format".to_owned(), response_format(schema));
        }
        if request.requirements.reasoning != ReasoningEffort::None {
            body.insert(
                "reasoning_effort".to_owned(),
                json!(request.requirements.reasoning.as_str()),
            );
        }
        Ok(Value::Object(body))
    }
}

#[async_trait]
impl InferenceProvider for GroqCloud {
    async fn endpoints(&self, model: &str) -> Result<Vec<Endpoint>, ProviderError> {
        let models = self.native_models().await?;
        let Some(native) = models.iter().find(|candidate| candidate.id == model) else {
            return Err(ProviderError::NoEligibleEndpoint(model.to_owned()));
        };
        Ok(vec![Endpoint {
            source: "groq".to_owned(),
            credential: None,
            provider_key: "groq".to_owned(),
            id: "groq".to_owned(),
            provider: "GroqCloud".to_owned(),
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
                    "max_completion_tokens",
                    "temperature",
                    "response_format",
                    "structured_outputs",
                    "reasoning",
                    "reasoning_effort",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
                supported_parameters: Self::claimed_parameters(model),
                verified_parameters: BTreeSet::new(),
                rejected_parameters: BTreeSet::new(),
                reasoning: Self::reasoning_support(model),
                supports_implicit_caching: false,
            },
        }])
    }

    fn configure_quota_route(
        &self,
        route: &QuotaRouteKey,
        ledger: &mut QuotaLedger,
    ) -> Result<(), ProviderError> {
        if route.source != "groq" || route.provider != "groq" {
            return Err(ProviderError::InvalidRequest(
                "Groq quota topology received a non-Groq route".to_owned(),
            ));
        }
        let Some(pool) = self.quota_pool.as_deref() else {
            return Ok(());
        };
        let prefix = format!("groq/{pool}/{}", route.model);
        Self::bind_quota_resource(
            ledger,
            route,
            QuotaResourceId(format!("{prefix}/requests")),
            QuotaMetric::Requests,
            "rate_limit.requests",
        );
        Self::bind_quota_resource(
            ledger,
            route,
            QuotaResourceId(format!("{prefix}/tokens")),
            QuotaMetric::TotalTokens,
            "rate_limit.tokens",
        );
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
        if candidate.endpoint.source != "groq" || candidate.endpoint.provider_key != "groq" {
            return Err(ProviderError::InvalidRequest(
                "Groq adapter received a non-Groq route".to_owned(),
            ));
        }
        let body = Self::compile_body(request)?;
        execute_chat_with_retry(
            &self.client,
            Some(&self.api_key),
            &format!("{}/chat/completions", self.base_url),
            &body,
            "groq",
            Some(0.0),
            retry,
        )
        .await
    }
}

#[derive(Debug, Deserialize)]
struct GroqModelEnvelope {
    data: Vec<GroqModel>,
}

#[derive(Debug, Deserialize)]
struct GroqModel {
    id: String,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    owned_by: Option<String>,
    #[serde(default)]
    active: Option<bool>,
    #[serde(default)]
    context_window: Option<u64>,
    #[serde(default)]
    max_completion_tokens: Option<u64>,
}

impl GroqModel {
    fn normalize(self) -> NativeModelRecord {
        let claimed_parameters = GroqCloud::claimed_parameters(&self.id);
        NativeModelRecord {
            id: self.id,
            display_name: None,
            owned_by: self.owned_by,
            created_unix: self.created,
            context_length: self.context_window,
            max_completion_tokens: self.max_completion_tokens,
            claimed_parameters,
            pricing: Some(Pricing::default()),
            route_class: Some(ModelRouteClass::Free),
            canonical_hint: None,
            virtual_route: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_groq_model_record() -> Result<(), Box<dyn std::error::Error>> {
        let envelope: GroqModelEnvelope = serde_json::from_value(serde_json::json!({
            "object": "list",
            "data": [{
                "id": "openai/gpt-oss-120b",
                "object": "model",
                "created": 1_693_721_698,
                "owned_by": "OpenAI",
                "active": true,
                "context_window": 131_072,
                "max_completion_tokens": 65_536
            }]
        }))?;
        let normalized = envelope
            .data
            .into_iter()
            .next()
            .ok_or("missing model")?
            .normalize();
        assert_eq!(normalized.id, "openai/gpt-oss-120b");
        assert_eq!(normalized.context_length, Some(131_072));
        assert_eq!(normalized.max_completion_tokens, Some(65_536));
        assert_eq!(normalized.route_class, Some(ModelRouteClass::Free));
        assert!(normalized.claimed_parameters.contains("structured_outputs"));
        assert!(normalized.claimed_parameters.contains("reasoning_effort"));
        Ok(())
    }

    #[test]
    fn compiles_reasoning_and_strict_schema() -> Result<(), ProviderError> {
        let mut request = InferenceRequest::new("openai/gpt-oss-120b", "hello", 16);
        request.max_output_tokens = Some(64);
        request.requirements.reasoning = ReasoningEffort::Low;
        request.requirements.response_schema = Some(ResponseSchema::JsonSchema {
            name: "result".to_owned(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": false
            }),
            strict: true,
        });
        let body = GroqCloud::compile_body(&request)?;
        assert_eq!(
            body.get("max_completion_tokens").and_then(Value::as_u64),
            Some(64)
        );
        assert_eq!(
            body.get("reasoning_effort").and_then(Value::as_str),
            Some("low")
        );
        assert_eq!(
            body.pointer("/response_format/json_schema/strict")
                .and_then(Value::as_bool),
            Some(true)
        );
        Ok(())
    }

    #[test]
    fn model_specific_reasoning_support_is_conservative() {
        assert!(GroqCloud::reasoning_support("openai/gpt-oss-20b").supports(ReasoningEffort::Low));
        assert!(
            !GroqCloud::reasoning_support("llama-3.3-70b-versatile").supports(ReasoningEffort::Low)
        );
    }

    #[test]
    fn quota_pool_is_shared_across_credentials_for_same_model() -> Result<(), ProviderError> {
        let provider = GroqCloud::new("test").with_quota_pool("org-a");
        let route = |credential: &str| QuotaRouteKey {
            source: "groq".to_owned(),
            credential: Some(credential.to_owned()),
            provider: "groq".to_owned(),
            model: "openai/gpt-oss-120b".to_owned(),
        };
        let mut ledger = QuotaLedger::new(crate::QuotaGraph::new());
        provider.configure_quota_route(&route("key-a"), &mut ledger)?;
        provider.configure_quota_route(&route("key-b"), &mut ledger)?;
        let a = ledger.evaluate(&route("key-a"), crate::QuotaDemand::default(), 100);
        let b = ledger.evaluate(&route("key-b"), crate::QuotaDemand::default(), 100);
        assert_eq!(a.checks.len(), 2);
        assert_eq!(b.checks.len(), 2);
        let a_ids = a
            .checks
            .iter()
            .map(|check| check.resource_id.clone())
            .collect::<BTreeSet<_>>();
        let b_ids = b
            .checks
            .iter()
            .map(|check| check.resource_id.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(a_ids, b_ids);
        assert!(
            a.checks
                .iter()
                .all(|check| check.scope == QuotaScope::Model)
        );
        Ok(())
    }

    #[test]
    fn rejects_strict_schema_on_unlisted_model() {
        let mut request = InferenceRequest::new("llama-3.3-70b-versatile", "hello", 16);
        request.requirements.response_schema = Some(ResponseSchema::JsonSchema {
            name: "result".to_owned(),
            schema: serde_json::json!({"type": "object"}),
            strict: true,
        });
        assert!(matches!(
            GroqCloud::compile_body(&request),
            Err(ProviderError::InvalidRequest(_))
        ));
    }
}
