use std::collections::BTreeSet;
use std::time::Duration;

use async_trait::async_trait;

use super::openai_compatible::{
    OpenAiCompatibleProfile, OpenAiCompatibleProvider, ReasoningDialect, TemperatureZeroPolicy,
    TokenLimitField,
};
use crate::{
    Endpoint, InferenceProvider, InferenceRequest, InferenceResponse, ModelCatalog,
    ModelRouteClass, Pricing, ProviderError, ProviderInventory, ReasoningEffort, ReasoningSupport,
    RetryPolicy, RoutePlan,
};

/// Direct `GroqCloud` chat-completions provider for a free-plan credential.
#[derive(Debug, Clone)]
pub struct Groq {
    inner: OpenAiCompatibleProvider,
}

impl Groq {
    #[must_use]
    pub fn new_free(api_key: impl Into<String>) -> Self {
        let adapter_parameters = [
            "chat_completions",
            "system_role",
            "max_completion_tokens",
            "temperature",
            "response_format",
            "json_schema",
            "structured_outputs",
            "reasoning",
            "reasoning_effort",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        let nominal_parameters = [
            "chat_completions",
            "system_role",
            "max_completion_tokens",
            "temperature",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        Self {
            inner: OpenAiCompatibleProvider::new(
                api_key,
                OpenAiCompatibleProfile {
                    source: "groq".to_owned(),
                    provider_key: "groq".to_owned(),
                    provider_name: "GroqCloud".to_owned(),
                    base_url: "https://api.groq.com/openai/v1".to_owned(),
                    route_class: ModelRouteClass::Free,
                    token_limit_field: TokenLimitField::MaxCompletionTokens,
                    reasoning_dialect: ReasoningDialect::EffortField,
                    temperature_zero: TemperatureZeroPolicy::ClampTo(1e-8),
                    adapter_parameters,
                    nominal_parameters,
                    reasoning_support: ReasoningSupport::GatewayNormalized,
                    supports_implicit_caching: false,
                    pricing: Pricing::default(),
                    usage_cost_usd: Some(0.0),
                },
            ),
        }
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner = self.inner.with_base_url(base_url);
        self
    }

    #[must_use]
    pub fn with_catalog_ttl(mut self, ttl: Duration) -> Self {
        self.inner = self.inner.with_catalog_ttl(ttl);
        self
    }

    /// Reconcile `GroqCloud` discovery against the canonical catalog.
    ///
    /// Only models currently documented on the free text-generation tier are marked `Free` with
    /// known zero pricing. Other discovered Groq models remain catalog-visible with unknown
    /// economics and are not exposed as executable routes by this free-plan provider instance.
    ///
    /// # Errors
    /// Returns a provider catalog error if `/models` cannot be loaded and no stale cache exists.
    pub async fn inventory(
        &self,
        catalog: &ModelCatalog,
    ) -> Result<ProviderInventory, ProviderError> {
        let mut inventory = self.inner.inventory(catalog).await?;
        for model in &mut inventory.models {
            Self::apply_parameter_claims(&model.native_id, &mut model.claimed_parameters);
            if Self::is_free_chat_model(&model.native_id) {
                model.pricing = Some(Pricing::default());
                model.route_class = Some(ModelRouteClass::Free);
            } else {
                model.claimed_parameters.clear();
                model.pricing = None;
                model.route_class = None;
            }
        }
        Ok(inventory)
    }

    fn is_free_chat_model(model: &str) -> bool {
        matches!(
            model,
            "openai/gpt-oss-20b"
                | "openai/gpt-oss-120b"
                | "openai/gpt-oss-safeguard-20b"
                | "qwen/qwen3.8-27b"
        )
    }

    fn apply_parameter_claims(model: &str, parameters: &mut BTreeSet<String>) {
        parameters.insert("response_format".to_owned());
        match model {
            "openai/gpt-oss-20b" | "openai/gpt-oss-120b" | "qwen/qwen3.8-27b" => {
                parameters.extend([
                    "json_schema".to_owned(),
                    "structured_outputs".to_owned(),
                    "reasoning".to_owned(),
                    "reasoning_effort".to_owned(),
                ]);
            }
            "openai/gpt-oss-safeguard-20b" => {
                parameters.insert("json_schema".to_owned());
            }
            _ => {}
        }
    }

    fn apply_model_claims(endpoint: &mut Endpoint) {
        Self::apply_parameter_claims(
            &endpoint.model,
            &mut endpoint.capabilities.supported_parameters,
        );
        match endpoint.model.as_str() {
            "openai/gpt-oss-20b" | "openai/gpt-oss-120b" => {
                endpoint.capabilities.supports_implicit_caching = true;
                endpoint.capabilities.reasoning = ReasoningSupport::Exact(BTreeSet::from([
                    ReasoningEffort::Low,
                    ReasoningEffort::Medium,
                    ReasoningEffort::High,
                ]));
            }
            "openai/gpt-oss-safeguard-20b" => {
                endpoint.capabilities.supports_implicit_caching = true;
                endpoint.capabilities.reasoning = ReasoningSupport::None;
            }
            "qwen/qwen3.8-27b" => {
                endpoint.capabilities.reasoning = ReasoningSupport::Exact(BTreeSet::from([
                    ReasoningEffort::Low,
                    ReasoningEffort::Medium,
                    ReasoningEffort::High,
                ]));
            }
            _ => endpoint.capabilities.reasoning = ReasoningSupport::None,
        }
    }
}

#[async_trait]
impl InferenceProvider for Groq {
    async fn endpoints(&self, model: &str) -> Result<Vec<Endpoint>, ProviderError> {
        if !Self::is_free_chat_model(model) {
            return Err(ProviderError::NoEligibleEndpoint(model.to_owned()));
        }
        let mut endpoints = self.inner.endpoints(model).await?;
        for endpoint in &mut endpoints {
            Self::apply_model_claims(endpoint);
        }
        Ok(endpoints)
    }

    async fn execute_plan(
        &self,
        request: &InferenceRequest,
        plan: &RoutePlan,
        retry: RetryPolicy,
    ) -> Result<InferenceResponse, ProviderError> {
        self.inner.execute_plan(request, plan, retry).await
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use serde_json::Value;

    use super::*;
    use crate::{EndpointCapabilities, EndpointStats, ResponseSchema, RouteCandidate};

    fn groq_plan(model: &str) -> RoutePlan {
        RoutePlan {
            prompt_tokens: 1,
            expected_output_tokens: 8,
            cheapest_cost_usd: 0.0,
            cost_ceiling_usd: 0.0,
            candidates: vec![RouteCandidate {
                endpoint: Endpoint {
                    source: "groq".to_owned(),
                    credential: None,
                    provider_key: "groq".to_owned(),
                    id: "groq".to_owned(),
                    provider: "GroqCloud".to_owned(),
                    model: model.to_owned(),
                    route_class: ModelRouteClass::Free,
                    display_name: model.to_owned(),
                    quantization: None,
                    status: 0,
                    context_length: Some(131_072),
                    max_prompt_tokens: None,
                    max_completion_tokens: None,
                    pricing: Pricing::default(),
                    stats: EndpointStats::default(),
                    capabilities: EndpointCapabilities::default(),
                },
                expected_cost_usd: 0.0,
                expected_response_p75_seconds: None,
                expected_response_p95_seconds: None,
            }],
        }
    }

    #[tokio::test]
    async fn inventory_keeps_unclassified_models_nonfree_and_claimless()
    -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let body = serde_json::json!({
            "object": "list",
            "data": [
                {
                    "id": "openai/gpt-oss-120b",
                    "owned_by": "OpenAI",
                    "active": true,
                    "context_window": 131_072
                },
                {
                    "id": "meta-models/muse-spark-open",
                    "owned_by": "Meta",
                    "active": true,
                    "context_window": 131_072
                }
            ]
        })
        .to_string();
        let server = thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buffer = [0_u8; 4096];
                let _ = stream.read(&mut buffer);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        let provider = Groq::new_free("test").with_base_url(format!("http://{address}"));
        let inventory = provider.inventory(&ModelCatalog::default()).await?;
        let _ = server.join();
        let free = inventory
            .models
            .iter()
            .find(|model| model.native_id == "openai/gpt-oss-120b")
            .ok_or("missing free model")?;
        assert_eq!(free.route_class, Some(ModelRouteClass::Free));
        assert!(free.pricing.is_some());
        assert!(free.claimed_parameters.contains("structured_outputs"));
        let unknown = inventory
            .models
            .iter()
            .find(|model| model.native_id == "meta-models/muse-spark-open")
            .ok_or("missing unclassified model")?;
        assert_eq!(unknown.route_class, None);
        assert!(unknown.pricing.is_none());
        assert!(unknown.claimed_parameters.is_empty());
        Ok(())
    }

    #[test]
    fn free_provider_routes_only_documented_free_chat_models() {
        for model in [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-safeguard-20b",
            "qwen/qwen3.8-27b",
        ] {
            assert!(Groq::is_free_chat_model(model));
        }
        assert!(!Groq::is_free_chat_model("whisper-large-v3"));
        assert!(!Groq::is_free_chat_model("meta-models/muse-spark-open"));
        assert!(!Groq::is_free_chat_model(
            "meta-llama/llama-prompt-guard-2-22m"
        ));
    }

    #[test]
    fn groq_profile_uses_completion_tokens_and_clamps_zero_temperature() -> Result<(), ProviderError>
    {
        let provider = Groq::new_free("test");
        let mut request = InferenceRequest::new("openai/gpt-oss-120b", "hello", 16);
        request.max_output_tokens = Some(32);
        request.temperature = Some(0.0);
        request.requirements.reasoning = ReasoningEffort::Low;
        request.requirements.response_schema = Some(ResponseSchema::JsonObject);
        let body = provider.inner.compile_body(&request)?;
        assert_eq!(
            body.get("max_completion_tokens").and_then(Value::as_u64),
            Some(32)
        );
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body.get("temperature").and_then(Value::as_f64), Some(1e-8));
        assert_eq!(
            body.get("reasoning_effort").and_then(Value::as_str),
            Some("low")
        );
        assert_eq!(
            body.pointer("/response_format/type")
                .and_then(Value::as_str),
            Some("json_object")
        );
        Ok(())
    }

    #[test]
    fn model_specific_claims_match_current_groq_schema_matrix() {
        let mut gpt = groq_plan("openai/gpt-oss-120b")
            .candidates
            .remove(0)
            .endpoint;
        let mut safeguard = groq_plan("openai/gpt-oss-safeguard-20b")
            .candidates
            .remove(0)
            .endpoint;
        let mut qwen = groq_plan("qwen/qwen3.8-27b").candidates.remove(0).endpoint;
        let mut other = groq_plan("meta-llama/whatever")
            .candidates
            .remove(0)
            .endpoint;
        for endpoint in [&mut gpt, &mut safeguard, &mut qwen, &mut other] {
            endpoint.capabilities.adapter_parameters = [
                "response_format",
                "json_schema",
                "structured_outputs",
                "reasoning",
                "reasoning_effort",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect();
            Groq::apply_model_claims(endpoint);
        }
        assert!(gpt.capabilities.supports_parameter("structured_outputs"));
        assert!(gpt.capabilities.supports_parameter("json_schema"));
        assert!(safeguard.capabilities.supports_parameter("json_schema"));
        assert!(
            !safeguard
                .capabilities
                .supports_parameter("structured_outputs")
        );
        assert!(qwen.capabilities.supports_parameter("json_schema"));
        assert!(qwen.capabilities.supports_parameter("structured_outputs"));
        assert!(qwen.capabilities.supports_parameter("reasoning_effort"));
        assert!(!other.capabilities.supports_parameter("reasoning_effort"));
    }
}
