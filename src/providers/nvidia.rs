use std::time::Duration;

use async_trait::async_trait;

use super::openai_compatible::{
    OpenAiCompatibleProfile, OpenAiCompatibleProvider, ReasoningDialect, TemperatureZeroPolicy,
    TokenLimitField,
};
use crate::{
    InferenceProvider, InferenceRequest, InferenceResponse, ModelCatalog, ModelRouteClass, Pricing,
    ProviderError, ProviderInventory, ReasoningSupport, RetryPolicy, RoutePlan,
};

/// Direct NVIDIA NIM OpenAI-compatible provider.
#[derive(Debug, Clone)]
pub struct NvidiaNim {
    inner: OpenAiCompatibleProvider,
}

impl NvidiaNim {
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAiCompatibleProvider::new(
                api_key,
                OpenAiCompatibleProfile {
                    source: "nvidia".to_owned(),
                    provider_key: "nvidia".to_owned(),
                    provider_name: "NVIDIA NIM".to_owned(),
                    base_url: "https://integrate.api.nvidia.com/v1".to_owned(),
                    route_class: ModelRouteClass::Free,
                    token_limit_field: TokenLimitField::MaxTokens,
                    reasoning_dialect: ReasoningDialect::Unsupported,
                    temperature_zero: TemperatureZeroPolicy::PassThrough,
                    adapter_parameters: [
                        "chat_completions",
                        "system_role",
                        "max_tokens",
                        "temperature",
                        "response_format",
                        "json_schema",
                        "structured_outputs",
                    ]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
                    nominal_parameters: [
                        "chat_completions",
                        "system_role",
                        "max_tokens",
                        "temperature",
                    ]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
                    reasoning_support: ReasoningSupport::None,
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

    /// Reconcile NVIDIA's native model listing against the canonical catalog.
    ///
    /// # Errors
    /// Returns a provider catalog error if `/models` cannot be loaded and no stale cache exists.
    pub async fn inventory(
        &self,
        catalog: &ModelCatalog,
    ) -> Result<ProviderInventory, ProviderError> {
        self.inner.inventory(catalog).await
    }
}

#[async_trait]
impl InferenceProvider for NvidiaNim {
    async fn endpoints(&self, model: &str) -> Result<Vec<crate::Endpoint>, ProviderError> {
        self.inner.endpoints(model).await
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
    use crate::{Endpoint, EndpointCapabilities, EndpointStats, ResponseSchema, RouteCandidate};

    fn test_plan(model: &str) -> RoutePlan {
        RoutePlan {
            prompt_tokens: 1,
            expected_output_tokens: 8,
            cheapest_cost_usd: 0.0,
            cost_ceiling_usd: 0.0,
            candidates: vec![RouteCandidate {
                endpoint: Endpoint {
                    source: "nvidia".to_owned(),
                    credential: None,
                    provider_key: "nvidia".to_owned(),
                    id: "nvidia".to_owned(),
                    provider: "NVIDIA NIM".to_owned(),
                    model: model.to_owned(),
                    route_class: ModelRouteClass::Free,
                    display_name: model.to_owned(),
                    quantization: None,
                    status: 0,
                    context_length: None,
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
    async fn http_error_is_response_received_phase() -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buffer = [0_u8; 4096];
                let _ = stream.read(&mut buffer);
                let response = concat!(
                    "HTTP/1.1 429 Too Many Requests\r\n",
                    "X-RateLimit-Limit-Requests: 1000\r\n",
                    "X-RateLimit-Remaining-Requests: 0\r\n",
                    "X-RateLimit-Reset-Requests: 8.5s\r\n",
                    "X-RateLimit-Limit-Tokens: 8000\r\n",
                    "X-RateLimit-Remaining-Tokens: 7777\r\n",
                    "X-RateLimit-Reset-Tokens: 2.25s\r\n",
                    "Content-Length: 0\r\n",
                    "Connection: close\r\n\r\n",
                )
                .as_bytes();
                let _ = stream.write_all(response);
                let _ = stream.flush();
            }
        });
        let provider = NvidiaNim::new("test").with_base_url(format!("http://{address}"));
        let request = InferenceRequest::new("nvidia/example", "hi", 8);
        let retry = RetryPolicy {
            max_attempts: 1,
            ..RetryPolicy::default()
        };
        let error = provider
            .execute_plan(&request, &test_plan(&request.model), retry)
            .await
            .err()
            .ok_or("expected NVIDIA test request to fail")?;
        let _ = server.join();
        assert_eq!(
            error.execution_phase(),
            Some(crate::ExecutionFailurePhase::ResponseReceived)
        );
        assert!(error.to_string().contains("HTTP 429"));
        let observations = error.quota_observations();
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0].remaining, Some(0));
        assert_eq!(observations[1].remaining, Some(7_777));
        Ok(())
    }

    #[tokio::test]
    async fn connection_failure_is_transport_unknown_phase()
    -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        drop(listener);
        let provider = NvidiaNim::new("test").with_base_url(format!("http://{address}"));
        let request = InferenceRequest::new("nvidia/example", "hi", 8);
        let retry = RetryPolicy {
            max_attempts: 1,
            ..RetryPolicy::default()
        };
        let error = provider
            .execute_plan(&request, &test_plan(&request.model), retry)
            .await
            .err()
            .ok_or("expected NVIDIA test transport to fail")?;
        assert_eq!(
            error.execution_phase(),
            Some(crate::ExecutionFailurePhase::TransportUnknown)
        );
        Ok(())
    }

    #[test]
    fn compiles_supported_basic_and_json_requests() -> Result<(), ProviderError> {
        let provider = NvidiaNim::new("test");
        let mut request = InferenceRequest::new("nvidia/example", "hello", 16);
        request.system_prompt = Some("system".to_owned());
        request.max_output_tokens = Some(32);
        request.temperature = Some(0.0);
        request.requirements.response_schema = Some(ResponseSchema::JsonObject);
        let body = provider.inner.compile_body(&request)?;
        assert_eq!(
            body.pointer("/messages/0/role").and_then(Value::as_str),
            Some("system")
        );
        assert_eq!(body.get("max_tokens").and_then(Value::as_u64), Some(32));
        assert!(body.get("max_completion_tokens").is_none());
        assert_eq!(body.get("temperature").and_then(Value::as_f64), Some(0.0));
        assert_eq!(
            body.pointer("/response_format/type")
                .and_then(Value::as_str),
            Some("json_object")
        );
        Ok(())
    }

    #[test]
    fn refuses_unimplemented_reasoning_controls() {
        let provider = NvidiaNim::new("test");
        let mut request = InferenceRequest::new("nvidia/example", "hello", 16);
        request.requirements.reasoning = crate::ReasoningEffort::Low;
        assert!(matches!(
            provider.inner.compile_body(&request),
            Err(ProviderError::InvalidRequest(_))
        ));
    }
}
