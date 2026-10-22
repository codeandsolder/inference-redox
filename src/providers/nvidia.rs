use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::{Client, StatusCode, header::RETRY_AFTER};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tokio::time::sleep;

use crate::{
    Endpoint, EndpointCapabilities, EndpointStats, InferenceProvider, InferenceRequest,
    InferenceResponse, ModelCatalog, ModelRouteClass, NativeModelRecord, Pricing, ProviderError,
    ProviderInventory, ReasoningEffort, ReasoningSupport, ResponseSchema, RetryPolicy, RoutePlan,
    TokenUsage,
};

const DEFAULT_BASE_URL: &str = "https://integrate.api.nvidia.com/v1";

/// NVIDIA hosted NIM adapter.
#[derive(Debug, Clone)]
pub struct NvidiaNim {
    client: Client,
    api_key: String,
    base_url: String,
    model_cache: Arc<Mutex<Option<CachedNvidiaModels>>>,
    catalog_ttl: Duration,
}

#[derive(Debug, Clone)]
struct CachedNvidiaModels {
    fetched_at: Instant,
    models: Vec<NativeModelRecord>,
}

#[derive(Debug)]
struct NvidiaAttemptFailure {
    phase: crate::ExecutionFailurePhase,
    detail: String,
    retryable: bool,
    retry_after: Option<Duration>,
    quota_observations: Vec<crate::QuotaObservation>,
}

impl NvidiaAttemptFailure {
    fn into_provider_error(self) -> ProviderError {
        ProviderError::Execution {
            phase: self.phase,
            detail: self.detail,
            quota_observations: self.quota_observations,
        }
    }
}

impl NvidiaNim {
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

    /// Fetch NVIDIA's provider-native model list and reconcile it against the canonical catalog.
    ///
    /// # Errors
    /// Returns a catalog error when NVIDIA's model inventory cannot be fetched or decoded.
    pub async fn inventory(
        &self,
        catalog: &ModelCatalog,
    ) -> Result<ProviderInventory, ProviderError> {
        let native_models = self.native_models().await?;
        Ok(ProviderInventory::reconcile(
            "nvidia",
            native_models,
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
                let body = response.text().await.unwrap_or_default();
                return Err(ProviderError::Catalog(format!("HTTP {status}: {body}")));
            }
            let envelope: NvidiaModelEnvelope = response
                .json()
                .await
                .map_err(|error| ProviderError::Catalog(error.to_string()))?;
            Ok::<Vec<NativeModelRecord>, ProviderError>(
                envelope
                    .data
                    .into_iter()
                    .map(NvidiaModel::normalize)
                    .collect(),
            )
        }
        .await;

        match fetched {
            Ok(models) => {
                cache = self.model_cache.lock().await;
                *cache = Some(CachedNvidiaModels {
                    fetched_at: Instant::now(),
                    models: models.clone(),
                });
                Ok(models)
            }
            Err(error) => stale.ok_or(error),
        }
    }

    fn response_format(schema: &ResponseSchema) -> Value {
        match schema {
            ResponseSchema::JsonObject => json!({"type": "json_object"}),
            ResponseSchema::JsonSchema {
                name,
                schema,
                strict,
            } => json!({
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": schema,
                    "strict": strict,
                }
            }),
        }
    }

    fn compile_body(request: &InferenceRequest) -> Result<Value, ProviderError> {
        if request.requirements.reasoning != ReasoningEffort::None {
            return Err(ProviderError::InvalidRequest(
                "NVIDIA reasoning controls are not normalized yet".to_owned(),
            ));
        }
        let mut messages = Vec::with_capacity(2);
        if let Some(system_prompt) = request.system_prompt.as_ref() {
            messages.push(json!({"role": "system", "content": system_prompt}));
        }
        messages.push(json!({"role": "user", "content": request.prompt}));
        let mut body = serde_json::Map::new();
        body.insert("model".to_owned(), json!(request.model));
        body.insert("messages".to_owned(), Value::Array(messages));
        if let Some(maximum) = request.max_output_tokens {
            body.insert("max_tokens".to_owned(), json!(maximum));
        }
        if let Some(temperature) = request.temperature {
            body.insert("temperature".to_owned(), json!(temperature));
        }
        if let Some(schema) = request.requirements.response_schema.as_ref() {
            body.insert("response_format".to_owned(), Self::response_format(schema));
        }
        Ok(Value::Object(body))
    }

    fn response_shape_detail(raw: &Value) -> String {
        let message_keys = raw
            .pointer("/choices/0/message")
            .and_then(Value::as_object)
            .map(|message| message.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let finish_reason = raw
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            .unwrap_or("missing");
        let prompt_tokens = raw.pointer("/usage/prompt_tokens").and_then(Value::as_u64);
        let completion_tokens = raw
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64);
        format!(
            "NVIDIA response had no non-empty string content; message_keys={message_keys:?}; finish_reason={finish_reason}; prompt_tokens={prompt_tokens:?}; completion_tokens={completion_tokens:?}"
        )
    }

    async fn execute_once(
        &self,
        url: &str,
        body: &Value,
    ) -> Result<InferenceResponse, NvidiaAttemptFailure> {
        let response = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(body)
            .send()
            .await
            .map_err(|error| NvidiaAttemptFailure {
                phase: crate::ExecutionFailurePhase::TransportUnknown,
                detail: error.to_string(),
                retryable: true,
                retry_after: None,
                quota_observations: Vec::new(),
            })?;
        let status = response.status();
        let retry_after = Self::retry_after(&response);
        let quota_observations =
            crate::quota_headers::standard_rate_limit_observations(response.headers());
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(NvidiaAttemptFailure {
                phase: crate::ExecutionFailurePhase::ResponseReceived,
                detail: format!("HTTP {status}: {text}"),
                retryable: Self::retryable_status(status),
                retry_after,
                quota_observations,
            });
        }
        let raw: Value = response
            .json()
            .await
            .map_err(|error| NvidiaAttemptFailure {
                phase: crate::ExecutionFailurePhase::ResponseReceived,
                detail: error.to_string(),
                retryable: false,
                retry_after,
                quota_observations: quota_observations.clone(),
            })?;
        let Some(content) = raw
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
        else {
            return Err(NvidiaAttemptFailure {
                phase: crate::ExecutionFailurePhase::ResponseReceived,
                detail: Self::response_shape_detail(&raw),
                retryable: false,
                retry_after,
                quota_observations,
            });
        };
        if content.is_empty() {
            return Err(NvidiaAttemptFailure {
                phase: crate::ExecutionFailurePhase::ResponseReceived,
                detail: Self::response_shape_detail(&raw),
                retryable: false,
                retry_after,
                quota_observations,
            });
        }
        Ok(InferenceResponse {
            content: content.to_owned(),
            model: raw.get("model").and_then(Value::as_str).map(str::to_owned),
            provider: Some("nvidia".to_owned()),
            usage_cost_usd: Some(0.0),
            usage: TokenUsage::from_openai_compatible(&raw),
            quota_observations,
            raw,
        })
    }

    fn retryable_status(status: StatusCode) -> bool {
        status == StatusCode::REQUEST_TIMEOUT
            || status == StatusCode::TOO_MANY_REQUESTS
            || status.is_server_error()
    }

    fn retry_after(response: &reqwest::Response) -> Option<Duration> {
        response
            .headers()
            .get(RETRY_AFTER)?
            .to_str()
            .ok()?
            .parse::<u64>()
            .ok()
            .map(Duration::from_secs)
    }
}

#[async_trait]
impl InferenceProvider for NvidiaNim {
    async fn endpoints(&self, model: &str) -> Result<Vec<Endpoint>, ProviderError> {
        let models = self.native_models().await?;
        if !models.iter().any(|candidate| candidate.id == model) {
            return Err(ProviderError::NoEligibleEndpoint(model.to_owned()));
        }
        Ok(vec![Endpoint {
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
            capabilities: EndpointCapabilities {
                adapter_parameters: [
                    "chat_completions",
                    "system_role",
                    "max_tokens",
                    "temperature",
                    "response_format",
                    "structured_outputs",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
                supported_parameters: [
                    "chat_completions",
                    "system_role",
                    "max_tokens",
                    "temperature",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
                verified_parameters: BTreeSet::new(),
                rejected_parameters: BTreeSet::new(),
                reasoning: ReasoningSupport::None,
                supports_implicit_caching: false,
            },
        }])
    }

    async fn execute_plan(
        &self,
        request: &InferenceRequest,
        plan: &RoutePlan,
        retry: RetryPolicy,
    ) -> Result<InferenceResponse, ProviderError> {
        if retry.max_attempts == 0 {
            return Err(ProviderError::InvalidRequest(
                "retry.max_attempts must be at least 1".to_owned(),
            ));
        }
        let Some(candidate) = plan.candidates.first() else {
            return Err(ProviderError::NoEligibleEndpoint(request.model.clone()));
        };
        if candidate.endpoint.source != "nvidia" || candidate.endpoint.provider_key != "nvidia" {
            return Err(ProviderError::InvalidRequest(
                "NVIDIA adapter received a non-NVIDIA route".to_owned(),
            ));
        }
        let body = Self::compile_body(request)?;
        let url = format!("{}/chat/completions", self.base_url);
        let mut backoff = retry.initial_backoff;
        let mut last_error = None;
        let mut last_phase = None;
        let mut last_quota_observations = Vec::new();

        for attempt in 1..=retry.max_attempts {
            match self.execute_once(&url, &body).await {
                Ok(response) => return Ok(response),
                Err(failure) if !failure.retryable => return Err(failure.into_provider_error()),
                Err(failure) => {
                    if last_phase.is_none()
                        || failure.phase == crate::ExecutionFailurePhase::ResponseReceived
                    {
                        last_phase = Some(failure.phase);
                    }
                    if !failure.quota_observations.is_empty() {
                        last_quota_observations = failure.quota_observations;
                    }
                    last_error = Some(failure.detail);
                    if attempt < retry.max_attempts {
                        let delay = failure
                            .retry_after
                            .unwrap_or(backoff)
                            .min(retry.max_backoff);
                        sleep(delay).await;
                        backoff = backoff.saturating_mul(2).min(retry.max_backoff);
                    }
                }
            }
        }
        Err(ProviderError::Execution {
            phase: last_phase.unwrap_or(crate::ExecutionFailurePhase::TransportUnknown),
            detail: last_error.unwrap_or_else(|| "request failed without an error body".to_owned()),
            quota_observations: last_quota_observations,
        })
    }
}

#[derive(Debug, Deserialize)]
struct NvidiaModelEnvelope {
    data: Vec<NvidiaModel>,
}

#[derive(Debug, Deserialize)]
struct NvidiaModel {
    id: String,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    owned_by: Option<String>,
}

impl NvidiaModel {
    fn normalize(self) -> NativeModelRecord {
        NativeModelRecord {
            id: self.id,
            display_name: None,
            owned_by: self.owned_by,
            created_unix: self.created,
            context_length: None,
            max_completion_tokens: None,
            claimed_parameters: BTreeSet::new(),
            pricing: Some(Pricing::default()),
            route_class: Some(ModelRouteClass::Free),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use super::*;

    fn test_plan(model: &str) -> RoutePlan {
        RoutePlan {
            prompt_tokens: 1,
            expected_output_tokens: 8,
            cheapest_cost_usd: 0.0,
            cost_ceiling_usd: 0.0,
            candidates: vec![crate::RouteCandidate {
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
        assert_eq!(observations[0].bucket, "rate_limit.requests");
        assert_eq!(observations[0].limit, Some(1_000));
        assert_eq!(observations[0].remaining, Some(0));
        assert_eq!(observations[0].reset_after_millis, Some(8_500));
        assert_eq!(observations[1].bucket, "rate_limit.tokens");
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
    fn parses_openai_compatible_nvidia_model_record() -> Result<(), Box<dyn std::error::Error>> {
        let envelope: NvidiaModelEnvelope = serde_json::from_value(serde_json::json!({
            "object": "list",
            "data": [{
                "id": "nvidia/nemotron-3-ultra-550b-a55b",
                "object": "model",
                "created": 1_790_000_000,
                "owned_by": "nvidia"
            }]
        }))?;
        let normalized = envelope
            .data
            .into_iter()
            .next()
            .ok_or("missing model")?
            .normalize();
        assert_eq!(normalized.id, "nvidia/nemotron-3-ultra-550b-a55b");
        assert_eq!(normalized.owned_by.as_deref(), Some("nvidia"));
        assert_eq!(normalized.route_class, Some(ModelRouteClass::Free));
        Ok(())
    }

    #[test]
    fn compiles_supported_basic_and_json_requests() -> Result<(), ProviderError> {
        let mut request = InferenceRequest::new("nvidia/example", "hello", 16);
        request.system_prompt = Some("system".to_owned());
        request.max_output_tokens = Some(32);
        request.temperature = Some(0.0);
        request.requirements.response_schema = Some(ResponseSchema::JsonObject);
        let body = NvidiaNim::compile_body(&request)?;
        assert_eq!(
            body.pointer("/messages/0/role").and_then(Value::as_str),
            Some("system")
        );
        assert_eq!(body.get("max_tokens").and_then(Value::as_u64), Some(32));
        assert_eq!(
            body.pointer("/response_format/type")
                .and_then(Value::as_str),
            Some("json_object")
        );
        Ok(())
    }

    #[test]
    fn refuses_unimplemented_reasoning_controls() {
        let mut request = InferenceRequest::new("nvidia/example", "hello", 16);
        request.requirements.reasoning = ReasoningEffort::Low;
        assert!(matches!(
            NvidiaNim::compile_body(&request),
            Err(ProviderError::InvalidRequest(_))
        ));
    }
}
