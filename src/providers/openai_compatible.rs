use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::{Client, StatusCode, header::RETRY_AFTER};
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tokio::time::sleep;

use crate::{
    Endpoint, EndpointCapabilities, EndpointStats, InferenceProvider, InferenceRequest,
    InferenceResponse, ModelCatalog, ModelRouteClass, NativeModelRecord, Pricing, ProviderError,
    ProviderInventory, ReasoningEffort, ReasoningSupport, ResponseSchema, RetryPolicy, RoutePlan,
    TokenUsage,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenLimitField {
    MaxTokens,
    MaxCompletionTokens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReasoningDialect {
    Unsupported,
    EffortField,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum TemperatureZeroPolicy {
    PassThrough,
    ClampTo(f64),
}

#[derive(Debug, Clone)]
pub(crate) struct OpenAiCompatibleProfile {
    pub source: String,
    pub provider_key: String,
    pub provider_name: String,
    pub base_url: String,
    pub route_class: ModelRouteClass,
    pub token_limit_field: TokenLimitField,
    pub reasoning_dialect: ReasoningDialect,
    pub temperature_zero: TemperatureZeroPolicy,
    pub adapter_parameters: BTreeSet<String>,
    pub nominal_parameters: BTreeSet<String>,
    pub reasoning_support: ReasoningSupport,
    pub supports_implicit_caching: bool,
    pub pricing: Pricing,
    pub usage_cost_usd: Option<f64>,
}

#[derive(Debug, Clone)]
pub(crate) struct OpenAiCompatibleProvider {
    client: Client,
    api_key: Option<String>,
    profile: OpenAiCompatibleProfile,
    model_cache: Arc<Mutex<Option<CachedModels>>>,
    catalog_ttl: Duration,
}

#[derive(Debug, Clone)]
struct CachedModels {
    fetched_at: Instant,
    models: Vec<NativeModelRecord>,
}

#[derive(Debug)]
struct AttemptFailure {
    phase: crate::ExecutionFailurePhase,
    detail: String,
    retryable: bool,
    retry_after: Option<Duration>,
    quota_observations: Vec<crate::QuotaObservation>,
}

impl AttemptFailure {
    fn into_provider_error(self) -> ProviderError {
        ProviderError::Execution {
            phase: self.phase,
            detail: self.detail,
            quota_observations: self.quota_observations,
        }
    }
}

impl OpenAiCompatibleProvider {
    #[must_use]
    pub fn new(api_key: impl Into<String>, profile: OpenAiCompatibleProfile) -> Self {
        Self::new_optional(Some(api_key.into()), profile)
    }

    #[must_use]
    pub fn new_optional(api_key: Option<String>, profile: OpenAiCompatibleProfile) -> Self {
        Self {
            client: Client::new(),
            api_key,
            profile,
            model_cache: Arc::new(Mutex::new(None)),
            catalog_ttl: Duration::from_secs(60),
        }
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        base_url
            .trim_end_matches('/')
            .clone_into(&mut self.profile.base_url);
        self
    }

    #[must_use]
    pub fn with_catalog_ttl(mut self, ttl: Duration) -> Self {
        self.catalog_ttl = ttl;
        self
    }

    fn request(&self, method: reqwest::Method, url: String) -> reqwest::RequestBuilder {
        let request = self.client.request(method, url);
        if let Some(api_key) = self.api_key.as_ref() {
            request.bearer_auth(api_key)
        } else {
            request
        }
    }

    pub async fn inventory(
        &self,
        catalog: &ModelCatalog,
    ) -> Result<ProviderInventory, ProviderError> {
        let models = self.native_models().await?;
        Ok(ProviderInventory::reconcile(
            self.profile.source.clone(),
            models,
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
                .request(
                    reqwest::Method::GET,
                    format!("{}/models", self.profile.base_url),
                )
                .send()
                .await
                .map_err(|error| ProviderError::Catalog(error.to_string()))?;
            let status = response.status();
            if !status.is_success() {
                return Err(ProviderError::Catalog(format!("HTTP {status}")));
            }
            let raw: Value = response
                .json()
                .await
                .map_err(|error| ProviderError::Catalog(error.to_string()))?;
            Self::decode_models(&raw, &self.profile)
        }
        .await;

        match fetched {
            Ok(models) => {
                cache = self.model_cache.lock().await;
                *cache = Some(CachedModels {
                    fetched_at: Instant::now(),
                    models: models.clone(),
                });
                Ok(models)
            }
            Err(error) => stale.ok_or(error),
        }
    }

    fn optional_number(value: Option<&Value>) -> Option<f64> {
        value.and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str().and_then(|text| text.parse::<f64>().ok()))
        })
    }

    fn model_pricing(object: &serde_json::Map<String, Value>) -> Option<Pricing> {
        let pricing = object.get("pricing")?.as_object()?;
        Some(Pricing {
            prompt_per_token: Self::optional_number(pricing.get("prompt"))?,
            completion_per_token: Self::optional_number(pricing.get("completion"))?,
            request: Self::optional_number(pricing.get("request")).unwrap_or(0.0),
            discount: Self::optional_number(pricing.get("discount")).unwrap_or(0.0),
            cache_read_per_token: Self::optional_number(
                pricing
                    .get("input_cache_read")
                    .or_else(|| pricing.get("cache_read")),
            ),
            cache_write_per_token: Self::optional_number(
                pricing
                    .get("input_cache_write")
                    .or_else(|| pricing.get("cache_write")),
            ),
            overrides: Vec::new(),
        })
    }

    fn decode_models(
        raw: &Value,
        profile: &OpenAiCompatibleProfile,
    ) -> Result<Vec<NativeModelRecord>, ProviderError> {
        let data = raw.get("data").and_then(Value::as_array).ok_or_else(|| {
            let top_level = raw
                .as_object()
                .map(|object| object.keys().cloned().collect::<Vec<_>>())
                .unwrap_or_default();
            ProviderError::Catalog(format!(
                "model inventory has no data array; top_level_keys={top_level:?}"
            ))
        })?;
        data.iter()
            .enumerate()
            .filter_map(|(index, value)| {
                let Some(object) = value.as_object() else {
                    return Some(Err(ProviderError::Catalog(format!(
                        "model inventory item {index} is not an object"
                    ))));
                };
                let id = match object.get("id").and_then(Value::as_str) {
                    Some(id) if !id.is_empty() => id.to_owned(),
                    _ => {
                        let keys = object.keys().cloned().collect::<Vec<_>>();
                        return Some(Err(ProviderError::Catalog(format!(
                            "model inventory item {index} has no string id; keys={keys:?}"
                        ))));
                    }
                };
                if object.get("active").and_then(Value::as_bool) == Some(false) {
                    return None;
                }
                let context_length = object
                    .get("context_length")
                    .or_else(|| object.get("context_window"))
                    .and_then(Value::as_u64);
                let top_provider = object.get("top_provider").and_then(Value::as_object);
                let max_completion_tokens = object
                    .get("max_completion_tokens")
                    .and_then(Value::as_u64)
                    .or_else(|| {
                        top_provider
                            .and_then(|provider| provider.get("max_completion_tokens"))
                            .and_then(Value::as_u64)
                    });
                let claimed_parameters = object
                    .get("supported_parameters")
                    .and_then(Value::as_array)
                    .map(|parameters| {
                        parameters
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect::<BTreeSet<_>>()
                    })
                    .filter(|parameters| !parameters.is_empty())
                    .unwrap_or_else(|| profile.nominal_parameters.clone());
                Some(Ok(NativeModelRecord {
                    id,
                    display_name: object
                        .get("name")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    owned_by: object
                        .get("owned_by")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    created_unix: object.get("created").and_then(Value::as_u64),
                    context_length,
                    max_completion_tokens,
                    claimed_parameters,
                    pricing: Self::model_pricing(object).or_else(|| Some(profile.pricing.clone())),
                    route_class: Some(profile.route_class),
                }))
            })
            .collect()
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

    pub(crate) fn compile_body(&self, request: &InferenceRequest) -> Result<Value, ProviderError> {
        let mut messages = Vec::with_capacity(2);
        if let Some(system_prompt) = request.system_prompt.as_ref() {
            messages.push(json!({"role": "system", "content": system_prompt}));
        }
        messages.push(json!({"role": "user", "content": request.prompt}));
        let mut object = serde_json::Map::new();
        object.insert("model".to_owned(), json!(request.model));
        object.insert("messages".to_owned(), Value::Array(messages));

        if let Some(maximum) = request.max_output_tokens {
            let field = match self.profile.token_limit_field {
                TokenLimitField::MaxTokens => "max_tokens",
                TokenLimitField::MaxCompletionTokens => "max_completion_tokens",
            };
            object.insert(field.to_owned(), json!(maximum));
        }
        if let Some(temperature) = request.temperature {
            let temperature = match self.profile.temperature_zero {
                TemperatureZeroPolicy::ClampTo(floor) if temperature == 0.0 => floor,
                TemperatureZeroPolicy::PassThrough | TemperatureZeroPolicy::ClampTo(_) => {
                    temperature
                }
            };
            object.insert("temperature".to_owned(), json!(temperature));
        }
        if let Some(schema) = request.requirements.response_schema.as_ref() {
            object.insert("response_format".to_owned(), Self::response_format(schema));
        }
        if request.requirements.reasoning != ReasoningEffort::None {
            match self.profile.reasoning_dialect {
                ReasoningDialect::Unsupported => {
                    return Err(ProviderError::InvalidRequest(format!(
                        "{} reasoning controls are not normalized yet",
                        self.profile.provider_name
                    )));
                }
                ReasoningDialect::EffortField => {
                    object.insert(
                        "reasoning_effort".to_owned(),
                        json!(request.requirements.reasoning.as_str()),
                    );
                }
            }
        }
        Ok(Value::Object(object))
    }

    fn response_shape_detail(&self, raw: &Value) -> String {
        let message_keys = raw
            .pointer("/choices/0/message")
            .and_then(Value::as_object)
            .map(|message| message.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let finish_reason = raw
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            .unwrap_or("missing");
        format!(
            "{} response had no non-empty string content; message_keys={message_keys:?}; finish_reason={finish_reason}",
            self.profile.provider_name
        )
    }

    async fn execute_once(
        &self,
        url: &str,
        body: &Value,
    ) -> Result<InferenceResponse, AttemptFailure> {
        let response = self
            .request(reqwest::Method::POST, url.to_owned())
            .json(body)
            .send()
            .await
            .map_err(|error| AttemptFailure {
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
            return Err(AttemptFailure {
                phase: crate::ExecutionFailurePhase::ResponseReceived,
                detail: format!("HTTP {status}"),
                retryable: Self::retryable_status(status),
                retry_after,
                quota_observations,
            });
        }
        let raw: Value = response.json().await.map_err(|error| AttemptFailure {
            phase: crate::ExecutionFailurePhase::ResponseReceived,
            detail: error.to_string(),
            retryable: false,
            retry_after,
            quota_observations: quota_observations.clone(),
        })?;
        let Some(content) = raw
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .filter(|content| !content.is_empty())
        else {
            return Err(AttemptFailure {
                phase: crate::ExecutionFailurePhase::ResponseReceived,
                detail: self.response_shape_detail(&raw),
                retryable: false,
                retry_after,
                quota_observations,
            });
        };
        Ok(InferenceResponse {
            content: content.to_owned(),
            model: raw.get("model").and_then(Value::as_str).map(str::to_owned),
            provider: Some(self.profile.source.clone()),
            usage_cost_usd: self.profile.usage_cost_usd,
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
impl InferenceProvider for OpenAiCompatibleProvider {
    async fn endpoints(&self, model: &str) -> Result<Vec<Endpoint>, ProviderError> {
        let models = self.native_models().await?;
        let Some(native) = models.iter().find(|candidate| candidate.id == model) else {
            return Err(ProviderError::NoEligibleEndpoint(model.to_owned()));
        };
        Ok(vec![Endpoint {
            source: self.profile.source.clone(),
            credential: None,
            provider_key: self.profile.provider_key.clone(),
            id: self.profile.provider_key.clone(),
            provider: self.profile.provider_name.clone(),
            model: model.to_owned(),
            route_class: self.profile.route_class,
            display_name: native
                .display_name
                .clone()
                .unwrap_or_else(|| model.to_owned()),
            quantization: None,
            status: 0,
            context_length: native.context_length,
            max_prompt_tokens: None,
            max_completion_tokens: native.max_completion_tokens,
            pricing: native
                .pricing
                .clone()
                .unwrap_or_else(|| self.profile.pricing.clone()),
            stats: EndpointStats::default(),
            capabilities: EndpointCapabilities {
                adapter_parameters: self.profile.adapter_parameters.clone(),
                supported_parameters: native.claimed_parameters.clone(),
                verified_parameters: BTreeSet::new(),
                rejected_parameters: BTreeSet::new(),
                reasoning: self.profile.reasoning_support.clone(),
                supports_implicit_caching: self.profile.supports_implicit_caching,
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
        if candidate.endpoint.source != self.profile.source
            || candidate.endpoint.provider_key != self.profile.provider_key
        {
            return Err(ProviderError::InvalidRequest(format!(
                "{} adapter received a route for {}/{}",
                self.profile.provider_name,
                candidate.endpoint.source,
                candidate.endpoint.provider_key
            )));
        }
        let body = self.compile_body(request)?;
        let url = format!("{}/chat/completions", self.profile.base_url);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> OpenAiCompatibleProfile {
        OpenAiCompatibleProfile {
            source: "test".to_owned(),
            provider_key: "test".to_owned(),
            provider_name: "Test".to_owned(),
            base_url: "http://example.invalid/v1".to_owned(),
            route_class: ModelRouteClass::Free,
            token_limit_field: TokenLimitField::MaxCompletionTokens,
            reasoning_dialect: ReasoningDialect::EffortField,
            temperature_zero: TemperatureZeroPolicy::ClampTo(1e-8),
            adapter_parameters: [
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
            .collect(),
            nominal_parameters: ["chat_completions", "system_role", "temperature"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            reasoning_support: ReasoningSupport::GatewayNormalized,
            supports_implicit_caching: false,
            pricing: Pricing::default(),
            usage_cost_usd: Some(0.0),
        }
    }

    #[test]
    fn optional_auth_provider_compiles_requests_without_fake_credentials()
    -> Result<(), ProviderError> {
        let provider = OpenAiCompatibleProvider::new_optional(None, profile());
        let request = InferenceRequest::new("vendor/model", "hi", 8);
        let body = provider.compile_body(&request)?;
        assert_eq!(
            body.get("model").and_then(Value::as_str),
            Some("vendor/model")
        );
        Ok(())
    }

    #[test]
    fn profile_controls_token_temperature_and_reasoning_wire_shape() -> Result<(), ProviderError> {
        let provider = OpenAiCompatibleProvider::new("key", profile());
        let mut request = InferenceRequest::new("vendor/model", "hi", 8);
        request.max_output_tokens = Some(32);
        request.temperature = Some(0.0);
        request.requirements.reasoning = ReasoningEffort::Low;
        let body = provider.compile_body(&request)?;
        assert_eq!(
            body.get("max_completion_tokens").and_then(Value::as_u64),
            Some(32)
        );
        assert_eq!(body.get("temperature").and_then(Value::as_f64), Some(1e-8));
        assert_eq!(
            body.get("reasoning_effort").and_then(Value::as_str),
            Some("low")
        );
        Ok(())
    }

    #[test]
    fn model_inventory_preserves_per_model_claims_pricing_and_output_limit()
    -> Result<(), ProviderError> {
        let raw = json!({
            "data": [{
                "id": "vendor/model:free",
                "name": "Vendor Model Free",
                "supported_parameters": ["max_tokens", "tools", 7],
                "pricing": {
                    "prompt": "0.000001",
                    "completion": 0.000_002,
                    "input_cache_read": "0.0000001"
                },
                "top_provider": {"max_completion_tokens": 12345}
            }]
        });
        let models = OpenAiCompatibleProvider::decode_models(&raw, &profile())?;
        let model = &models[0];
        assert_eq!(model.display_name.as_deref(), Some("Vendor Model Free"));
        assert_eq!(model.max_completion_tokens, Some(12_345));
        assert_eq!(
            model.claimed_parameters,
            BTreeSet::from(["max_tokens".to_owned(), "tools".to_owned()])
        );
        let pricing = model
            .pricing
            .as_ref()
            .ok_or_else(|| ProviderError::Catalog("missing normalized pricing".to_owned()))?;
        assert_eq!(pricing.prompt_per_token, 0.000_001);
        assert_eq!(pricing.completion_per_token, 0.000_002);
        assert_eq!(pricing.cache_read_per_token, Some(0.000_000_1));
        Ok(())
    }

    #[test]
    fn model_record_accepts_groq_style_context_window() -> Result<(), ProviderError> {
        let models = OpenAiCompatibleProvider::decode_models(
            &json!({
                "object": "list",
                "data": [{
                    "id": "openai/gpt-oss-120b",
                    "created": 1,
                    "owned_by": "OpenAI",
                    "active": true,
                    "context_window": 131_072
                }]
            }),
            &profile(),
        )?;
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].context_length, Some(131_072));
        Ok(())
    }

    #[test]
    fn model_inventory_ignores_wrong_optional_field_types() -> Result<(), ProviderError> {
        let models = OpenAiCompatibleProvider::decode_models(
            &json!({
                "data": [{
                    "id": "vendor/model",
                    "created": "recently",
                    "owned_by": {"name": "vendor"},
                    "active": "yes",
                    "context_window": "131072",
                    "max_completion_tokens": false
                }]
            }),
            &profile(),
        )?;
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "vendor/model");
        assert_eq!(models[0].created_unix, None);
        assert_eq!(models[0].owned_by, None);
        assert_eq!(models[0].context_length, None);
        assert_eq!(models[0].max_completion_tokens, None);
        Ok(())
    }

    #[test]
    fn model_inventory_requires_data_array_and_string_id() {
        let missing_data =
            OpenAiCompatibleProvider::decode_models(&json!({"models": []}), &profile());
        assert!(matches!(missing_data, Err(ProviderError::Catalog(_))));
        let missing_id = OpenAiCompatibleProvider::decode_models(
            &json!({"data": [{"context_window": 1}]}),
            &profile(),
        );
        assert!(matches!(missing_id, Err(ProviderError::Catalog(_))));
    }
}
