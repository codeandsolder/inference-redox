use std::collections::BTreeSet;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::sleep;

use crate::{
    Endpoint, EndpointCapabilities, EndpointStats, InferenceProvider, InferenceRequest,
    InferenceResponse, Percentiles, Pricing, ProviderError, ReasoningEffort, ReasoningSupport,
    ResponseSchema, RetryPolicy, RoutePlan,
};

const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// `OpenRouter` adapter using its native chat-completions and endpoint-catalog grammars.
#[derive(Debug, Clone)]
pub struct OpenRouter {
    client: Client,
    api_key: String,
    base_url: String,
    app_title: Option<String>,
    http_referer: Option<String>,
}

impl OpenRouter {
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_owned(),
            app_title: None,
            http_referer: None,
        }
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        self.base_url.clear();
        self.base_url.push_str(base_url.trim_end_matches('/'));
        self
    }

    #[must_use]
    pub fn with_app_attribution(
        mut self,
        title: impl Into<String>,
        http_referer: impl Into<String>,
    ) -> Self {
        self.app_title = Some(title.into());
        self.http_referer = Some(http_referer.into());
        self
    }

    fn request(&self, method: reqwest::Method, url: String) -> reqwest::RequestBuilder {
        let mut request = self.client.request(method, url).bearer_auth(&self.api_key);
        if let Some(title) = self.app_title.as_ref() {
            request = request.header("X-Title", title);
        }
        if let Some(referer) = self.http_referer.as_ref() {
            request = request.header("HTTP-Referer", referer);
        }
        request
    }

    fn model_endpoint_url(&self, model: &str) -> Result<String, ProviderError> {
        let (author, slug) = model.split_once('/').ok_or_else(|| {
            ProviderError::Catalog(format!("OpenRouter model must be author/slug: {model}"))
        })?;
        Ok(format!(
            "{}/models/{author}/{slug}/endpoints",
            self.base_url
        ))
    }

    fn provider_order(plan: &RoutePlan) -> Vec<String> {
        plan.candidates
            .iter()
            .map(|candidate| candidate.endpoint.id.clone())
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
                    "strict": strict,
                    "schema": schema,
                }
            }),
        }
    }

    fn compile_body(request: &InferenceRequest, plan: &RoutePlan) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("model".to_owned(), json!(request.model));
        object.insert(
            "messages".to_owned(),
            json!([{"role": "user", "content": request.prompt}]),
        );
        object.insert(
            "max_completion_tokens".to_owned(),
            json!(request.expected_output_tokens),
        );
        let order = Self::provider_order(plan);
        let max_prompt_price = plan
            .candidates
            .iter()
            .map(|candidate| candidate.endpoint.pricing.prompt_per_token)
            .fold(0.0_f64, f64::max)
            * 1_000_000.0;
        let max_completion_price = plan
            .candidates
            .iter()
            .map(|candidate| candidate.endpoint.pricing.completion_per_token)
            .fold(0.0_f64, f64::max)
            * 1_000_000.0;
        object.insert(
            "provider".to_owned(),
            json!({
                "order": order,
                "only": Self::provider_order(plan),
                "allow_fallbacks": true,
                "require_parameters": true,
                "max_price": {
                    "prompt": max_prompt_price,
                    "completion": max_completion_price,
                },
            }),
        );
        if let Some(schema) = request.requirements.response_schema.as_ref() {
            object.insert("response_format".to_owned(), Self::response_format(schema));
        }
        if request.requirements.reasoning != ReasoningEffort::None {
            object.insert(
                "reasoning".to_owned(),
                json!({"effort": request.requirements.reasoning.as_str()}),
            );
        }
        Value::Object(object)
    }

    fn retryable_status(status: StatusCode) -> bool {
        status == StatusCode::REQUEST_TIMEOUT
            || status == StatusCode::TOO_MANY_REQUESTS
            || status.is_server_error()
    }

    fn next_backoff(current: Duration, max: Duration) -> Duration {
        current.saturating_mul(2).min(max)
    }
}

#[derive(Debug, Deserialize)]
struct EndpointEnvelope {
    data: EndpointCatalog,
}

#[derive(Debug, Deserialize)]
struct EndpointCatalog {
    endpoints: Vec<OpenRouterEndpoint>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterEndpoint {
    context_length: Option<u64>,
    latency_last_30m: Option<Percentiles>,
    max_completion_tokens: Option<u64>,
    max_prompt_tokens: Option<u64>,
    model_id: Option<String>,
    name: Option<String>,
    pricing: OpenRouterPricing,
    provider_name: Option<String>,
    quantization: Option<String>,
    status: Option<i32>,
    supported_parameters: Option<Vec<String>>,
    supports_implicit_caching: Option<bool>,
    tag: String,
    throughput_last_30m: Option<Percentiles>,
    uptime_last_1d: Option<f64>,
    uptime_last_30m: Option<f64>,
    uptime_last_5m: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterPricing {
    prompt: Option<String>,
    completion: Option<String>,
    request: Option<String>,
    discount: Option<f64>,
    cache_read: Option<String>,
    cache_write: Option<String>,
}

fn parse_price(value: Option<&str>) -> f64 {
    value.and_then(|value| value.parse().ok()).unwrap_or(0.0)
}

impl OpenRouterEndpoint {
    fn normalize(self, requested_model: &str) -> Endpoint {
        let parameters: BTreeSet<String> = self
            .supported_parameters
            .unwrap_or_default()
            .into_iter()
            .collect();
        let reasoning =
            if parameters.contains("reasoning") || parameters.contains("reasoning_effort") {
                ReasoningSupport::GatewayNormalized
            } else {
                ReasoningSupport::None
            };
        Endpoint {
            id: self.tag,
            provider: self.provider_name.unwrap_or_else(|| "unknown".to_owned()),
            model: self.model_id.unwrap_or_else(|| requested_model.to_owned()),
            display_name: self.name.unwrap_or_else(|| requested_model.to_owned()),
            quantization: self.quantization,
            status: self.status.unwrap_or(0),
            context_length: self.context_length,
            max_prompt_tokens: self.max_prompt_tokens,
            max_completion_tokens: self.max_completion_tokens,
            pricing: Pricing {
                prompt_per_token: parse_price(self.pricing.prompt.as_deref()),
                completion_per_token: parse_price(self.pricing.completion.as_deref()),
                request: parse_price(self.pricing.request.as_deref()),
                discount: self.pricing.discount.unwrap_or(0.0).clamp(0.0, 1.0),
                cache_read_per_token: self
                    .pricing
                    .cache_read
                    .as_deref()
                    .and_then(|v| v.parse().ok()),
                cache_write_per_token: self
                    .pricing
                    .cache_write
                    .as_deref()
                    .and_then(|v| v.parse().ok()),
            },
            stats: EndpointStats {
                latency_seconds: self.latency_last_30m.unwrap_or_default(),
                throughput_tokens_per_second: self.throughput_last_30m.unwrap_or_default(),
                uptime_5m: self.uptime_last_5m,
                uptime_30m: self.uptime_last_30m,
                uptime_1d: self.uptime_last_1d,
            },
            capabilities: EndpointCapabilities {
                supported_parameters: parameters,
                reasoning,
                supports_implicit_caching: self.supports_implicit_caching.unwrap_or(false),
            },
        }
    }
}

#[async_trait]
impl InferenceProvider for OpenRouter {
    async fn endpoints(&self, model: &str) -> Result<Vec<Endpoint>, ProviderError> {
        let url = self.model_endpoint_url(model)?;
        let response = self
            .request(reqwest::Method::GET, url)
            .send()
            .await
            .map_err(|error| ProviderError::Catalog(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::Catalog(format!("HTTP {status}: {body}")));
        }
        let envelope: EndpointEnvelope = response
            .json()
            .await
            .map_err(|error| ProviderError::Catalog(error.to_string()))?;
        Ok(envelope
            .data
            .endpoints
            .into_iter()
            .map(|endpoint| endpoint.normalize(model))
            .collect())
    }

    async fn execute_plan(
        &self,
        request: &InferenceRequest,
        plan: &RoutePlan,
        retry: RetryPolicy,
    ) -> Result<InferenceResponse, ProviderError> {
        if retry.max_attempts == 0 {
            return Err(ProviderError::Request(
                "retry.max_attempts must be at least 1".to_owned(),
            ));
        }
        let url = format!("{}/chat/completions", self.base_url);
        let body = Self::compile_body(request, plan);
        let mut backoff = retry.initial_backoff;
        let mut last_error = None;

        for attempt in 1..=retry.max_attempts {
            match self
                .request(reqwest::Method::POST, url.clone())
                .json(&body)
                .send()
                .await
            {
                Ok(response) if response.status().is_success() => {
                    let raw: Value = response
                        .json()
                        .await
                        .map_err(|error| ProviderError::Response(error.to_string()))?;
                    let content = raw
                        .pointer("/choices/0/message/content")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    return Ok(InferenceResponse {
                        content,
                        model: raw.get("model").and_then(Value::as_str).map(str::to_owned),
                        provider: raw
                            .get("provider")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                        usage_cost_usd: raw.pointer("/usage/cost").and_then(Value::as_f64),
                        raw,
                    });
                }
                Ok(response) => {
                    let status = response.status();
                    let retryable = Self::retryable_status(status);
                    let text = response.text().await.unwrap_or_default();
                    last_error = Some(format!("HTTP {status}: {text}"));
                    if !retryable || attempt == retry.max_attempts {
                        break;
                    }
                }
                Err(error) => {
                    last_error = Some(error.to_string());
                    if attempt == retry.max_attempts {
                        break;
                    }
                }
            }
            sleep(backoff).await;
            backoff = Self::next_backoff(backoff, retry.max_backoff);
        }

        Err(ProviderError::Request(last_error.unwrap_or_else(|| {
            "request failed without an error body".to_owned()
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_openrouter_provider_order_and_features() {
        let request = InferenceRequest {
            model: "openai/example".to_owned(),
            prompt: "hello".to_owned(),
            expected_output_tokens: 42,
            prompt_tokens: Some(5),
            requirements: crate::RequestRequirements {
                response_schema: Some(ResponseSchema::JsonSchema {
                    name: "answer".to_owned(),
                    schema: json!({"type":"object"}),
                    strict: true,
                }),
                reasoning: ReasoningEffort::Low,
                ..crate::RequestRequirements::default()
            },
        };
        let endpoint = Endpoint {
            id: "fast-provider/fp8".to_owned(),
            provider: "Fast Provider".to_owned(),
            model: request.model.clone(),
            display_name: "Fast".to_owned(),
            quantization: Some("fp8".to_owned()),
            status: 0,
            context_length: Some(1000),
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            stats: EndpointStats::default(),
            capabilities: EndpointCapabilities::default(),
        };
        let plan = RoutePlan {
            prompt_tokens: 5,
            expected_output_tokens: 42,
            cheapest_cost_usd: 0.0,
            cost_ceiling_usd: 0.0,
            candidates: vec![crate::RouteCandidate {
                endpoint,
                expected_cost_usd: 0.0,
                expected_response_p75_seconds: None,
                expected_response_p95_seconds: None,
            }],
        };
        let body = OpenRouter::compile_body(&request, &plan);
        assert_eq!(
            body.pointer("/provider/order/0").and_then(Value::as_str),
            Some("fast-provider/fp8")
        );
        assert_eq!(
            body.pointer("/reasoning/effort").and_then(Value::as_str),
            Some("low")
        );
        assert_eq!(
            body.pointer("/response_format/type")
                .and_then(Value::as_str),
            Some("json_schema")
        );
    }
}
