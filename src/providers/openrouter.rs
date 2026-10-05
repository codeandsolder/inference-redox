use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use reqwest::{Client, StatusCode, header::RETRY_AFTER};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tokio::time::sleep;

use crate::{
    Endpoint, EndpointCapabilities, EndpointStats, InferenceProvider, InferenceRequest,
    InferenceResponse, Percentiles, Pricing, PricingOverride, ProviderError, ReasoningEffort,
    ReasoningSupport, ResponseSchema, RetryPolicy, RouteCandidate, RoutePlan,
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
    catalog_cache: Arc<Mutex<HashMap<String, CachedEndpoints>>>,
    catalog_ttl: Duration,
}

#[derive(Debug, Clone)]
struct CachedEndpoints {
    fetched_at: Instant,
    endpoints: Vec<Endpoint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenRouterServiceTier {
    Flex,
    Priority,
    Scale,
}

impl OpenRouterServiceTier {
    fn as_str(self) -> &'static str {
        match self {
            Self::Flex => "flex",
            Self::Priority => "priority",
            Self::Scale => "scale",
        }
    }
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
            catalog_cache: Arc::new(Mutex::new(HashMap::new())),
            catalog_ttl: Duration::from_secs(60),
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

    #[must_use]
    pub fn with_catalog_ttl(mut self, ttl: Duration) -> Self {
        self.catalog_ttl = ttl;
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

    fn service_tier(endpoint_id: &str) -> Option<OpenRouterServiceTier> {
        match endpoint_id.rsplit_once('/').map(|(_, suffix)| suffix) {
            Some("flex") => Some(OpenRouterServiceTier::Flex),
            Some("priority") => Some(OpenRouterServiceTier::Priority),
            Some("scale") => Some(OpenRouterServiceTier::Scale),
            _ => None,
        }
    }

    fn provider_selector(candidate: &RouteCandidate) -> String {
        if Self::service_tier(&candidate.endpoint.id).is_some() {
            candidate.endpoint.id.split_once('/').map_or_else(
                || candidate.endpoint.id.clone(),
                |(base, _)| base.to_owned(),
            )
        } else {
            candidate.endpoint.id.clone()
        }
    }

    fn route_batches(plan: &RoutePlan) -> Vec<Vec<&RouteCandidate>> {
        let mut batches: Vec<Vec<&RouteCandidate>> = Vec::new();
        for candidate in &plan.candidates {
            let tier = Self::service_tier(&candidate.endpoint.id);
            let append = batches.last().is_some_and(|batch| {
                batch
                    .first()
                    .is_some_and(|first| Self::service_tier(&first.endpoint.id) == tier)
            });
            if append {
                let index = batches.len() - 1;
                batches[index].push(candidate);
            } else {
                batches.push(vec![candidate]);
            }
        }
        batches
    }

    fn selectors(candidates: &[&RouteCandidate]) -> Vec<String> {
        let mut selectors = Vec::new();
        for candidate in candidates {
            let selector = Self::provider_selector(candidate);
            if !selectors.contains(&selector) {
                selectors.push(selector);
            }
        }
        selectors
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

    fn compile_body(
        request: &InferenceRequest,
        plan: &RoutePlan,
        candidates: &[&RouteCandidate],
    ) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("model".to_owned(), json!(request.model));
        let mut messages = Vec::with_capacity(2);
        if let Some(system_prompt) = request.system_prompt.as_ref() {
            messages.push(json!({"role": "system", "content": system_prompt}));
        }
        messages.push(json!({"role": "user", "content": request.prompt}));
        object.insert("messages".to_owned(), Value::Array(messages));
        if let Some(maximum) = request.max_output_tokens {
            let use_completion_name = candidates.first().is_some_and(|candidate| {
                candidate
                    .endpoint
                    .capabilities
                    .supported_parameters
                    .contains("max_completion_tokens")
            });
            let key = if use_completion_name {
                "max_completion_tokens"
            } else {
                "max_tokens"
            };
            object.insert(key.to_owned(), json!(maximum));
        }

        let selectors = Self::selectors(candidates);
        let max_prompt_price = candidates
            .iter()
            .map(|candidate| {
                candidate
                    .endpoint
                    .pricing
                    .effective(plan.prompt_tokens)
                    .prompt_per_token
            })
            .fold(0.0_f64, f64::max)
            * 1_000_000.0;
        let max_completion_price = candidates
            .iter()
            .map(|candidate| {
                candidate
                    .endpoint
                    .pricing
                    .effective(plan.prompt_tokens)
                    .completion_per_token
            })
            .fold(0.0_f64, f64::max)
            * 1_000_000.0;
        let has_local_latency = candidates
            .iter()
            .any(|candidate| candidate.expected_response_p75_seconds.is_some());
        let mut provider = serde_json::Map::new();
        provider.insert("only".to_owned(), json!(selectors));
        provider.insert("allow_fallbacks".to_owned(), json!(true));
        provider.insert("require_parameters".to_owned(), json!(true));
        provider.insert(
            "max_price".to_owned(),
            json!({
                "prompt": max_prompt_price,
                "completion": max_completion_price,
            }),
        );
        if has_local_latency {
            provider.insert("order".to_owned(), json!(Self::selectors(candidates)));
        } else {
            let sort = if request.expected_output_tokens <= 512 {
                "latency"
            } else {
                "throughput"
            };
            provider.insert("sort".to_owned(), json!(sort));
        }
        object.insert("provider".to_owned(), Value::Object(provider));

        if let Some(tier) = candidates
            .first()
            .and_then(|candidate| Self::service_tier(&candidate.endpoint.id))
        {
            object.insert("service_tier".to_owned(), json!(tier.as_str()));
        }
        if let Some(schema) = request.requirements.response_schema.as_ref() {
            object.insert("response_format".to_owned(), Self::response_format(schema));
        }
        if let Some(temperature) = request.temperature {
            object.insert("temperature".to_owned(), json!(temperature));
        }
        if request.requirements.reasoning != ReasoningEffort::None {
            object.insert(
                "reasoning".to_owned(),
                json!({"effort": request.requirements.reasoning.as_str()}),
            );
        } else if candidates.iter().all(|candidate| {
            candidate
                .endpoint
                .capabilities
                .supported_parameters
                .contains("reasoning")
        }) {
            object.insert("reasoning".to_owned(), json!({"enabled": false}));
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

    fn retry_after(response: &reqwest::Response) -> Option<Duration> {
        response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
            .map(Duration::from_secs)
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
    #[serde(alias = "cache_read")]
    input_cache_read: Option<String>,
    #[serde(alias = "cache_write")]
    input_cache_write: Option<String>,
    overrides: Option<Vec<OpenRouterPricingOverride>>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterPricingOverride {
    min_prompt_tokens: Option<u64>,
    utc_start: Option<u32>,
    utc_end: Option<u32>,
    prompt: Option<String>,
    completion: Option<String>,
    #[serde(alias = "cache_read")]
    input_cache_read: Option<String>,
    #[serde(alias = "cache_write")]
    input_cache_write: Option<String>,
}

fn parse_price(value: Option<&str>) -> f64 {
    value.and_then(|value| value.parse().ok()).unwrap_or(0.0)
}

fn parse_optional_price(value: Option<&str>) -> Option<f64> {
    value.and_then(|value| value.parse().ok())
}

fn utc_minutes_now() -> Option<u32> {
    let elapsed = SystemTime::now().duration_since(UNIX_EPOCH).ok()?;
    Some(((elapsed.as_secs() % 86_400) / 60) as u32)
}

fn hhmm_to_minutes(value: u32) -> Option<u32> {
    let hours = value / 100;
    let minutes = value % 100;
    (hours < 24 && minutes < 60).then_some(hours * 60 + minutes)
}

fn utc_window_active(start_hhmm: u32, end_hhmm: u32, now_minutes: u32) -> bool {
    let Some(start) = hhmm_to_minutes(start_hhmm) else {
        return false;
    };
    let Some(end) = hhmm_to_minutes(end_hhmm) else {
        return false;
    };
    match start.cmp(&end) {
        std::cmp::Ordering::Equal => true,
        std::cmp::Ordering::Less => (start..end).contains(&now_minutes),
        std::cmp::Ordering::Greater => now_minutes >= start || now_minutes < end,
    }
}

fn normalize_pricing(raw: OpenRouterPricing) -> Pricing {
    let mut prompt_per_token = parse_price(raw.prompt.as_deref());
    let mut completion_per_token = parse_price(raw.completion.as_deref());
    let mut cache_read_per_token = parse_optional_price(raw.input_cache_read.as_deref());
    let mut cache_write_per_token = parse_optional_price(raw.input_cache_write.as_deref());
    let mut overrides = Vec::new();
    let now_minutes = utc_minutes_now();

    for override_ in raw.overrides.unwrap_or_default() {
        if let Some(min_prompt_tokens) = override_.min_prompt_tokens {
            overrides.push(PricingOverride {
                min_prompt_tokens,
                prompt_per_token: parse_optional_price(override_.prompt.as_deref()),
                completion_per_token: parse_optional_price(override_.completion.as_deref()),
                cache_read_per_token: parse_optional_price(override_.input_cache_read.as_deref()),
                cache_write_per_token: parse_optional_price(override_.input_cache_write.as_deref()),
            });
            continue;
        }
        let active = match (override_.utc_start, override_.utc_end, now_minutes) {
            (Some(start), Some(end), Some(now)) => utc_window_active(start, end, now),
            _ => false,
        };
        if active {
            prompt_per_token =
                parse_optional_price(override_.prompt.as_deref()).unwrap_or(prompt_per_token);
            completion_per_token = parse_optional_price(override_.completion.as_deref())
                .unwrap_or(completion_per_token);
            cache_read_per_token = parse_optional_price(override_.input_cache_read.as_deref())
                .or(cache_read_per_token);
            cache_write_per_token = parse_optional_price(override_.input_cache_write.as_deref())
                .or(cache_write_per_token);
        }
    }

    Pricing {
        prompt_per_token,
        completion_per_token,
        request: parse_price(raw.request.as_deref()),
        discount: raw.discount.unwrap_or(0.0).clamp(0.0, 1.0),
        cache_read_per_token,
        cache_write_per_token,
        overrides,
    }
}

fn normalize_latency_seconds(mut latency: Percentiles) -> Percentiles {
    // OpenRouter's per-model endpoint API currently emits some latency telemetry
    // in milliseconds even though the documented normalized endpoint example is
    // in seconds. Detect the live millisecond shape from a typical percentile;
    // a >120 s p50/p75 would not be a useful interactive endpoint anyway.
    let anchor = latency.p50.or(latency.p75).or(latency.p90);
    if anchor.is_some_and(|value| value > 120.0) {
        for value in [
            &mut latency.p50,
            &mut latency.p75,
            &mut latency.p90,
            &mut latency.p95,
            &mut latency.p99,
        ] {
            if let Some(sample) = value.as_mut() {
                *sample /= 1_000.0;
            }
        }
    }
    latency
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
            pricing: normalize_pricing(self.pricing),
            stats: EndpointStats {
                latency_seconds: normalize_latency_seconds(
                    self.latency_last_30m.unwrap_or_default(),
                ),
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
        let mut cache = self.catalog_cache.lock().await;
        if let Some(cached) = cache.get(model)
            && cached.fetched_at.elapsed() <= self.catalog_ttl
        {
            return Ok(cached.endpoints.clone());
        }
        let stale = cache.get(model).map(|cached| cached.endpoints.clone());
        let url = self.model_endpoint_url(model)?;
        let fetched = async {
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
            Ok::<Vec<Endpoint>, ProviderError>(
                envelope
                    .data
                    .endpoints
                    .into_iter()
                    .map(|endpoint| endpoint.normalize(model))
                    .collect(),
            )
        }
        .await;
        match fetched {
            Ok(endpoints) => {
                cache.insert(
                    model.to_owned(),
                    CachedEndpoints {
                        fetched_at: Instant::now(),
                        endpoints: endpoints.clone(),
                    },
                );
                Ok(endpoints)
            }
            Err(error) => stale.ok_or(error),
        }
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
        let batches = Self::route_batches(plan);
        if batches.is_empty() {
            return Err(ProviderError::NoEligibleEndpoint(request.model.clone()));
        }
        let url = format!("{}/chat/completions", self.base_url);
        let mut backoff = retry.initial_backoff;
        let mut last_error = None;

        for attempt in 1..=retry.max_attempts {
            let mut retry_after = None;
            for batch in &batches {
                let body = Self::compile_body(request, plan, batch);
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
                        retry_after = Self::retry_after(&response).or(retry_after);
                        let retryable =
                            Self::retryable_status(status) || status == StatusCode::NOT_FOUND;
                        let text = response.text().await.unwrap_or_default();
                        last_error = Some(format!("HTTP {status}: {text}"));
                        if !retryable {
                            return Err(ProviderError::Request(
                                last_error.unwrap_or_else(|| "request failed".to_owned()),
                            ));
                        }
                    }
                    Err(error) => {
                        last_error = Some(error.to_string());
                    }
                }
            }
            if attempt == retry.max_attempts {
                break;
            }
            let delay = retry_after.unwrap_or(backoff).min(retry.max_backoff);
            sleep(delay).await;
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
            system_prompt: Some("system".to_owned()),
            prompt: "hello".to_owned(),
            expected_output_tokens: 42,
            prompt_tokens: Some(5),
            max_output_tokens: None,
            temperature: Some(0.0),
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
        let body = OpenRouter::compile_body(&request, &plan, &[&plan.candidates[0]]);
        assert_eq!(
            body.pointer("/provider/only/0").and_then(Value::as_str),
            Some("fast-provider/fp8")
        );
        assert_eq!(
            body.pointer("/provider/sort").and_then(Value::as_str),
            Some("latency")
        );
        assert!(body.get("max_completion_tokens").is_none());
        assert!(body.get("max_tokens").is_none());
        assert_eq!(
            body.pointer("/messages/0/role").and_then(Value::as_str),
            Some("system")
        );
        assert_eq!(
            body.pointer("/messages/1/role").and_then(Value::as_str),
            Some("user")
        );
        assert_eq!(body.get("temperature").and_then(Value::as_f64), Some(0.0));
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

    #[test]
    fn parses_shape_dependent_pricing_and_cache_fields() -> Result<(), serde_json::Error> {
        let raw = json!({
            "context_length": 1_050_000,
            "max_completion_tokens": 128_000,
            "max_prompt_tokens": 922_000,
            "model_id": "openai/gpt-6-luna",
            "name": "OpenAI: GPT-6 Luna (Flex)",
            "pricing": {
                "prompt": "0.00000005",
                "completion": "0.00000025",
                "input_cache_read": "0.000000005",
                "input_cache_write": "0.0000000625",
                "discount": 0,
                "overrides": [{
                    "min_prompt_tokens": 272_000,
                    "prompt": "0.0000001",
                    "completion": "0.000000375",
                    "input_cache_read": "0.00000001",
                    "input_cache_write": "0.000_000_125"
                }]
            },
            "provider_name": "OpenAI",
            "status": 0,
            "supported_parameters": ["reasoning", "response_format", "structured_outputs"],
            "supports_implicit_caching": false,
            "tag": "openai/flex",
            "uptime_last_30m": 99.9
        });
        let endpoint: OpenRouterEndpoint = serde_json::from_value(raw)?;
        let normalized = endpoint.normalize("openai/gpt-6-luna");
        assert_eq!(
            normalized.pricing.effective(100_000).prompt_per_token,
            0.000_000_05
        );
        assert_eq!(
            normalized.pricing.effective(300_000).prompt_per_token,
            0.000_000_1
        );
        assert_eq!(
            normalized.pricing.effective(300_000).cache_read_per_token,
            Some(0.000_000_01)
        );
        Ok(())
    }

    #[test]
    fn flex_endpoint_sets_service_tier_and_base_provider_selector() {
        let request = InferenceRequest::new("openai/gpt-6-luna", "hello", 64);
        let mut endpoint = Endpoint {
            id: "openai/flex".to_owned(),
            provider: "OpenAI".to_owned(),
            model: request.model.clone(),
            display_name: "Flex".to_owned(),
            quantization: None,
            status: 0,
            context_length: Some(1_000_000),
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            stats: EndpointStats::default(),
            capabilities: EndpointCapabilities::default(),
        };
        endpoint
            .capabilities
            .supported_parameters
            .insert("response_format".to_owned());
        let plan = RoutePlan {
            prompt_tokens: 5,
            expected_output_tokens: 64,
            cheapest_cost_usd: 0.0,
            cost_ceiling_usd: 0.0,
            candidates: vec![crate::RouteCandidate {
                endpoint,
                expected_cost_usd: 0.0,
                expected_response_p75_seconds: None,
                expected_response_p95_seconds: None,
            }],
        };
        let body = OpenRouter::compile_body(&request, &plan, &[&plan.candidates[0]]);
        assert_eq!(
            body.get("service_tier").and_then(Value::as_str),
            Some("flex")
        );
        assert_eq!(
            body.pointer("/provider/only/0").and_then(Value::as_str),
            Some("openai")
        );
    }

    #[test]
    fn local_telemetry_uses_explicit_ranked_order() {
        let request = InferenceRequest::new("vendor/model", "hello", 64);
        let mut fast = Endpoint {
            id: "fast/fp8".to_owned(),
            provider: "Fast".to_owned(),
            model: request.model.clone(),
            display_name: "Fast".to_owned(),
            quantization: Some("fp8".to_owned()),
            status: 0,
            context_length: Some(1_000_000),
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            stats: EndpointStats::default(),
            capabilities: EndpointCapabilities::default(),
        };
        fast.stats.latency_seconds.p75 = Some(0.2);
        fast.stats.throughput_tokens_per_second.p75 = Some(100.0);
        let plan = RoutePlan {
            prompt_tokens: 5,
            expected_output_tokens: 64,
            cheapest_cost_usd: 0.0,
            cost_ceiling_usd: 0.0,
            candidates: vec![crate::RouteCandidate {
                endpoint: fast,
                expected_cost_usd: 0.0,
                expected_response_p75_seconds: Some(0.84),
                expected_response_p95_seconds: None,
            }],
        };
        let body = OpenRouter::compile_body(&request, &plan, &[&plan.candidates[0]]);
        assert_eq!(
            body.pointer("/provider/order/0").and_then(Value::as_str),
            Some("fast/fp8")
        );
        assert!(body.pointer("/provider/sort").is_none());
    }

    #[test]
    fn normalizes_live_millisecond_latency_shape() {
        let normalized = normalize_latency_seconds(Percentiles {
            p50: Some(3109.5),
            p75: Some(7074.75),
            p90: Some(13656.5),
            p95: None,
            p99: Some(54984.87),
        });
        assert_eq!(normalized.p50, Some(3.1095));
        assert_eq!(normalized.p75, Some(7.07475));
        assert_eq!(normalized.p99, Some(54.98487));
    }

    #[test]
    fn keeps_documented_second_latency_shape() {
        let normalized = normalize_latency_seconds(Percentiles {
            p50: Some(0.25),
            p75: Some(0.35),
            p90: Some(0.48),
            p95: None,
            p99: Some(0.85),
        });
        assert_eq!(normalized.p50, Some(0.25));
        assert_eq!(normalized.p75, Some(0.35));
    }

    #[test]
    fn utc_pricing_windows_handle_day_and_wraparound() {
        assert!(utc_window_active(0, 1400, 13 * 60 + 59));
        assert!(!utc_window_active(0, 1400, 14 * 60));
        assert!(utc_window_active(1400, 0, 23 * 60 + 59));
        assert!(!utc_window_active(1400, 0, 13 * 60 + 59));
    }

    #[test]
    fn accepts_time_of_day_pricing_override_shape() -> Result<(), serde_json::Error> {
        let raw = json!({
            "context_length": 128_000,
            "model_id": "deepseek/deepseek-v4.1-flash",
            "name": "Alibaba: DeepSeek V4.1 Flash",
            "pricing": {
                "prompt": "0.0000003", "completion": "0.0000012",
                "input_cache_read": "0.00000003", "discount": 0,
                "overrides": [
                    {"utc_start": 0, "utc_end": 1400, "prompt": "0.0000003", "completion": "0.0000012", "input_cache_read": "0.00000003"},
                    {"utc_start": 1400, "utc_end": 0, "prompt": "0.00000015", "completion": "0.0000006", "input_cache_read": "0.000000015"}
                ]
            },
            "provider_name": "Alibaba", "status": 0,
            "supported_parameters": ["reasoning", "response_format", "structured_outputs"],
            "supports_implicit_caching": true, "tag": "alibaba", "uptime_last_30m": 99.9
        });
        let endpoint: OpenRouterEndpoint = serde_json::from_value(raw)?;
        let normalized = endpoint.normalize("deepseek/deepseek-v4.1-flash");
        assert!(normalized.pricing.prompt_per_token > 0.0);
        assert!(normalized.pricing.completion_per_token > 0.0);
        Ok(())
    }
}
