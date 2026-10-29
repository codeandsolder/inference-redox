use crate::batch::BatchStatus;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use reqwest::{Client, StatusCode, header::RETRY_AFTER};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tokio::time::{sleep, timeout};

use crate::model::STRUCTURED_TOOL_TRANSPORT_INSTRUCTION;
use crate::{
    ArtificialAnalysisBenchmarks, BatchInferenceItem, BatchJob, BenchmarkObservation,
    BenchmarkSnapshot, BenchmarkSnapshotMeta, BenchmarkSource, BenchmarkTaskType, BenchmarkUnit,
    DesignArenaBenchmark, Endpoint, EndpointCapabilities, EndpointStats, ExecutionFailurePhase,
    InferenceProvider, InferenceRequest, InferenceResponse, ModelArchitecture, ModelBenchmarks,
    ModelCatalog, ModelCatalogEntry, ModelLookup, ModelReasoning, ModelRouteClass,
    ModelTopProvider, Percentiles, Pricing, PricingOverride, ProviderError, ReasoningEffort,
    ReasoningSupport, ResponseSchema, RetryPolicy, RouteCandidate, RoutePlan,
    StrictOutputTransport, TokenUsage, ToolCall, ToolChoice, ToolTransport,
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
    model_cache: Arc<Mutex<Option<CachedModels>>>,
    benchmark_cache: Arc<Mutex<HashMap<(BenchmarkSource, BenchmarkTaskType), CachedBenchmarks>>>,
    catalog_ttl: Duration,
    benchmark_ttl: Duration,
}

#[derive(Debug, Clone)]
struct CachedEndpoints {
    fetched_at: Instant,
    endpoints: Vec<Endpoint>,
}

#[derive(Debug, Clone)]
struct CachedModels {
    fetched_at: Instant,
    catalog: ModelCatalog,
}

#[derive(Debug, Clone)]
struct CachedBenchmarks {
    fetched_at: Instant,
    snapshot: BenchmarkSnapshot,
}

#[derive(Debug)]
struct OpenRouterAttemptFailure {
    phase: ExecutionFailurePhase,
    detail: String,
    retryable: bool,
    retry_after: Option<Duration>,
    quota_observations: Vec<crate::QuotaObservation>,
}

impl OpenRouterAttemptFailure {
    fn into_provider_error(self) -> ProviderError {
        ProviderError::Execution {
            phase: self.phase,
            detail: self.detail,
            quota_observations: self.quota_observations,
        }
    }
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
            model_cache: Arc::new(Mutex::new(None)),
            benchmark_cache: Arc::new(Mutex::new(HashMap::new())),
            catalog_ttl: Duration::from_secs(60),
            benchmark_ttl: Duration::from_secs(3_600),
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

    #[must_use]
    pub fn with_benchmark_ttl(mut self, ttl: Duration) -> Self {
        self.benchmark_ttl = ttl;
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
        // OpenRouter base provider slugs may match multiple endpoint variants. The catalog tag in
        // endpoint.id is the exact selector (for example `near-ai/fp4`). Always pin that full tag.
        candidate.endpoint.id.clone()
    }

    fn only_candidate<'a>(
        plan: &'a RoutePlan,
        model: &str,
    ) -> Result<&'a RouteCandidate, ProviderError> {
        match plan.candidates.as_slice() {
            [candidate] => Ok(candidate),
            [] => Err(ProviderError::NoEligibleEndpoint(model.to_owned())),
            _ => Err(ProviderError::InvalidRequest(
                "OpenRouter execution requires exactly one pinned route; failover belongs to the generic router".to_owned(),
            )),
        }
    }

    fn response_format(schema: &ResponseSchema) -> Value {
        match schema {
            ResponseSchema::JsonObject => json!({"type": "json_object"}),
            ResponseSchema::JsonSchema {
                name,
                schema,
                strict,
            } => {
                let lowered = crate::lower_json_schema(schema);
                json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "strict": strict,
                        "schema": lowered,
                    }
                })
            }
        }
    }

    fn validate_reasoning_control(request: &InferenceRequest) -> Result<(), ProviderError> {
        if request.requirements.reasoning != ReasoningEffort::None
            && request.requirements.reasoning_max_tokens.is_some()
        {
            return Err(ProviderError::InvalidRequest(
                "OpenRouter reasoning effort and max_tokens are mutually exclusive".to_owned(),
            ));
        }
        Ok(())
    }

    fn reasoning_control(request: &InferenceRequest) -> Option<Value> {
        let explicit = request.requirements.reasoning != ReasoningEffort::None
            || request.requirements.reasoning_max_tokens.is_some()
            || request
                .requirements
                .required_parameters
                .contains("reasoning");
        if !explicit {
            return None;
        }
        let mut reasoning = serde_json::Map::new();
        if request.requirements.reasoning == ReasoningEffort::None
            && request.requirements.reasoning_max_tokens.is_none()
        {
            reasoning.insert("enabled".to_owned(), json!(false));
        } else if request.requirements.reasoning != ReasoningEffort::None {
            reasoning.insert(
                "effort".to_owned(),
                json!(request.requirements.reasoning.as_str()),
            );
        }
        if let Some(maximum) = request.requirements.reasoning_max_tokens {
            reasoning.insert("max_tokens".to_owned(), json!(maximum));
        }
        Some(Value::Object(reasoning))
    }

    fn structured_transport(
        request: &InferenceRequest,
        endpoint: &Endpoint,
    ) -> Option<StrictOutputTransport> {
        match request.requirements.response_schema.as_ref() {
            Some(ResponseSchema::JsonSchema { strict: true, .. })
                if request
                    .requirements
                    .required_parameters
                    .contains("structured_outputs") =>
            {
                Some(StrictOutputTransport::Native)
            }
            Some(ResponseSchema::JsonSchema {
                schema,
                strict: true,
                ..
            }) => endpoint
                .capabilities
                .strict_output_transport_for_schema_with_policy(
                    schema,
                    request.requirements.schema_evidence,
                ),
            _ => None,
        }
    }

    fn forced_output_tool(schema: &ResponseSchema) -> Option<Value> {
        let ResponseSchema::JsonSchema {
            name,
            schema,
            strict: true,
        } = schema
        else {
            return None;
        };
        let (parameters, _) = crate::model::forced_tool_parameter_schema(schema);
        Some(json!({
            "type": "function",
            "function": {
                "name": "__inference_redox_output",
                "description": format!("Return the structured {name} result."),
                "strict": true,
                "parameters": parameters,
            }
        }))
    }

    fn tool_choice(choice: &ToolChoice) -> Value {
        match choice {
            ToolChoice::Auto => json!("auto"),
            ToolChoice::Required => json!("required"),
            ToolChoice::None => json!("none"),
            ToolChoice::Function { name } => {
                json!({"type": "function", "function": {"name": name}})
            }
        }
    }

    fn function_tool(tool: &crate::FunctionTool) -> Value {
        let mut function = serde_json::Map::new();
        function.insert("name".to_owned(), json!(tool.name));
        if let Some(description) = tool.description.as_ref() {
            function.insert("description".to_owned(), json!(description));
        }
        function.insert("strict".to_owned(), json!(tool.strict));
        function.insert(
            "parameters".to_owned(),
            crate::lower_json_schema(&tool.parameters),
        );
        json!({"type": "function", "function": function})
    }

    fn tool_response_format(request: &InferenceRequest) -> Option<Value> {
        let (schema, _) = request.lowered_structured_tool_response_schema()?;
        Some(json!({
            "type": "json_schema",
            "json_schema": {
                "name": "__inference_redox_tool_turn",
                "strict": true,
                "schema": schema,
            }
        }))
    }

    fn insert_provider_routing(
        object: &mut serde_json::Map<String, Value>,
        plan: &RoutePlan,
        candidate: &RouteCandidate,
    ) {
        let effective_pricing = candidate.endpoint.pricing.effective(plan.prompt_tokens);
        let mut provider = serde_json::Map::new();
        provider.insert(
            "only".to_owned(),
            json!([Self::provider_selector(candidate)]),
        );
        provider.insert("allow_fallbacks".to_owned(), json!(false));
        // Endpoint eligibility comes from inference-redox conformance evidence, not OpenRouter's
        // supported_parameters declarations. With an exact endpoint pin, metadata filtering can
        // only hide under-declared capabilities or turn stale catalog data into false negatives.
        provider.insert("require_parameters".to_owned(), json!(false));
        provider.insert(
            "max_price".to_owned(),
            json!({
                "prompt": effective_pricing.prompt_per_token * 1_000_000.0,
                "completion": effective_pricing.completion_per_token * 1_000_000.0,
            }),
        );
        object.insert("provider".to_owned(), Value::Object(provider));
    }

    fn compile_body(
        request: &InferenceRequest,
        plan: &RoutePlan,
        candidate: &RouteCandidate,
    ) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("model".to_owned(), json!(request.model));
        let forced_output = Self::structured_transport(request, &candidate.endpoint)
            == Some(StrictOutputTransport::ForcedTool);
        let tool_transport = candidate.endpoint.capabilities.tool_transport(request);
        let translated_tool_response = (tool_transport == Some(ToolTransport::StructuredOutput))
            .then(|| Self::tool_response_format(request))
            .flatten();
        let user_prompt = if translated_tool_response.is_some() {
            format!(
                "{}\n\n{STRUCTURED_TOOL_TRANSPORT_INSTRUCTION}",
                request.prompt
            )
        } else {
            request.prompt.clone()
        };
        let mut messages = Vec::with_capacity(2);
        if let Some(system_prompt) = request.system_prompt.as_ref() {
            messages.push(json!({"role": "system", "content": system_prompt}));
        }
        messages.push(json!({"role": "user", "content": user_prompt}));
        object.insert("messages".to_owned(), Value::Array(messages));
        if let Some(maximum) = request.max_output_tokens {
            let use_completion_name = candidate
                .endpoint
                .capabilities
                .supports_parameter("max_completion_tokens");
            let key = if use_completion_name {
                "max_completion_tokens"
            } else {
                "max_tokens"
            };
            object.insert(key.to_owned(), json!(maximum));
        }

        Self::insert_provider_routing(&mut object, plan, candidate);

        if let Some(tier) = Self::service_tier(&candidate.endpoint.id) {
            object.insert("service_tier".to_owned(), json!(tier.as_str()));
        }
        if let Some(schema) = request.requirements.response_schema.as_ref() {
            if forced_output {
                if let Some(tool) = Self::forced_output_tool(schema) {
                    object.insert("tools".to_owned(), Value::Array(vec![tool]));
                    object.insert(
                        "tool_choice".to_owned(),
                        json!({"type": "function", "function": {"name": "__inference_redox_output"}}),
                    );
                }
            } else {
                object.insert("response_format".to_owned(), Self::response_format(schema));
            }
        } else if let Some(response_format) = translated_tool_response {
            object.insert("response_format".to_owned(), response_format);
        }
        if !forced_output
            && tool_transport == Some(ToolTransport::Native)
            && !request.tools.is_empty()
        {
            object.insert(
                "tools".to_owned(),
                Value::Array(request.tools.iter().map(Self::function_tool).collect()),
            );
        }
        if !forced_output
            && tool_transport == Some(ToolTransport::Native)
            && let Some(choice) = request.tool_choice.as_ref()
        {
            object.insert("tool_choice".to_owned(), Self::tool_choice(choice));
        }
        if !forced_output
            && tool_transport == Some(ToolTransport::Native)
            && let Some(parallel) = request.parallel_tool_calls
        {
            object.insert("parallel_tool_calls".to_owned(), json!(parallel));
        }
        if let Some(temperature) = request.temperature {
            object.insert("temperature".to_owned(), json!(temperature));
        }
        if let Some(reasoning) = Self::reasoning_control(request) {
            object.insert("reasoning".to_owned(), reasoning);
        }
        if request.include_reasoning {
            object.insert("include_reasoning".to_owned(), json!(true));
        }
        Value::Object(object)
    }

    fn batch_base_model(batch_model: &str) -> Result<&str, ProviderError> {
        batch_model.strip_suffix(":batch").ok_or_else(|| {
            ProviderError::InvalidRequest(format!(
                "OpenRouter batch submission requires a :batch model route, got {batch_model}"
            ))
        })
    }

    fn compile_batch_item_body(request: &InferenceRequest, model: &str) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("model".to_owned(), json!(model));
        let mut messages = Vec::with_capacity(2);
        if let Some(system_prompt) = request.system_prompt.as_ref() {
            messages.push(json!({"role": "system", "content": system_prompt}));
        }
        messages.push(json!({"role": "user", "content": request.prompt}));
        object.insert("messages".to_owned(), Value::Array(messages));
        if let Some(maximum) = request.max_output_tokens {
            object.insert("max_tokens".to_owned(), json!(maximum));
        }
        if let Some(schema) = request.requirements.response_schema.as_ref() {
            object.insert("response_format".to_owned(), Self::response_format(schema));
        }
        if !request.tools.is_empty() {
            object.insert(
                "tools".to_owned(),
                Value::Array(request.tools.iter().map(Self::function_tool).collect()),
            );
        }
        if let Some(choice) = request.tool_choice.as_ref() {
            object.insert("tool_choice".to_owned(), Self::tool_choice(choice));
        }
        if let Some(parallel) = request.parallel_tool_calls {
            object.insert("parallel_tool_calls".to_owned(), json!(parallel));
        }
        if let Some(temperature) = request.temperature {
            object.insert("temperature".to_owned(), json!(temperature));
        }
        if let Some(reasoning) = Self::reasoning_control(request) {
            object.insert("reasoning".to_owned(), reasoning);
        }
        if request.include_reasoning {
            object.insert("include_reasoning".to_owned(), json!(true));
        }
        Value::Object(object)
    }

    fn compile_batch_submission(
        batch_model: &str,
        items: &[BatchInferenceItem],
        provider: &str,
    ) -> Result<Value, ProviderError> {
        if items.is_empty() {
            return Err(ProviderError::InvalidRequest(
                "OpenRouter batch must contain at least one request".to_owned(),
            ));
        }
        let model = Self::batch_base_model(batch_model)?;
        let mut custom_ids = BTreeSet::new();
        let requests = items
            .iter()
            .map(|item| {
                Self::validate_reasoning_control(&item.request)?;
                if item.custom_id.trim().is_empty() {
                    return Err(ProviderError::InvalidRequest(
                        "batch custom_id must not be empty".to_owned(),
                    ));
                }
                if !custom_ids.insert(item.custom_id.as_str()) {
                    return Err(ProviderError::InvalidRequest(format!(
                        "duplicate batch custom_id: {}",
                        item.custom_id
                    )));
                }
                Ok(json!({
                    "custom_id": item.custom_id,
                    "body": Self::compile_batch_item_body(&item.request, model),
                }))
            })
            .collect::<Result<Vec<_>, ProviderError>>()?;
        let mut body = serde_json::Map::new();
        body.insert("endpoint".to_owned(), json!("/v1/chat/completions"));
        body.insert("model".to_owned(), json!(model));
        if provider.trim().is_empty() {
            return Err(ProviderError::InvalidRequest(
                "batch provider endpoint must not be empty".to_owned(),
            ));
        }
        body.insert(
            "provider".to_owned(),
            json!({"only": [provider], "allow_fallbacks": false, "require_parameters": false}),
        );
        body.insert("requests".to_owned(), Value::Array(requests));
        Ok(Value::Object(body))
    }

    fn normalize_batch(raw: Value) -> Result<BatchJob, ProviderError> {
        let id = raw
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| ProviderError::Response("batch response omitted id".to_owned()))?
            .to_owned();
        let status: BatchStatus =
            serde_json::from_value(raw.get("status").cloned().ok_or_else(|| {
                ProviderError::Response("batch response omitted status".to_owned())
            })?)
            .map_err(|error| ProviderError::Response(format!("invalid batch status: {error}")))?;
        let results = raw.get("results").and_then(Value::as_array).cloned();
        Ok(BatchJob {
            id,
            status,
            results,
            raw,
        })
    }

    /// Submit a batch pinned to one `OpenRouter` provider slug.
    ///
    /// # Errors
    /// Returns an error for invalid input or when the pinned provider has no eligible batch route.
    pub async fn submit_batch_to_provider(
        &self,
        batch_model: &str,
        provider: &str,
        items: &[BatchInferenceItem],
    ) -> Result<BatchJob, ProviderError> {
        let body = Self::compile_batch_submission(batch_model, items, provider)?;
        let response = self
            .request(reqwest::Method::POST, format!("{}/batches", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|error| ProviderError::Request(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::Request(format!("HTTP {status}: {body}")));
        }
        let raw: Value = response
            .json()
            .await
            .map_err(|error| ProviderError::Response(error.to_string()))?;
        Self::normalize_batch(raw)
    }

    /// Fetch current asynchronous batch state and inlined results when completed.
    ///
    /// # Errors
    /// Returns an error when the batch cannot be fetched or decoded.
    pub async fn batch(&self, batch_id: &str) -> Result<BatchJob, ProviderError> {
        if batch_id.trim().is_empty() {
            return Err(ProviderError::InvalidRequest(
                "batch id must not be empty".to_owned(),
            ));
        }
        let response = self
            .request(
                reqwest::Method::GET,
                format!("{}/batches/{batch_id}", self.base_url),
            )
            .send()
            .await
            .map_err(|error| ProviderError::Request(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::Request(format!("HTTP {status}: {body}")));
        }
        let raw: Value = response
            .json()
            .await
            .map_err(|error| ProviderError::Response(error.to_string()))?;
        Self::normalize_batch(raw)
    }

    /// Delete retained `OpenRouter` batch inputs/results. This is also the cleanup primitive for
    /// jobs no longer needed by a standing-task queue.
    ///
    /// # Errors
    /// Returns an error when `OpenRouter` rejects the delete request.
    pub async fn delete_batch(&self, batch_id: &str) -> Result<(), ProviderError> {
        if batch_id.trim().is_empty() {
            return Err(ProviderError::InvalidRequest(
                "batch id must not be empty".to_owned(),
            ));
        }
        let response = self
            .request(
                reqwest::Method::DELETE,
                format!("{}/batches/{batch_id}", self.base_url),
            )
            .send()
            .await
            .map_err(|error| ProviderError::Request(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::Request(format!("HTTP {status}: {body}")));
        }
        Ok(())
    }

    async fn execute_batch_once(
        &self,
        url: &str,
        request: &InferenceRequest,
        plan: &RoutePlan,
        candidate: &RouteCandidate,
    ) -> Result<InferenceResponse, OpenRouterAttemptFailure> {
        let body = Self::compile_body(request, plan, candidate);
        let response = self
            .request(reqwest::Method::POST, url.to_owned())
            .json(&body)
            .send()
            .await
            .map_err(|error| OpenRouterAttemptFailure {
                phase: ExecutionFailurePhase::TransportUnknown,
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
            let retryable = Self::retryable_status(status) || status == StatusCode::NOT_FOUND;
            let text = response.text().await.unwrap_or_default();
            return Err(OpenRouterAttemptFailure {
                phase: ExecutionFailurePhase::ResponseReceived,
                detail: format!("HTTP {status}: {text}"),
                retryable,
                retry_after,
                quota_observations,
            });
        }
        let raw: Value = response
            .json()
            .await
            .map_err(|error| OpenRouterAttemptFailure {
                phase: ExecutionFailurePhase::ResponseReceived,
                detail: error.to_string(),
                retryable: false,
                retry_after,
                quota_observations: quota_observations.clone(),
            })?;
        if let Some(error) = raw.get("error") {
            return Err(OpenRouterAttemptFailure {
                phase: ExecutionFailurePhase::ResponseReceived,
                detail: format!("OpenRouter API error: {error}"),
                retryable: Self::embedded_error_retryable(error),
                retry_after,
                quota_observations,
            });
        }
        Self::normalize_success_response(raw, request, candidate, retry_after, quota_observations)
    }

    fn normalize_success_response(
        raw: Value,
        request: &InferenceRequest,
        candidate: &RouteCandidate,
        retry_after: Option<Duration>,
        quota_observations: Vec<crate::QuotaObservation>,
    ) -> Result<InferenceResponse, OpenRouterAttemptFailure> {
        let forced_output = Self::structured_transport(request, &candidate.endpoint)
            == Some(StrictOutputTransport::ForcedTool);
        let tool_transport = candidate.endpoint.capabilities.tool_transport(request);
        let translated_tool_response = tool_transport == Some(ToolTransport::StructuredOutput);
        let Some(mut content) = Self::response_content(&raw, forced_output) else {
            return Err(OpenRouterAttemptFailure {
                phase: ExecutionFailurePhase::ResponseReceived,
                detail: if forced_output {
                    "OpenRouter constrained-output fallback omitted the forced tool call".to_owned()
                } else {
                    "OpenRouter success response omitted choices[0].message".to_owned()
                },
                retryable: false,
                retry_after,
                quota_observations,
            });
        };
        if forced_output {
            content = Self::normalize_forced_output_content(
                request,
                &content,
                retry_after,
                &quota_observations,
            )?;
        }
        if translated_tool_response {
            content = Self::normalize_structured_tool_content(
                &raw,
                content,
                retry_after,
                &quota_observations,
            )?;
        }
        let mut tool_calls = if tool_transport == Some(ToolTransport::Native) {
            Self::response_tool_calls(&raw)
        } else {
            Vec::new()
        };
        if translated_tool_response {
            let (translated_content, translated_calls) = Self::parse_structured_tool_turn(
                request,
                &content,
                retry_after,
                &quota_observations,
            )?;
            content = translated_content;
            tool_calls = translated_calls;
        }
        Ok(InferenceResponse {
            content,
            model: raw.get("model").and_then(Value::as_str).map(str::to_owned),
            provider: raw
                .get("provider")
                .and_then(Value::as_str)
                .map(str::to_owned),
            usage_cost_usd: raw.pointer("/usage/cost").and_then(Value::as_f64),
            usage: TokenUsage::from_openai_compatible(&raw),
            tool_calls,
            output_repairs: Vec::new(),
            quota_observations,
            raw,
        })
    }

    fn normalize_structured_tool_content(
        raw: &Value,
        mut content: String,
        retry_after: Option<Duration>,
        quota_observations: &[crate::QuotaObservation],
    ) -> Result<String, OpenRouterAttemptFailure> {
        let failure = |detail: String| OpenRouterAttemptFailure {
            phase: ExecutionFailurePhase::ResponseReceived,
            detail,
            retryable: false,
            retry_after,
            quota_observations: quota_observations.to_vec(),
        };
        if content.trim().is_empty()
            && let Some(value) = raw
                .pointer("/choices/0/message/content")
                .filter(|value| !value.is_null() && !value.is_string())
        {
            content = serde_json::to_string(value).map_err(|error| {
                failure(format!(
                    "OpenRouter structured-output tool fallback could not normalize JSON content: {error}"
                ))
            })?;
        }
        if !content.trim().is_empty() {
            return Ok(content);
        }
        let finish_reason = raw
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let usage = TokenUsage::from_openai_compatible(raw);
        let content_kind = match raw.pointer("/choices/0/message/content") {
            None => "missing",
            Some(Value::Null) => "null",
            Some(Value::String(_)) => "string",
            Some(Value::Array(_)) => "array",
            Some(Value::Object(_)) => "object",
            Some(Value::Bool(_)) => "boolean",
            Some(Value::Number(_)) => "number",
        };
        let unexpected_calls = Self::response_tool_calls(raw);
        let call_names = unexpected_calls
            .iter()
            .map(|call| call.name.as_str())
            .collect::<Vec<_>>()
            .join(",");
        Err(failure(format!(
            "OpenRouter structured-output tool fallback returned empty content \
             (finish_reason={finish_reason}, content_kind={content_kind}, \
             output_tokens={:?}, reasoning_tokens={:?}, unexpected_tool_calls={}, names=[{}])",
            usage.output_tokens,
            usage.reasoning_tokens,
            unexpected_calls.len(),
            call_names
        )))
    }

    fn normalize_forced_output_content(
        request: &InferenceRequest,
        content: &str,
        retry_after: Option<Duration>,
        quota_observations: &[crate::QuotaObservation],
    ) -> Result<String, OpenRouterAttemptFailure> {
        let Some(ResponseSchema::JsonSchema { schema, .. }) =
            request.requirements.response_schema.as_ref()
        else {
            return Ok(content.to_owned());
        };
        let (_, wrapped) = crate::model::forced_tool_parameter_schema(schema);
        if !wrapped {
            return Ok(content.to_owned());
        }
        let invalid = |detail: String| OpenRouterAttemptFailure {
            phase: ExecutionFailurePhase::ResponseReceived,
            detail,
            retryable: false,
            retry_after,
            quota_observations: quota_observations.to_vec(),
        };
        let arguments: Value = serde_json::from_str(content).map_err(|error| {
            invalid(format!(
                "OpenRouter forced-output wrapper returned invalid JSON: {error}"
            ))
        })?;
        let value = arguments
            .get("value")
            .ok_or_else(|| invalid("OpenRouter forced-output wrapper omitted value".to_owned()))?;
        serde_json::to_string(value).map_err(|error| {
            invalid(format!(
                "OpenRouter forced-output wrapper could not serialize value: {error}"
            ))
        })
    }

    fn parse_structured_tool_turn(
        request: &InferenceRequest,
        content: &str,
        retry_after: Option<Duration>,
        quota_observations: &[crate::QuotaObservation],
    ) -> Result<(String, Vec<ToolCall>), OpenRouterAttemptFailure> {
        let invalid = |detail: String| OpenRouterAttemptFailure {
            phase: ExecutionFailurePhase::ResponseReceived,
            detail,
            retryable: false,
            retry_after,
            quota_observations: quota_observations.to_vec(),
        };
        let value: Value = serde_json::from_str(content).map_err(|error| {
            invalid(format!(
                "OpenRouter structured-output tool fallback returned invalid JSON: {error}"
            ))
        })?;
        let text = value
            .get("text_response")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("structured tool fallback omitted text_response".to_owned()))?
            .to_owned();
        let calls = value
            .get("tool_calls")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                invalid("structured tool fallback omitted tool_calls array".to_owned())
            })?;
        if matches!(
            request.tool_choice,
            Some(ToolChoice::Required | ToolChoice::Function { .. })
        ) && calls.is_empty()
        {
            return Err(invalid(
                "structured tool fallback violated required tool choice".to_owned(),
            ));
        }
        if request.parallel_tool_calls == Some(false) && calls.len() > 1 {
            return Err(invalid(
                "structured tool fallback violated parallel_tool_calls=false".to_owned(),
            ));
        }
        let mut normalized = Vec::with_capacity(calls.len());
        for call in calls {
            let name = call
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("structured tool call omitted name".to_owned()))?;
            let tool = request
                .tools
                .iter()
                .find(|tool| tool.name == name)
                .ok_or_else(|| {
                    invalid(format!(
                        "structured tool fallback returned unknown tool {name}"
                    ))
                })?;
            if let Some(ToolChoice::Function { name: required }) = request.tool_choice.as_ref()
                && name != required
            {
                return Err(invalid(format!(
                    "structured tool fallback called {name} instead of required {required}"
                )));
            }
            let arguments = call.get("arguments").cloned().ok_or_else(|| {
                invalid(format!(
                    "structured tool call {} omitted arguments",
                    tool.name
                ))
            })?;
            normalized.push(ToolCall {
                id: None,
                name: tool.name.clone(),
                arguments,
            });
        }
        Ok((text, normalized))
    }

    fn response_content(raw: &Value, forced_output: bool) -> Option<String> {
        let message = raw.pointer("/choices/0/message")?;
        if forced_output {
            return message
                .pointer("/tool_calls/0/function")
                .filter(|function| {
                    function.get("name").and_then(Value::as_str) == Some("__inference_redox_output")
                })
                .and_then(|function| function.get("arguments"))
                .and_then(Value::as_str)
                .map(str::to_owned);
        }
        Some(
            message
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        )
    }

    fn response_tool_calls(raw: &Value) -> Vec<ToolCall> {
        raw.pointer("/choices/0/message/tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|call| {
                let function = call.get("function")?;
                let name = function.get("name")?.as_str()?.to_owned();
                let arguments = function.get("arguments").and_then(Value::as_str).map_or(
                    Value::Null,
                    |arguments| {
                        serde_json::from_str(arguments)
                            .unwrap_or_else(|_| Value::String(arguments.to_owned()))
                    },
                );
                Some(ToolCall {
                    id: call.get("id").and_then(Value::as_str).map(str::to_owned),
                    name,
                    arguments,
                })
            })
            .collect()
    }

    fn retryable_status(status: StatusCode) -> bool {
        status == StatusCode::REQUEST_TIMEOUT
            || status == StatusCode::TOO_MANY_REQUESTS
            || status.is_server_error()
    }

    fn embedded_error_retryable(error: &Value) -> bool {
        let code = error.get("code").and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str().and_then(|raw| raw.parse().ok()))
        });
        if code.is_some_and(|code| code == 408 || code == 429 || code >= 500) {
            return true;
        }

        error
            .pointer("/metadata/error_type")
            .and_then(Value::as_str)
            .is_some_and(|kind| {
                matches!(
                    kind,
                    "provider_unavailable"
                        | "rate_limit_exceeded"
                        | "server_error"
                        | "service_unavailable"
                        | "timeout"
                )
            })
    }

    fn next_backoff(current: Duration, max: Duration) -> Duration {
        current.saturating_mul(2).min(max)
    }

    /// Fetch `OpenRouter` model metadata, canonical slugs, and benchmark summaries.
    ///
    /// # Errors
    /// Returns a catalog error when metadata cannot be fetched or decoded and no stale catalog
    /// is available.
    pub async fn models(&self) -> Result<ModelCatalog, ProviderError> {
        let mut cache = self.model_cache.lock().await;
        if let Some(cached) = cache.as_ref()
            && cached.fetched_at.elapsed() <= self.catalog_ttl
        {
            return Ok(cached.catalog.clone());
        }
        let stale = cache.as_ref().map(|cached| cached.catalog.clone());
        drop(cache);

        let url = format!("{}/models", self.base_url);
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
            let envelope: ModelEnvelope = response
                .json()
                .await
                .map_err(|error| ProviderError::Catalog(error.to_string()))?;
            Ok::<ModelCatalog, ProviderError>(ModelCatalog {
                models: envelope
                    .data
                    .into_iter()
                    .filter_map(|model| {
                        let route_class = catalog_route_class(&model.id)?;
                        Some(model.normalize(route_class))
                    })
                    .collect(),
            })
        }
        .await;

        match fetched {
            Ok(catalog) => {
                cache = self.model_cache.lock().await;
                *cache = Some(CachedModels {
                    fetched_at: Instant::now(),
                    catalog: catalog.clone(),
                });
                Ok(catalog)
            }
            Err(error) => stale.ok_or(error),
        }
    }

    /// Fetch and cache one normalized `OpenRouter` benchmark source/task snapshot.
    ///
    /// The endpoint is rate-limited, so successful snapshots are cached for the benchmark TTL and
    /// stale data is retained as a fallback when refresh fails.
    ///
    /// # Errors
    /// Returns a catalog error when no cached snapshot exists and the benchmark API fails.
    pub async fn benchmarks(
        &self,
        source: BenchmarkSource,
        task_type: BenchmarkTaskType,
    ) -> Result<BenchmarkSnapshot, ProviderError> {
        let key = (source, task_type);
        let mut cache = self.benchmark_cache.lock().await;
        if let Some(cached) = cache.get(&key)
            && cached.fetched_at.elapsed() <= self.benchmark_ttl
        {
            return Ok(cached.snapshot.clone());
        }
        let stale = cache.get(&key).map(|cached| cached.snapshot.clone());
        drop(cache);

        let url = format!(
            "{}/benchmarks?source={}&task_type={}&max_results=1000",
            self.base_url,
            source.as_query_str(),
            task_type.as_query_str()
        );
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
            let value: Value = response
                .json()
                .await
                .map_err(|error| ProviderError::Catalog(error.to_string()))?;
            let shape = benchmark_response_shape(&value);
            let envelope: BenchmarkEnvelope = serde_json::from_value(value).map_err(|error| {
                ProviderError::Catalog(format!("invalid benchmark response shape {shape}: {error}"))
            })?;
            normalize_benchmark_snapshot(source, task_type, envelope)
        }
        .await;

        match fetched {
            Ok(snapshot) => {
                cache = self.benchmark_cache.lock().await;
                cache.insert(
                    key,
                    CachedBenchmarks {
                        fetched_at: Instant::now(),
                        snapshot: snapshot.clone(),
                    },
                );
                Ok(snapshot)
            }
            Err(error) => {
                if let Some(mut snapshot) = stale {
                    snapshot.meta.stale = true;
                    Ok(snapshot)
                } else {
                    Err(error)
                }
            }
        }
    }

    /// Resolve a human or provider-native model name through the `OpenRouter` catalog.
    ///
    /// # Errors
    /// Returns a catalog error when model metadata cannot be loaded.
    pub async fn resolve_model(&self, query: &str) -> Result<ModelLookup, ProviderError> {
        Ok(self.models().await?.resolve(query))
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

fn benchmark_response_shape(value: &Value) -> String {
    match value {
        Value::Object(object) => {
            let mut keys = object.keys().map(String::as_str).collect::<Vec<_>>();
            keys.sort_unstable();
            format!("object keys=[{}]", keys.join(","))
        }
        Value::Array(values) => format!("array len={}", values.len()),
        Value::Null => "null".to_owned(),
        Value::Bool(_) => "boolean".to_owned(),
        Value::Number(_) => "number".to_owned(),
        Value::String(_) => "string".to_owned(),
    }
}

#[derive(Debug, Deserialize)]
struct BenchmarkEnvelope {
    data: Vec<Value>,
    meta: OpenRouterBenchmarkMeta,
}

#[derive(Debug, Deserialize)]
struct OpenRouterBenchmarkMeta {
    as_of: Option<String>,
    citation: Option<String>,
    model_count: Option<u64>,
    source: String,
    source_url: Option<String>,
    task_type: String,
    version: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ArtificialAnalysisRow {
    model_permaslug: String,
    display_name: Option<String>,
    intelligence_index: Option<f64>,
    coding_index: Option<f64>,
    agentic_index: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct DesignArenaRow {
    model_permaslug: String,
    display_name: Option<String>,
    arena: Option<String>,
    category: Option<String>,
    elo: Option<f64>,
    win_rate: Option<f64>,
    avg_generation_time_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterBenchmarkRow {
    model_permaslug: String,
    display_name: Option<String>,
    benchmark_type: String,
    accuracy: Option<f64>,
    accuracy_stddev: Option<f64>,
    avg_cost_per_task: Option<f64>,
    last_run_timestamp: Option<String>,
    total_tasks: Option<u64>,
}

fn normalize_benchmark_snapshot(
    source: BenchmarkSource,
    task_type: BenchmarkTaskType,
    envelope: BenchmarkEnvelope,
) -> Result<BenchmarkSnapshot, ProviderError> {
    if envelope.meta.source != source.as_query_str()
        || envelope.meta.task_type != task_type.as_query_str()
    {
        return Err(ProviderError::Catalog(format!(
            "benchmark response identity mismatch: requested {}/{}, got {}/{}",
            source.as_query_str(),
            task_type.as_query_str(),
            envelope.meta.source,
            envelope.meta.task_type
        )));
    }
    let observations = match source {
        BenchmarkSource::ArtificialAnalysis => {
            normalize_artificial_analysis(task_type, envelope.meta.as_of.as_deref(), envelope.data)?
        }
        BenchmarkSource::DesignArena => {
            normalize_design_arena(task_type, envelope.meta.as_of.as_deref(), envelope.data)?
        }
        BenchmarkSource::Openrouter => normalize_openrouter_evals(task_type, envelope.data)?,
    };
    Ok(BenchmarkSnapshot {
        meta: BenchmarkSnapshotMeta {
            source,
            task_type,
            as_of: envelope.meta.as_of,
            citation: envelope.meta.citation,
            source_url: envelope.meta.source_url,
            version: envelope.meta.version,
            model_count: envelope.meta.model_count,
            stale: false,
        },
        observations,
    })
}

fn normalize_artificial_analysis(
    task_type: BenchmarkTaskType,
    as_of: Option<&str>,
    rows: Vec<Value>,
) -> Result<Vec<BenchmarkObservation>, ProviderError> {
    let mut observations = Vec::new();
    for raw in rows {
        let row: ArtificialAnalysisRow = serde_json::from_value(raw)
            .map_err(|error| ProviderError::Catalog(error.to_string()))?;
        let (metric, score) = match task_type {
            BenchmarkTaskType::Coding => ("coding_index", row.coding_index),
            BenchmarkTaskType::Intelligence => ("intelligence_index", row.intelligence_index),
            BenchmarkTaskType::Agentic => ("agentic_index", row.agentic_index),
        };
        let Some(score) = score else {
            continue;
        };
        observations.push(BenchmarkObservation {
            canonical_id: row.model_permaslug,
            source: BenchmarkSource::ArtificialAnalysis,
            task_type,
            metric: metric.to_owned(),
            score,
            unit: BenchmarkUnit::Index,
            configuration: row.display_name,
            qualifiers: BTreeMap::new(),
            sample_count: None,
            standard_deviation: None,
            average_cost_usd: None,
            average_generation_time_ms: None,
            observed_at: as_of.map(str::to_owned),
        });
    }
    Ok(observations)
}

fn normalize_design_arena(
    task_type: BenchmarkTaskType,
    as_of: Option<&str>,
    rows: Vec<Value>,
) -> Result<Vec<BenchmarkObservation>, ProviderError> {
    let mut observations = Vec::new();
    for raw in rows {
        let row: DesignArenaRow = serde_json::from_value(raw)
            .map_err(|error| ProviderError::Catalog(error.to_string()))?;
        let mut qualifiers = BTreeMap::new();
        if let Some(arena) = row.arena {
            qualifiers.insert("arena".to_owned(), arena);
        }
        if let Some(category) = row.category {
            qualifiers.insert("category".to_owned(), category);
        }
        for (metric, score, unit) in [
            ("design_arena_elo", row.elo, BenchmarkUnit::Elo),
            (
                "design_arena_win_rate",
                row.win_rate,
                BenchmarkUnit::Percent,
            ),
        ] {
            let Some(score) = score else {
                continue;
            };
            observations.push(BenchmarkObservation {
                canonical_id: row.model_permaslug.clone(),
                source: BenchmarkSource::DesignArena,
                task_type,
                metric: metric.to_owned(),
                score,
                unit,
                configuration: row.display_name.clone(),
                qualifiers: qualifiers.clone(),
                sample_count: None,
                standard_deviation: None,
                average_cost_usd: None,
                average_generation_time_ms: row.avg_generation_time_ms,
                observed_at: as_of.map(str::to_owned),
            });
        }
    }
    Ok(observations)
}

fn normalize_openrouter_evals(
    task_type: BenchmarkTaskType,
    rows: Vec<Value>,
) -> Result<Vec<BenchmarkObservation>, ProviderError> {
    let mut observations = Vec::new();
    for raw in rows {
        let row: OpenRouterBenchmarkRow = serde_json::from_value(raw)
            .map_err(|error| ProviderError::Catalog(error.to_string()))?;
        let Some(score) = row.accuracy else {
            continue;
        };
        observations.push(BenchmarkObservation {
            canonical_id: row.model_permaslug,
            source: BenchmarkSource::Openrouter,
            task_type,
            metric: row.benchmark_type,
            score,
            unit: BenchmarkUnit::Fraction,
            configuration: row.display_name,
            qualifiers: BTreeMap::new(),
            sample_count: row.total_tasks,
            standard_deviation: row.accuracy_stddev,
            average_cost_usd: row.avg_cost_per_task,
            average_generation_time_ms: None,
            observed_at: row.last_run_timestamp,
        });
    }
    Ok(observations)
}

#[derive(Debug, Deserialize)]
struct ModelEnvelope {
    data: Vec<OpenRouterModel>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterModel {
    id: String,
    canonical_slug: Option<String>,
    hugging_face_id: Option<String>,
    name: String,
    description: Option<String>,
    created: Option<u64>,
    context_length: Option<u64>,
    architecture: Option<OpenRouterArchitecture>,
    pricing: OpenRouterPricing,
    top_provider: Option<OpenRouterTopProvider>,
    supported_parameters: Option<Vec<String>>,
    knowledge_cutoff: Option<String>,
    expiration_date: Option<String>,
    benchmarks: Option<OpenRouterBenchmarks>,
    reasoning: Option<OpenRouterModelReasoning>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterArchitecture {
    modality: Option<String>,
    input_modalities: Option<Vec<String>>,
    output_modalities: Option<Vec<String>>,
    tokenizer: Option<String>,
    instruct_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterTopProvider {
    context_length: Option<u64>,
    max_completion_tokens: Option<u64>,
    is_moderated: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenRouterBenchmarks {
    #[serde(default)]
    design_arena: Vec<OpenRouterDesignArenaBenchmark>,
    artificial_analysis: Option<OpenRouterArtificialAnalysisBenchmarks>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterArtificialAnalysisBenchmarks {
    #[serde(rename = "intelligence_index")]
    intelligence: Option<f64>,
    #[serde(rename = "coding_index")]
    coding: Option<f64>,
    #[serde(rename = "agentic_index")]
    agentic: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterDesignArenaBenchmark {
    arena: Option<String>,
    category: Option<String>,
    elo: Option<f64>,
    win_rate: Option<f64>,
    rank: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterModelReasoning {
    mandatory: Option<bool>,
    default_enabled: Option<bool>,
    supported_efforts: Option<Vec<String>>,
    default_effort: Option<String>,
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

fn catalog_route_class(model_id: &str) -> Option<ModelRouteClass> {
    match model_id.rsplit_once(':') {
        None => Some(ModelRouteClass::Standard),
        Some((_, "free")) => Some(ModelRouteClass::Free),
        Some((_, "batch")) => Some(ModelRouteClass::Batch),
        Some(_) => None,
    }
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

impl OpenRouterModel {
    fn normalize(self, route_class: ModelRouteClass) -> ModelCatalogEntry {
        let canonical_id = self
            .canonical_slug
            .clone()
            .unwrap_or_else(|| self.id.clone());
        let mut aliases =
            BTreeSet::from([self.id.clone(), canonical_id.clone(), self.name.clone()]);
        if let Some(hugging_face_id) = self.hugging_face_id.as_ref() {
            aliases.insert(hugging_face_id.clone());
        }
        let architecture = self
            .architecture
            .map_or_else(ModelArchitecture::default, |raw| ModelArchitecture {
                modality: raw.modality,
                input_modalities: raw
                    .input_modalities
                    .unwrap_or_default()
                    .into_iter()
                    .collect(),
                output_modalities: raw
                    .output_modalities
                    .unwrap_or_default()
                    .into_iter()
                    .collect(),
                tokenizer: raw.tokenizer,
                instruct_type: raw.instruct_type,
            });
        let top_provider = self
            .top_provider
            .map_or_else(ModelTopProvider::default, |raw| ModelTopProvider {
                context_length: raw.context_length,
                max_completion_tokens: raw.max_completion_tokens,
                is_moderated: raw.is_moderated,
            });
        let reasoning = self
            .reasoning
            .map_or_else(ModelReasoning::default, |raw| ModelReasoning {
                mandatory: raw.mandatory,
                default_enabled: raw.default_enabled,
                supported_efforts: raw
                    .supported_efforts
                    .unwrap_or_default()
                    .into_iter()
                    .collect(),
                default_effort: raw.default_effort,
            });
        let benchmarks = self
            .benchmarks
            .map_or_else(ModelBenchmarks::default, |raw| ModelBenchmarks {
                artificial_analysis: raw.artificial_analysis.map(|benchmark| {
                    ArtificialAnalysisBenchmarks {
                        intelligence_index: benchmark.intelligence,
                        coding_index: benchmark.coding,
                        agentic_index: benchmark.agentic,
                    }
                }),
                design_arena: raw
                    .design_arena
                    .into_iter()
                    .map(|benchmark| DesignArenaBenchmark {
                        arena: benchmark.arena,
                        category: benchmark.category,
                        elo: benchmark.elo,
                        win_rate: benchmark.win_rate,
                        rank: benchmark.rank,
                    })
                    .collect(),
            });
        ModelCatalogEntry {
            id: self.id,
            route_class,
            canonical_id,
            name: self.name,
            description: self.description,
            aliases,
            created_unix: self.created,
            context_length: self.context_length,
            architecture,
            pricing: normalize_pricing(self.pricing),
            pricing_known: true,
            top_provider,
            supported_parameters: self
                .supported_parameters
                .unwrap_or_default()
                .into_iter()
                .collect(),
            reasoning,
            benchmarks,
            knowledge_cutoff: self.knowledge_cutoff,
            expiration_date: self.expiration_date,
            source: "openrouter".to_owned(),
        }
    }
}

fn openrouter_workload_policy(model_id: &str) -> crate::WorkloadPolicy {
    let route_class = catalog_route_class(model_id);
    let base = model_id.strip_suffix(":free").unwrap_or(model_id);
    if route_class == Some(ModelRouteClass::Free) && base.starts_with("thinkingmachines/inkling") {
        crate::WorkloadPolicy {
            agentic: crate::AgenticPolicy::Required,
            requires_stable_session: false,
        }
    } else {
        crate::WorkloadPolicy::default()
    }
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
        let id = self.tag;
        let provider_key = id.split('/').next().unwrap_or(&id).to_owned();
        Endpoint {
            source: "openrouter".to_owned(),
            credential: None,
            provider_key,
            id,
            provider: self.provider_name.unwrap_or_else(|| "unknown".to_owned()),
            model: self.model_id.unwrap_or_else(|| requested_model.to_owned()),
            route_class: catalog_route_class(requested_model).unwrap_or(ModelRouteClass::Standard),
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
            workload_policy: openrouter_workload_policy(requested_model),
            capabilities: EndpointCapabilities {
                adapter_parameters: [
                    "max_tokens",
                    "max_completion_tokens",
                    "response_format",
                    "structured_outputs",
                    "temperature",
                    "reasoning",
                    "reasoning_effort",
                    "reasoning_max_tokens",
                    "include_reasoning",
                    "system_role",
                    "tools",
                    "tool_choice",
                    "parallel_tool_calls",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
                supported_parameters: parameters,
                verified_parameters: BTreeSet::new(),
                rejected_parameters: BTreeSet::new(),
                verified_reasoning_efforts: BTreeSet::new(),
                rejected_reasoning_efforts: BTreeSet::new(),
                constraint_enforcement: BTreeMap::new(),
                verified_schema_features: BTreeMap::new(),
                rejected_schema_features: BTreeMap::new(),
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
        Self::validate_reasoning_control(request)?;
        if retry.max_attempts == 0 {
            return Err(ProviderError::InvalidRequest(
                "retry.max_attempts must be at least 1".to_owned(),
            ));
        }
        if retry.attempt_timeout.is_zero() {
            return Err(ProviderError::InvalidRequest(
                "retry.attempt_timeout must be greater than zero".to_owned(),
            ));
        }
        let candidate = Self::only_candidate(plan, &request.model)?;
        let url = format!("{}/chat/completions", self.base_url);
        let mut backoff = retry.initial_backoff;
        let mut last_error = None;
        let mut last_phase = None;
        let mut last_quota_observations = Vec::new();

        for attempt in 1..=retry.max_attempts {
            let result = timeout(
                retry.attempt_timeout,
                self.execute_batch_once(&url, request, plan, candidate),
            )
            .await
            .unwrap_or_else(|_| {
                Err(OpenRouterAttemptFailure {
                    phase: ExecutionFailurePhase::TransportUnknown,
                    detail: format!(
                        "request attempt timed out after {} ms",
                        retry.attempt_timeout.as_millis()
                    ),
                    retryable: true,
                    retry_after: None,
                    quota_observations: Vec::new(),
                })
            });
            match result {
                Ok(response) => return Ok(response),
                Err(failure) if !failure.retryable => {
                    return Err(failure.into_provider_error());
                }
                Err(failure) => {
                    let retry_after = failure.retry_after;
                    if last_phase.is_none()
                        || failure.phase == ExecutionFailurePhase::ResponseReceived
                    {
                        last_phase = Some(failure.phase);
                    }
                    if !failure.quota_observations.is_empty() {
                        last_quota_observations = failure.quota_observations;
                    }
                    last_error = Some(failure.detail);
                    if attempt < retry.max_attempts {
                        let delay = retry_after.unwrap_or(backoff).min(retry.max_backoff);
                        sleep(delay).await;
                        backoff = Self::next_backoff(backoff, retry.max_backoff);
                    }
                }
            }
        }

        Err(ProviderError::Execution {
            phase: last_phase.unwrap_or(ExecutionFailurePhase::TransportUnknown),
            detail: last_error.unwrap_or_else(|| "request failed without an error body".to_owned()),
            quota_observations: last_quota_observations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema_only_endpoint(
        model: &str,
        features: impl IntoIterator<Item = crate::SchemaFeature>,
    ) -> Endpoint {
        let mut capabilities = EndpointCapabilities::default();
        capabilities
            .adapter_parameters
            .insert("structured_outputs".to_owned());
        capabilities
            .verified_parameters
            .insert("structured_outputs".to_owned());
        capabilities.constraint_enforcement.insert(
            "structured_outputs".to_owned(),
            crate::ConstraintEnforcement::Enforced,
        );
        capabilities.verified_schema_features.insert(
            "structured_outputs".to_owned(),
            features.into_iter().collect(),
        );
        Endpoint {
            source: "openrouter".to_owned(),
            credential: None,
            provider_key: "schema-only".to_owned(),
            id: "schema-only/fp8".to_owned(),
            provider: "Schema Only".to_owned(),
            model: model.to_owned(),
            route_class: ModelRouteClass::Standard,
            display_name: "Schema Only".to_owned(),
            quantization: None,
            status: 0,
            context_length: Some(1_000),
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            stats: EndpointStats::default(),
            workload_policy: crate::WorkloadPolicy::default(),
            capabilities,
        }
    }

    #[test]
    fn execution_rejects_multi_candidate_gateway_plans() {
        let request = InferenceRequest::new("vendor/model", "hello", 8);
        let endpoint = |id: &str| Endpoint {
            source: "openrouter".to_owned(),
            credential: None,
            provider_key: id.split('/').next().unwrap_or(id).to_owned(),
            id: id.to_owned(),
            provider: id.to_owned(),
            model: request.model.clone(),
            route_class: ModelRouteClass::Standard,
            display_name: id.to_owned(),
            quantization: None,
            status: 0,
            context_length: Some(1_000),
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            stats: EndpointStats::default(),
            workload_policy: crate::WorkloadPolicy::default(),
            capabilities: EndpointCapabilities::default(),
        };
        let plan = RoutePlan {
            prompt_tokens: 1,
            expected_output_tokens: 8,
            cheapest_cost_usd: 0.0,
            cost_ceiling_usd: 0.0,
            candidates: ["first/fp8", "second/fp8"]
                .into_iter()
                .map(|id| RouteCandidate {
                    endpoint: endpoint(id),
                    expected_cost_usd: 0.0,
                    expected_response_p75_seconds: None,
                    expected_response_p95_seconds: None,
                })
                .collect(),
        };
        assert!(matches!(
            OpenRouter::only_candidate(&plan, &request.model),
            Err(ProviderError::InvalidRequest(_))
        ));
    }

    #[test]
    fn reasoning_only_success_can_have_empty_visible_content() {
        let raw = json!({
            "choices": [{
                "message": {"content": null, "reasoning": "still thinking"},
                "finish_reason": "length"
            }]
        });
        assert_eq!(
            OpenRouter::response_content(&raw, false).as_deref(),
            Some("")
        );
        assert!(OpenRouter::response_content(&json!({"choices": [{}]}), false).is_none());
    }

    #[test]
    fn classifies_embedded_openrouter_errors() {
        assert!(OpenRouter::embedded_error_retryable(&json!({
            "code": 502,
            "message": "Flex processing is temporarily unavailable",
            "metadata": {"error_type": "provider_unavailable"}
        })));
        assert!(OpenRouter::embedded_error_retryable(&json!({
            "code": "429",
            "message": "rate limited"
        })));
        assert!(!OpenRouter::embedded_error_retryable(&json!({
            "code": 400,
            "message": "bad request"
        })));
    }

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
            include_reasoning: false,
            tools: Vec::new(),
            tool_choice: None,
            parallel_tool_calls: None,
            agentic: false,
            session_id: None,
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
            source: "openrouter".to_owned(),
            credential: None,
            provider_key: "fast-provider".to_owned(),
            id: "fast-provider/fp8".to_owned(),
            provider: "Fast Provider".to_owned(),
            model: request.model.clone(),
            route_class: ModelRouteClass::Standard,
            display_name: "Fast".to_owned(),
            quantization: Some("fp8".to_owned()),
            status: 0,
            context_length: Some(1000),
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            stats: EndpointStats::default(),
            workload_policy: crate::WorkloadPolicy::default(),
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
        let body = OpenRouter::compile_body(&request, &plan, &plan.candidates[0]);
        assert_eq!(
            body.pointer("/provider/only/0").and_then(Value::as_str),
            Some("fast-provider/fp8")
        );
        assert!(body.pointer("/provider/sort").is_none());
        assert!(body.pointer("/provider/order").is_none());
        assert_eq!(
            body.pointer("/provider/allow_fallbacks")
                .and_then(Value::as_bool),
            Some(false)
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
        assert_eq!(
            body.pointer("/provider/require_parameters")
                .and_then(Value::as_bool),
            Some(false)
        );
    }

    #[test]
    fn compiles_verified_strict_output_as_forced_tool_when_native_schema_is_missing() {
        let mut request = InferenceRequest::new("openai/example", "hello", 16);
        request.requirements.response_schema = Some(ResponseSchema::JsonSchema {
            name: "answer".to_owned(),
            schema: json!({
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": false
            }),
            strict: true,
        });
        let mut endpoint = Endpoint {
            source: "openrouter".to_owned(),
            credential: None,
            provider_key: "tools-only".to_owned(),
            id: "tools-only/int4".to_owned(),
            provider: "Tools Only".to_owned(),
            model: request.model.clone(),
            route_class: ModelRouteClass::Standard,
            display_name: "Tools Only".to_owned(),
            quantization: None,
            status: 0,
            context_length: Some(1000),
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            stats: EndpointStats::default(),
            workload_policy: crate::WorkloadPolicy::default(),
            capabilities: EndpointCapabilities::default(),
        };
        endpoint
            .capabilities
            .adapter_parameters
            .extend(["tools".to_owned(), "tool_choice".to_owned()]);
        endpoint
            .capabilities
            .supported_parameters
            .extend(["tools".to_owned(), "tool_choice".to_owned()]);
        endpoint
            .capabilities
            .constraint_enforcement
            .insert("tools".to_owned(), crate::ConstraintEnforcement::Enforced);
        let plan = RoutePlan {
            prompt_tokens: 5,
            expected_output_tokens: 16,
            cheapest_cost_usd: 0.0,
            cost_ceiling_usd: 0.0,
            candidates: vec![RouteCandidate {
                endpoint,
                expected_cost_usd: 0.0,
                expected_response_p75_seconds: None,
                expected_response_p95_seconds: None,
            }],
        };
        let body = OpenRouter::compile_body(&request, &plan, &plan.candidates[0]);
        assert!(body.get("response_format").is_none());
        assert_eq!(
            body.pointer("/tools/0/function/name")
                .and_then(Value::as_str),
            Some("__inference_redox_output")
        );
        assert_eq!(
            body.pointer("/tools/0/function/strict")
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            body.pointer("/tool_choice/function/name")
                .and_then(Value::as_str),
            Some("__inference_redox_output")
        );
        assert_eq!(
            body.pointer("/provider/require_parameters")
                .and_then(Value::as_bool),
            Some(false)
        );
    }

    #[test]
    fn forced_output_response_extracts_only_expected_tool_arguments() {
        let raw = json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "function": {
                            "name": "__inference_redox_output",
                            "arguments": r#"{"value":7}"#
                        }
                    }]
                }
            }]
        });
        assert_eq!(
            OpenRouter::response_content(&raw, true).as_deref(),
            Some(r#"{"value":7}"#)
        );
        let wrong = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{"function": {"name": "other", "arguments": "{}"}}]
                }
            }]
        });
        assert!(OpenRouter::response_content(&wrong, true).is_none());
    }

    #[test]
    fn forced_tool_fallback_wraps_and_unwraps_non_object_root_schema() -> Result<(), String> {
        let schema = ResponseSchema::JsonSchema {
            name: "array_answer".to_owned(),
            schema: json!({
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
                "maxItems": 1
            }),
            strict: true,
        };
        let tool = OpenRouter::forced_output_tool(&schema)
            .ok_or_else(|| "forced output tool should compile".to_owned())?;
        assert_eq!(
            tool.pointer("/function/parameters/type")
                .and_then(Value::as_str),
            Some("object")
        );
        assert_eq!(
            tool.pointer("/function/parameters/properties/value/type")
                .and_then(Value::as_str),
            Some("array")
        );

        let mut request = InferenceRequest::new("openai/example", "return one integer", 8);
        request.requirements.response_schema = Some(schema);
        let normalized =
            OpenRouter::normalize_forced_output_content(&request, r#"{"value":[7]}"#, None, &[])
                .map_err(|failure| failure.detail)?;
        assert_eq!(normalized, "[7]");
        Ok(())
    }

    #[test]
    fn compiles_explicit_function_tools_and_exact_choice() {
        let mut request = InferenceRequest::new("openai/example", "hello", 16);
        request.tools.push(crate::FunctionTool {
            name: "lookup".to_owned(),
            description: Some("Look up a value".to_owned()),
            parameters: json!({"type": "object", "properties": {}, "additionalProperties": false}),
            strict: true,
        });
        request.tool_choice = Some(ToolChoice::Function {
            name: "lookup".to_owned(),
        });
        let mut endpoint = Endpoint {
            source: "openrouter".to_owned(),
            credential: None,
            provider_key: "provider".to_owned(),
            id: "provider/fp8".to_owned(),
            provider: "Provider".to_owned(),
            model: request.model.clone(),
            route_class: ModelRouteClass::Standard,
            display_name: "Provider".to_owned(),
            quantization: None,
            status: 0,
            context_length: Some(1000),
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            stats: EndpointStats::default(),
            workload_policy: crate::WorkloadPolicy::default(),
            capabilities: EndpointCapabilities::default(),
        };
        endpoint
            .capabilities
            .adapter_parameters
            .extend(["tools".to_owned(), "tool_choice".to_owned()]);
        endpoint
            .capabilities
            .verified_parameters
            .extend(["tools".to_owned(), "tool_choice".to_owned()]);
        endpoint
            .capabilities
            .constraint_enforcement
            .insert("tools".to_owned(), crate::ConstraintEnforcement::Enforced);
        let plan = RoutePlan {
            prompt_tokens: 5,
            expected_output_tokens: 16,
            cheapest_cost_usd: 0.0,
            cost_ceiling_usd: 0.0,
            candidates: vec![RouteCandidate {
                endpoint,
                expected_cost_usd: 0.0,
                expected_response_p75_seconds: None,
                expected_response_p95_seconds: None,
            }],
        };
        let body = OpenRouter::compile_body(&request, &plan, &plan.candidates[0]);
        assert_eq!(
            body.pointer("/tools/0/function/name")
                .and_then(Value::as_str),
            Some("lookup")
        );
        assert_eq!(
            body.pointer("/tools/0/function/strict")
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            body.pointer("/tool_choice/function/name")
                .and_then(Value::as_str),
            Some("lookup")
        );
        assert_eq!(
            body.pointer("/messages/0/content").and_then(Value::as_str),
            Some("hello")
        );
    }

    #[test]
    fn compiles_exact_tool_as_structured_output_when_native_tools_are_missing() {
        let mut request = InferenceRequest::new("openai/example", "call lookup", 16);
        request.tools.push(crate::FunctionTool {
            name: "lookup".to_owned(),
            description: Some("Return a lookup value".to_owned()),
            parameters: json!({
                "type":"object",
                "properties":{"value":{"type":"integer"}},
                "required":["value"],
                "additionalProperties":false
            }),
            strict: true,
        });
        request.tool_choice = Some(ToolChoice::Function {
            name: "lookup".to_owned(),
        });
        let mut endpoint = Endpoint {
            source: "openrouter".to_owned(),
            credential: None,
            provider_key: "schema-only".to_owned(),
            id: "schema-only/fp8".to_owned(),
            provider: "Schema Only".to_owned(),
            model: request.model.clone(),
            route_class: ModelRouteClass::Standard,
            display_name: "Schema Only".to_owned(),
            quantization: None,
            status: 0,
            context_length: Some(1000),
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            stats: EndpointStats::default(),
            workload_policy: crate::WorkloadPolicy::default(),
            capabilities: EndpointCapabilities::default(),
        };
        endpoint
            .capabilities
            .adapter_parameters
            .insert("structured_outputs".to_owned());
        endpoint
            .capabilities
            .verified_parameters
            .insert("structured_outputs".to_owned());
        endpoint.capabilities.constraint_enforcement.insert(
            "structured_outputs".to_owned(),
            crate::ConstraintEnforcement::Enforced,
        );
        endpoint.capabilities.verified_schema_features.insert(
            "structured_outputs".to_owned(),
            BTreeSet::from([
                crate::SchemaFeature::ArrayType,
                crate::SchemaFeature::ArrayItems,
                crate::SchemaFeature::ArrayBounds,
                crate::SchemaFeature::NestedObject,
                crate::SchemaFeature::ToolTurnEnvelope,
            ]),
        );
        let plan = RoutePlan {
            prompt_tokens: 5,
            expected_output_tokens: 16,
            cheapest_cost_usd: 0.0,
            cost_ceiling_usd: 0.0,
            candidates: vec![RouteCandidate {
                endpoint,
                expected_cost_usd: 0.0,
                expected_response_p75_seconds: None,
                expected_response_p95_seconds: None,
            }],
        };
        let body = OpenRouter::compile_body(&request, &plan, &plan.candidates[0]);
        assert_eq!(
            body.pointer("/response_format/type")
                .and_then(Value::as_str),
            Some("json_schema")
        );
        assert_eq!(
            body.pointer("/response_format/json_schema/schema/properties/tool_calls/items/properties/arguments/properties/value/type")
                .and_then(Value::as_str),
            Some("integer")
        );
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
        assert!(
            body.pointer("/messages/0/content")
                .and_then(Value::as_str)
                .is_some_and(|content| content.contains(STRUCTURED_TOOL_TRANSPORT_INSTRUCTION))
        );
    }

    #[test]
    fn compiles_auto_multi_tool_turn_as_structured_envelope() {
        let mut request = InferenceRequest::new("openai/example", "maybe use tools", 32);
        for name in ["lookup", "search"] {
            request.tools.push(crate::FunctionTool {
                name: name.to_owned(),
                description: None,
                parameters: json!({
                    "type":"object",
                    "properties":{"value":{"type":"integer"}},
                    "required":["value"],
                    "additionalProperties":false
                }),
                strict: true,
            });
        }
        request.tool_choice = Some(ToolChoice::Auto);
        request.parallel_tool_calls = Some(false);
        let endpoint = schema_only_endpoint(
            &request.model,
            [
                crate::SchemaFeature::OneOf,
                crate::SchemaFeature::NestedObject,
                crate::SchemaFeature::ArrayType,
                crate::SchemaFeature::ArrayItems,
                crate::SchemaFeature::ArrayBounds,
                crate::SchemaFeature::ToolTurnEnvelope,
            ],
        );
        let plan = RoutePlan {
            prompt_tokens: 5,
            expected_output_tokens: 32,
            cheapest_cost_usd: 0.0,
            cost_ceiling_usd: 0.0,
            candidates: vec![RouteCandidate {
                endpoint,
                expected_cost_usd: 0.0,
                expected_response_p75_seconds: None,
                expected_response_p95_seconds: None,
            }],
        };
        let body = OpenRouter::compile_body(&request, &plan, &plan.candidates[0]);
        let schema = body.pointer("/response_format/json_schema/schema");
        assert_eq!(
            schema
                .and_then(|value| value.pointer("/properties/tool_calls/maxItems"))
                .and_then(Value::as_u64),
            Some(1)
        );
        assert!(
            schema
                .and_then(|value| value.pointer("/properties/tool_calls/minItems"))
                .is_none()
        );
        assert_eq!(
            schema
                .and_then(|value| value.pointer("/properties/tool_calls/items/oneOf"))
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(2)
        );
    }

    #[test]
    fn normalizes_structured_json_object_content_before_tool_turn_parsing() -> Result<(), String> {
        let mut request = InferenceRequest::new("openai/example", "use lookup", 16);
        request.tools.push(crate::FunctionTool {
            name: "lookup".to_owned(),
            description: None,
            parameters: json!({
                "type":"object",
                "properties":{"value":{"type":"integer"}},
                "required":["value"],
                "additionalProperties":false
            }),
            strict: true,
        });
        request.tool_choice = Some(ToolChoice::Auto);
        let mut endpoint = schema_only_endpoint(
            &request.model,
            [
                crate::SchemaFeature::ArrayType,
                crate::SchemaFeature::ArrayItems,
                crate::SchemaFeature::NestedObject,
                crate::SchemaFeature::ToolTurnEnvelope,
            ],
        );
        endpoint.stats.uptime_30m = Some(100.0);
        let candidate = RouteCandidate {
            endpoint,
            expected_cost_usd: 0.0,
            expected_response_p75_seconds: None,
            expected_response_p95_seconds: None,
        };
        let raw = json!({
            "model":"openai/example",
            "provider":"schema-only",
            "choices":[{
                "finish_reason":"stop",
                "message":{
                    "content":{
                        "text_response":"using lookup",
                        "tool_calls":[{"name":"lookup","arguments":{"value":7}}]
                    }
                }
            }]
        });
        let response =
            OpenRouter::normalize_success_response(raw, &request, &candidate, None, vec![])
                .map_err(|failure| failure.detail)?;
        assert_eq!(response.content, "using lookup");
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].name, "lookup");
        assert_eq!(response.tool_calls[0].arguments, json!({"value":7}));
        Ok(())
    }

    #[test]
    fn normalizes_structured_auto_turn_with_text_and_multiple_calls() -> Result<(), String> {
        let mut request = InferenceRequest::new("openai/example", "maybe use tools", 32);
        request.tools.extend([
            crate::FunctionTool {
                name: "lookup".to_owned(),
                description: None,
                parameters: json!({"type":"object"}),
                strict: true,
            },
            crate::FunctionTool {
                name: "search".to_owned(),
                description: None,
                parameters: json!({"type":"object"}),
                strict: true,
            },
        ]);
        request.tool_choice = Some(ToolChoice::Auto);
        request.parallel_tool_calls = Some(true);
        let content = r#"{"text_response":"I will use both.","tool_calls":[{"name":"lookup","arguments":{"value":7}},{"name":"search","arguments":{"query":"rust"}}]}"#;
        let (text, calls) = OpenRouter::parse_structured_tool_turn(&request, content, None, &[])
            .map_err(|failure| failure.detail)?;
        assert_eq!(text, "I will use both.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "lookup");
        assert_eq!(calls[0].arguments, json!({"value":7}));
        assert_eq!(calls[1].name, "search");
        assert_eq!(calls[1].arguments, json!({"query":"rust"}));
        Ok(())
    }

    #[test]
    fn compiles_explicit_reasoning_off_budget_and_trace_controls() {
        let mut request = InferenceRequest::new("vendor/model", "puzzle", 32);
        request.include_reasoning = true;
        request
            .requirements
            .required_parameters
            .insert("reasoning".to_owned());
        let endpoint = Endpoint {
            source: "openrouter".to_owned(),
            credential: None,
            provider_key: "provider".to_owned(),
            id: "provider/fp8".to_owned(),
            provider: "Provider".to_owned(),
            model: request.model.clone(),
            route_class: ModelRouteClass::Free,
            display_name: "Provider".to_owned(),
            quantization: None,
            status: 0,
            context_length: Some(1000),
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            stats: EndpointStats::default(),
            workload_policy: crate::WorkloadPolicy::default(),
            capabilities: EndpointCapabilities::default(),
        };
        let plan = RoutePlan {
            prompt_tokens: 5,
            expected_output_tokens: 32,
            cheapest_cost_usd: 0.0,
            cost_ceiling_usd: 0.0,
            candidates: vec![crate::RouteCandidate {
                endpoint,
                expected_cost_usd: 0.0,
                expected_response_p75_seconds: None,
                expected_response_p95_seconds: None,
            }],
        };

        let off = OpenRouter::compile_body(&request, &plan, &plan.candidates[0]);
        assert_eq!(
            off.pointer("/reasoning/enabled").and_then(Value::as_bool),
            Some(false)
        );
        assert_eq!(
            off.get("include_reasoning").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            off.pointer("/provider/require_parameters")
                .and_then(Value::as_bool),
            Some(false)
        );

        request.requirements.reasoning = ReasoningEffort::None;
        request.requirements.reasoning_max_tokens = Some(256);
        let bounded = OpenRouter::compile_body(&request, &plan, &plan.candidates[0]);
        assert!(bounded.pointer("/reasoning/effort").is_none());
        assert_eq!(
            bounded
                .pointer("/reasoning/max_tokens")
                .and_then(Value::as_u64),
            Some(256)
        );
    }

    #[test]
    fn compiles_documented_batch_submission_shape() -> Result<(), ProviderError> {
        let mut request = InferenceRequest::new("ignored/by/batch", "summarize", 64);
        request.system_prompt = Some("datasheet parser".to_owned());
        request.max_output_tokens = Some(128);
        request.requirements.response_schema = Some(ResponseSchema::JsonObject);
        let body = OpenRouter::compile_batch_submission(
            "openai/gpt-6-luna:batch",
            &[BatchInferenceItem {
                custom_id: "sheet-1".to_owned(),
                request,
            }],
            "azure/global",
        )?;
        assert_eq!(
            body.get("endpoint").and_then(Value::as_str),
            Some("/v1/chat/completions")
        );
        assert_eq!(
            body.get("model").and_then(Value::as_str),
            Some("openai/gpt-6-luna")
        );
        assert_eq!(
            body.pointer("/requests/0/body/model")
                .and_then(Value::as_str),
            Some("openai/gpt-6-luna")
        );
        assert_eq!(
            body.pointer("/requests/0/body/response_format/type")
                .and_then(Value::as_str),
            Some("json_object")
        );
        assert_eq!(
            body.pointer("/provider/only/0").and_then(Value::as_str),
            Some("azure/global")
        );
        assert_eq!(
            body.pointer("/provider/allow_fallbacks")
                .and_then(Value::as_bool),
            Some(false)
        );
        let pinned = OpenRouter::compile_batch_submission(
            "openai/gpt-6-luna:batch",
            &[BatchInferenceItem {
                custom_id: "sheet-2".to_owned(),
                request: InferenceRequest::new("ignored", "summarize", 64),
            }],
            "azure",
        )?;
        assert_eq!(
            pinned.pointer("/provider/only/0").and_then(Value::as_str),
            Some("azure")
        );
        assert_eq!(
            pinned
                .pointer("/provider/allow_fallbacks")
                .and_then(Value::as_bool),
            Some(false)
        );
        Ok(())
    }

    #[test]
    fn batch_submission_requires_unique_ids_and_batch_route() {
        let request = InferenceRequest::new("x", "x", 1);
        let duplicate = vec![
            BatchInferenceItem {
                custom_id: "same".to_owned(),
                request: request.clone(),
            },
            BatchInferenceItem {
                custom_id: "same".to_owned(),
                request,
            },
        ];
        assert!(matches!(
            OpenRouter::compile_batch_submission("openai/gpt-6-luna:batch", &duplicate, "azure"),
            Err(ProviderError::InvalidRequest(_))
        ));
        assert!(matches!(
            OpenRouter::compile_batch_submission("openai/gpt-6-luna", &duplicate[..1], "azure"),
            Err(ProviderError::InvalidRequest(_))
        ));
    }

    #[test]
    fn normalizes_terminal_batch_state() -> Result<(), ProviderError> {
        let batch = OpenRouter::normalize_batch(json!({
            "id": "batch-1",
            "status": "completed",
            "results": [{"custom_id": "sheet-1", "response": {"status_code": 200}}]
        }))?;
        assert!(batch.is_terminal());
        assert!(batch.is_success());
        assert_eq!(batch.results.as_ref().map(Vec::len), Some(1));
        Ok(())
    }

    #[test]
    fn free_inkling_routes_require_agentic_harnesses() {
        for model in [
            "thinkingmachines/inkling:free",
            "thinkingmachines/inkling-20260715:free",
            "thinkingmachines/inkling-small:free",
        ] {
            assert_eq!(
                openrouter_workload_policy(model).agentic,
                crate::AgenticPolicy::Required
            );
        }
        assert_eq!(
            openrouter_workload_policy("thinkingmachines/inkling").agentic,
            crate::AgenticPolicy::Any
        );
        assert_eq!(
            openrouter_workload_policy("nvidia/nemotron-3-ultra-550b-a55b:free").agentic,
            crate::AgenticPolicy::Any
        );
    }

    #[test]
    fn model_catalog_classifies_standard_free_and_batch_routes() {
        assert_eq!(
            catalog_route_class("nvidia/nemotron-3-ultra-550b-a55b"),
            Some(ModelRouteClass::Standard)
        );
        assert_eq!(
            catalog_route_class("nvidia/nemotron-3-ultra-550b-a55b:free"),
            Some(ModelRouteClass::Free)
        );
        assert_eq!(
            catalog_route_class("openai/gpt-6-luna:batch"),
            Some(ModelRouteClass::Batch)
        );
        assert_eq!(catalog_route_class("openai/gpt-6-luna:floor"), None);
        assert_eq!(catalog_route_class("openai/gpt-6-luna:nitro"), None);
    }

    #[test]
    fn normalizes_model_identity_reasoning_and_benchmarks() -> Result<(), serde_json::Error> {
        let raw = json!({
            "id": "openai/gpt-6-luna:batch",
            "canonical_slug": "openai/gpt-6-luna-20260922",
            "hugging_face_id": null,
            "name": "OpenAI: GPT-6 Luna (batch)",
            "description": "Fast model",
            "created": 1_790_100_786,
            "context_length": 1_050_000,
            "architecture": {
                "modality": "text+image->text",
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
                "tokenizer": "GPT",
                "instruct_type": null
            },
            "pricing": {"prompt": "0.00000005", "completion": "0.00000025"},
            "top_provider": {
                "context_length": 1_050_000,
                "max_completion_tokens": 128_000,
                "is_moderated": true
            },
            "supported_parameters": ["reasoning", "tools"],
            "knowledge_cutoff": "2025-08",
            "expiration_date": null,
            "benchmarks": {
                "design_arena": [],
                "artificial_analysis": {
                    "intelligence_index": 38.1,
                    "coding_index": 41.2,
                    "agentic_index": null
                }
            },
            "reasoning": {
                "mandatory": false,
                "default_enabled": true,
                "supported_efforts": ["high", "medium", "low", "none"],
                "default_effort": "medium"
            }
        });
        let model: OpenRouterModel = serde_json::from_value(raw)?;
        let normalized = model.normalize(ModelRouteClass::Batch);
        assert_eq!(normalized.canonical_id, "openai/gpt-6-luna-20260922");
        assert_eq!(normalized.route_class, ModelRouteClass::Batch);
        assert!(normalized.aliases.contains("openai/gpt-6-luna:batch"));
        assert_eq!(normalized.top_provider.max_completion_tokens, Some(128_000));
        assert!(normalized.reasoning.supported_efforts.contains("none"));
        assert_eq!(
            normalized
                .benchmarks
                .artificial_analysis
                .as_ref()
                .and_then(|benchmark| benchmark.intelligence_index),
            Some(38.1)
        );
        Ok(())
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
        assert!(
            normalized
                .capabilities
                .supported_parameters
                .contains("structured_outputs")
        );
        assert!(normalized.capabilities.verified_parameters.is_empty());
        assert!(normalized.capabilities.rejected_parameters.is_empty());
        Ok(())
    }

    #[test]
    fn flex_endpoint_sets_service_tier_and_base_provider_selector() {
        let request = InferenceRequest::new("openai/gpt-6-luna", "hello", 64);
        let mut endpoint = Endpoint {
            source: "openrouter".to_owned(),
            credential: None,
            provider_key: "openai".to_owned(),
            id: "openai/flex".to_owned(),
            provider: "OpenAI".to_owned(),
            model: request.model.clone(),
            route_class: ModelRouteClass::Standard,
            display_name: "Flex".to_owned(),
            quantization: None,
            status: 0,
            context_length: Some(1_000_000),
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            stats: EndpointStats::default(),
            workload_policy: crate::WorkloadPolicy::default(),
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
        let body = OpenRouter::compile_body(&request, &plan, &plan.candidates[0]);
        assert_eq!(
            body.get("service_tier").and_then(Value::as_str),
            Some("flex")
        );
        assert_eq!(
            body.pointer("/provider/only/0").and_then(Value::as_str),
            Some("openai/flex")
        );
    }

    #[test]
    fn local_telemetry_never_delegates_ordering_to_openrouter() {
        let request = InferenceRequest::new("vendor/model", "hello", 64);
        let mut fast = Endpoint {
            source: "openrouter".to_owned(),
            credential: None,
            provider_key: "fast".to_owned(),
            id: "fast/fp8".to_owned(),
            provider: "Fast".to_owned(),
            model: request.model.clone(),
            route_class: ModelRouteClass::Standard,
            display_name: "Fast".to_owned(),
            quantization: Some("fp8".to_owned()),
            status: 0,
            context_length: Some(1_000_000),
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            stats: EndpointStats::default(),
            workload_policy: crate::WorkloadPolicy::default(),
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
        let body = OpenRouter::compile_body(&request, &plan, &plan.candidates[0]);
        assert_eq!(
            body.pointer("/provider/only/0").and_then(Value::as_str),
            Some("fast/fp8")
        );
        assert!(body.pointer("/provider/order").is_none());
        assert!(body.pointer("/provider/sort").is_none());
        assert_eq!(
            body.pointer("/provider/allow_fallbacks")
                .and_then(Value::as_bool),
            Some(false)
        );
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

    #[test]
    fn benchmark_response_shape_does_not_echo_error_body() {
        let value = json!({
            "error": {
                "message": "upstream detail that should not be copied into diagnostics",
                "request_id": "opaque"
            }
        });
        assert_eq!(benchmark_response_shape(&value), "object keys=[error]");
    }

    #[test]
    fn normalizes_artificial_analysis_benchmark_payload() -> Result<(), Box<dyn std::error::Error>>
    {
        let envelope: BenchmarkEnvelope = serde_json::from_value(json!({
            "data": [{
                "agentic_index": 57.9,
                "coding_index": 81.6,
                "display_name": "Claude Fable 5.1 (Max, Default Fallback)",
                "intelligence_index": 53.4,
                "model_permaslug": "anthropic/claude-fable-5.1-20260831",
                "source": "artificial-analysis"
            }],
            "meta": {
                "as_of": "2026-10-22T12:01:07.781Z",
                "citation": "AA via OpenRouter",
                "model_count": 1,
                "source": "artificial-analysis",
                "source_url": "https://artificialanalysis.ai",
                "task_type": "coding",
                "version": "v1"
            }
        }))?;
        let snapshot = normalize_benchmark_snapshot(
            BenchmarkSource::ArtificialAnalysis,
            BenchmarkTaskType::Coding,
            envelope,
        )?;
        assert_eq!(snapshot.observations.len(), 1);
        assert_eq!(snapshot.observations[0].metric, "coding_index");
        assert_eq!(snapshot.observations[0].score, 81.6);
        assert_eq!(snapshot.observations[0].unit, BenchmarkUnit::Index);
        Ok(())
    }

    #[test]
    fn normalizes_design_arena_configuration_without_flattening()
    -> Result<(), Box<dyn std::error::Error>> {
        let envelope: BenchmarkEnvelope = serde_json::from_value(json!({
            "data": [{
                "arena": "models",
                "avg_generation_time_ms": 288_179,
                "category": "codecategories",
                "display_name": "Claude Opus 5.5",
                "elo": 1389,
                "model_permaslug": "anthropic/claude-opus-5.5-20260921",
                "source": "design-arena",
                "win_rate": 63.8
            }],
            "meta": {
                "as_of": "2026-10-22T12:01:11.227Z",
                "citation": "Design Arena via OpenRouter",
                "model_count": 1,
                "source": "design-arena",
                "source_url": "https://www.designarena.ai",
                "task_type": "coding",
                "version": "v1"
            }
        }))?;
        let snapshot = normalize_benchmark_snapshot(
            BenchmarkSource::DesignArena,
            BenchmarkTaskType::Coding,
            envelope,
        )?;
        assert_eq!(snapshot.observations.len(), 2);
        assert!(snapshot.observations.iter().all(|observation| {
            observation.configuration.as_deref() == Some("Claude Opus 5.5")
        }));
        assert!(snapshot.observations.iter().any(|observation| {
            observation.metric == "design_arena_elo"
                && (observation.score - 1389.0).abs() < f64::EPSILON
        }));
        Ok(())
    }

    #[test]
    fn normalizes_openrouter_eval_provenance() -> Result<(), Box<dyn std::error::Error>> {
        let envelope: BenchmarkEnvelope = serde_json::from_value(json!({
            "data": [{
                "accuracy": 0.801_818_181_818_181_8,
                "accuracy_stddev": 0.026_222_075_532_643_652,
                "avg_cost_per_task": 0.918_943_667_272_727_2,
                "benchmark_type": "tau_bench_verified_airline",
                "display_name": "Anthropic: Claude Fable 5",
                "last_run_timestamp": "2026-10-17T10:30:43.110Z",
                "model_permaslug": "anthropic/claude-5-fable-20260609",
                "source": "openrouter",
                "total_tasks": 550
            }],
            "meta": {
                "as_of": "2026-10-21T18:18:28.156Z",
                "citation": "OpenRouter evals",
                "model_count": 1,
                "source": "openrouter",
                "source_url": "https://openrouter.ai/rankings",
                "task_type": "agentic",
                "version": "v1"
            }
        }))?;
        let snapshot = normalize_benchmark_snapshot(
            BenchmarkSource::Openrouter,
            BenchmarkTaskType::Agentic,
            envelope,
        )?;
        let observation = &snapshot.observations[0];
        assert_eq!(observation.metric, "tau_bench_verified_airline");
        assert_eq!(observation.unit, BenchmarkUnit::Fraction);
        assert_eq!(observation.sample_count, Some(550));
        assert_eq!(observation.average_cost_usd, Some(0.918_943_667_272_727_2));
        Ok(())
    }
}
