use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::{
    Client,
    header::{HeaderMap, HeaderValue, USER_AGENT},
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio::sync::Mutex;

use super::openai_compatible::{
    JsonResponse, base_chat_body, execute_json_with_retry_headers, response_format,
};
use crate::{
    AgenticPolicy, Endpoint, EndpointCapabilities, EndpointStats, ExecutionFailurePhase,
    InferenceProvider, InferenceRequest, InferenceResponse, ModelCatalog, ModelRouteClass,
    NativeModelRecord, Pricing, PricingOverride, ProviderError, ProviderInventory, ReasoningEffort,
    ReasoningSupport, ResponseSchema, RetryPolicy, RoutePlan, TokenUsage, WorkloadPolicy,
};

const DEFAULT_INFERENCE_BASE: &str = "https://opencode.ai/inference";
const DEFAULT_GO_INFERENCE_BASE: &str = "https://opencode.ai/inference/go";
const USER_AGENT_VALUE: &str = concat!("inference-redox/", env!("CARGO_PKG_VERSION"));
const DEFAULT_MODELS_DEV_URL: &str = "https://models.dev/api.json";
const PER_MILLION: f64 = 1_000_000.0;

/// `OpenCode` Console pay-as-you-go adapter. Free-tier rows are intentionally excluded because
/// `OpenCode` documents them as available only from the `OpenCode` client, not arbitrary API clients.
#[derive(Debug, Clone)]
pub struct OpenCodeConsole {
    client: Client,
    api_key: String,
    inference_base: String,
    models_dev_url: String,
    model_cache: Arc<Mutex<Option<CachedModels>>>,
    catalog_ttl: Duration,
    surface: OpenCodeSurface,
}

#[derive(Debug, Clone)]
struct CachedModels {
    fetched_at: Instant,
    models: Vec<OpenCodeModel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenCodeSurface {
    PayAsYouGo,
    Go,
}

impl OpenCodeSurface {
    fn source(self) -> &'static str {
        match self {
            Self::PayAsYouGo => "opencode",
            Self::Go => "opencode-go",
        }
    }

    fn display_name(self) -> &'static str {
        match self {
            Self::PayAsYouGo => "OpenCode Console",
            Self::Go => "OpenCode Go",
        }
    }

    fn route_class(self) -> ModelRouteClass {
        match self {
            Self::PayAsYouGo => ModelRouteClass::Standard,
            Self::Go => ModelRouteClass::Subscription,
        }
    }

    fn accepts_cost(self, cost: &Cost) -> bool {
        self == Self::Go || cost.is_paid()
    }

    fn workload_policy(self) -> WorkloadPolicy {
        match self {
            Self::PayAsYouGo => WorkloadPolicy::default(),
            Self::Go => WorkloadPolicy {
                agentic: AgenticPolicy::Required,
                requires_stable_session: true,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dialect {
    OpenAiChat,
    OpenAiResponses,
    AnthropicMessages,
    Gemini,
}

impl Dialect {
    fn endpoint_name(self) -> &'static str {
        match self {
            Self::OpenAiChat => "chat-completions",
            Self::OpenAiResponses => "responses",
            Self::AnthropicMessages => "messages",
            Self::Gemini => "gemini",
        }
    }
}

#[derive(Debug, Clone)]
struct OpenCodeModel {
    id: String,
    name: String,
    canonical_hint: Option<String>,
    context_length: Option<u64>,
    max_prompt_tokens: Option<u64>,
    max_completion_tokens: Option<u64>,
    pricing: Pricing,
    dialect: Dialect,
    temperature: bool,
    structured_output: bool,
    reasoning_efforts: BTreeSet<ReasoningEffort>,
    api_base: String,
}

impl OpenCodeModel {
    fn native_record(&self, route_class: ModelRouteClass) -> NativeModelRecord {
        NativeModelRecord {
            id: self.id.clone(),
            display_name: Some(self.name.clone()),
            owned_by: Some("opencode".to_owned()),
            created_unix: None,
            context_length: self.context_length,
            max_completion_tokens: self.max_completion_tokens,
            claimed_parameters: self.claimed_parameters(),
            pricing: Some(self.pricing.clone()),
            route_class: Some(route_class),
            canonical_hint: self.canonical_hint.clone(),
            virtual_route: false,
        }
    }

    fn adapter_parameters(&self) -> BTreeSet<String> {
        let mut parameters = ["system_role", "max_tokens"]
            .into_iter()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        parameters.insert(self.dialect.endpoint_name().to_owned());
        match self.dialect {
            Dialect::OpenAiChat | Dialect::OpenAiResponses => {
                parameters.insert("response_format".to_owned());
                parameters.insert("structured_outputs".to_owned());
            }
            Dialect::Gemini => {
                parameters.insert("response_format".to_owned());
            }
            Dialect::AnthropicMessages => {}
        }
        if self.temperature {
            parameters.insert("temperature".to_owned());
        }
        if matches!(self.dialect, Dialect::OpenAiResponses) && !self.reasoning_efforts.is_empty() {
            parameters.insert("reasoning".to_owned());
            parameters.insert("reasoning_effort".to_owned());
        }
        parameters
    }

    fn claimed_parameters(&self) -> BTreeSet<String> {
        let mut parameters = ["system_role", "max_tokens"]
            .into_iter()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        parameters.insert(self.dialect.endpoint_name().to_owned());
        if self.temperature {
            parameters.insert("temperature".to_owned());
        }
        if self.structured_output {
            parameters.insert("response_format".to_owned());
            if matches!(self.dialect, Dialect::OpenAiChat | Dialect::OpenAiResponses) {
                parameters.insert("structured_outputs".to_owned());
            }
        }
        if matches!(self.dialect, Dialect::OpenAiResponses) && !self.reasoning_efforts.is_empty() {
            parameters.insert("reasoning".to_owned());
            parameters.insert("reasoning_effort".to_owned());
        }
        parameters
    }

    fn reasoning_support(&self) -> ReasoningSupport {
        if matches!(self.dialect, Dialect::OpenAiResponses) && !self.reasoning_efforts.is_empty() {
            ReasoningSupport::Exact(self.reasoning_efforts.clone())
        } else {
            ReasoningSupport::None
        }
    }
}

impl OpenCodeConsole {
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_surface(api_key.into(), OpenCodeSurface::PayAsYouGo)
    }

    #[must_use]
    pub fn new_go(api_key: impl Into<String>) -> Self {
        Self::with_surface(api_key.into(), OpenCodeSurface::Go)
    }

    fn with_surface(api_key: String, surface: OpenCodeSurface) -> Self {
        Self {
            client: Client::new(),
            api_key,
            inference_base: match surface {
                OpenCodeSurface::PayAsYouGo => DEFAULT_INFERENCE_BASE,
                OpenCodeSurface::Go => DEFAULT_GO_INFERENCE_BASE,
            }
            .to_owned(),
            models_dev_url: DEFAULT_MODELS_DEV_URL.to_owned(),
            model_cache: Arc::new(Mutex::new(None)),
            catalog_ttl: Duration::from_secs(60),
            surface,
        }
    }

    #[must_use]
    pub fn with_inference_base(mut self, base: impl Into<String>) -> Self {
        base.into()
            .trim_end_matches('/')
            .clone_into(&mut self.inference_base);
        self
    }

    #[must_use]
    pub fn with_models_dev_url(mut self, url: impl Into<String>) -> Self {
        self.models_dev_url = url.into();
        self
    }

    #[must_use]
    pub fn with_catalog_ttl(mut self, ttl: Duration) -> Self {
        self.catalog_ttl = ttl;
        self
    }

    /// Fetch the account-visible `OpenCode` model list and intersect it with models.dev metadata.
    /// Pay-as-you-go excludes client-only free promo rows; Go retains every priced subscription row.
    ///
    /// # Errors
    /// Returns a catalog error when either metadata source cannot be fetched or decoded.
    pub async fn inventory(
        &self,
        catalog: &ModelCatalog,
    ) -> Result<ProviderInventory, ProviderError> {
        let route_class = self.surface.route_class();
        Ok(ProviderInventory::reconcile(
            self.surface.source(),
            self.models()
                .await?
                .iter()
                .map(|model| model.native_record(route_class))
                .collect(),
            catalog,
        ))
    }

    async fn models(&self) -> Result<Vec<OpenCodeModel>, ProviderError> {
        let cache = self.model_cache.lock().await;
        if let Some(cached) = cache.as_ref()
            && cached.fetched_at.elapsed() <= self.catalog_ttl
        {
            return Ok(cached.models.clone());
        }
        let stale = cache.as_ref().map(|cached| cached.models.clone());
        drop(cache);

        let fetched = self.fetch_models().await;
        match fetched {
            Ok(models) => {
                *self.model_cache.lock().await = Some(CachedModels {
                    fetched_at: Instant::now(),
                    models: models.clone(),
                });
                Ok(models)
            }
            Err(error) => stale.ok_or(error),
        }
    }

    async fn fetch_models(&self) -> Result<Vec<OpenCodeModel>, ProviderError> {
        let live_request = self
            .client
            .get(format!("{}/v1/models", self.inference_base))
            .bearer_auth(&self.api_key)
            .send();
        let metadata_request = self.client.get(&self.models_dev_url).send();
        let (live_response, metadata_response) =
            tokio::try_join!(live_request, metadata_request)
                .map_err(|error| ProviderError::Catalog(error.to_string()))?;
        if !live_response.status().is_success() {
            return Err(ProviderError::Catalog(format!(
                "OpenCode model list HTTP {}",
                live_response.status()
            )));
        }
        if !metadata_response.status().is_success() {
            return Err(ProviderError::Catalog(format!(
                "models.dev HTTP {}",
                metadata_response.status()
            )));
        }
        let live: LiveModelEnvelope = live_response
            .json()
            .await
            .map_err(|error| ProviderError::Catalog(error.to_string()))?;
        let metadata: ModelsDevRoot = metadata_response
            .json()
            .await
            .map_err(|error| ProviderError::Catalog(error.to_string()))?;
        let live_ids = live
            .data
            .into_iter()
            .map(|model| model.id)
            .collect::<BTreeSet<_>>();
        let metadata_models = match self.surface {
            OpenCodeSurface::PayAsYouGo => metadata.opencode.models,
            OpenCodeSurface::Go => metadata.opencode_go.models,
        };
        let mut models = metadata_models
            .into_iter()
            .filter(|(id, model)| {
                live_ids.contains(id)
                    && model
                        .cost
                        .as_ref()
                        .is_some_and(|cost| self.surface.accepts_cost(cost))
            })
            .filter_map(|(id, model)| model.normalize(id))
            .collect::<Vec<_>>();
        models.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(models)
    }

    fn compile_body(
        model: &OpenCodeModel,
        request: &InferenceRequest,
    ) -> Result<Value, ProviderError> {
        if request.temperature.is_some() && !model.temperature {
            return Err(ProviderError::InvalidRequest(format!(
                "OpenCode model {} does not advertise temperature control",
                request.model
            )));
        }
        if request.requirements.reasoning != ReasoningEffort::None
            && !model
                .reasoning_support()
                .supports(request.requirements.reasoning)
        {
            return Err(ProviderError::InvalidRequest(format!(
                "OpenCode model {} does not support normalized reasoning effort {}",
                request.model,
                request.requirements.reasoning.as_str()
            )));
        }
        if request.requirements.response_schema.is_some()
            && (!model.structured_output || matches!(model.dialect, Dialect::AnthropicMessages))
        {
            return Err(ProviderError::InvalidRequest(format!(
                "OpenCode model {} has no normalized structured-output dialect",
                request.model
            )));
        }
        match model.dialect {
            Dialect::OpenAiChat => Ok(Self::compile_chat(request)),
            Dialect::OpenAiResponses => Ok(Self::compile_responses(request)),
            Dialect::AnthropicMessages => Ok(Self::compile_anthropic(request)),
            Dialect::Gemini => Self::compile_gemini(request),
        }
    }

    fn compile_chat(request: &InferenceRequest) -> Value {
        let mut body = base_chat_body(request, "max_tokens");
        if let Some(schema) = request.requirements.response_schema.as_ref() {
            body.insert("response_format".to_owned(), response_format(schema));
        }
        Value::Object(body)
    }

    fn compile_responses(request: &InferenceRequest) -> Value {
        let mut body = Map::new();
        body.insert("model".to_owned(), json!(request.model));
        body.insert("input".to_owned(), json!(request.prompt));
        if let Some(system) = request.system_prompt.as_ref() {
            body.insert("instructions".to_owned(), json!(system));
        }
        if let Some(maximum) = request.max_output_tokens {
            body.insert("max_output_tokens".to_owned(), json!(maximum));
        }
        if let Some(temperature) = request.temperature {
            body.insert("temperature".to_owned(), json!(temperature));
        }
        if let Some(schema) = request.requirements.response_schema.as_ref() {
            let format = match schema {
                ResponseSchema::JsonObject => json!({"type": "json_object"}),
                ResponseSchema::JsonSchema {
                    name,
                    schema,
                    strict,
                } => json!({
                    "type": "json_schema",
                    "name": name,
                    "schema": schema,
                    "strict": strict,
                }),
            };
            body.insert("text".to_owned(), json!({"format": format}));
        }
        if request.requirements.reasoning != ReasoningEffort::None {
            body.insert(
                "reasoning".to_owned(),
                json!({"effort": request.requirements.reasoning.as_str()}),
            );
        }
        Value::Object(body)
    }

    fn compile_anthropic(request: &InferenceRequest) -> Value {
        let maximum = request
            .max_output_tokens
            .unwrap_or(request.expected_output_tokens.max(1));
        let mut body = Map::new();
        body.insert("model".to_owned(), json!(request.model));
        body.insert("max_tokens".to_owned(), json!(maximum));
        body.insert(
            "messages".to_owned(),
            json!([{"role": "user", "content": request.prompt}]),
        );
        if let Some(system) = request.system_prompt.as_ref() {
            body.insert("system".to_owned(), json!(system));
        }
        if let Some(temperature) = request.temperature {
            body.insert("temperature".to_owned(), json!(temperature));
        }
        Value::Object(body)
    }

    fn compile_gemini(request: &InferenceRequest) -> Result<Value, ProviderError> {
        let mut body = Map::new();
        body.insert(
            "contents".to_owned(),
            json!([{"parts": [{"text": request.prompt}]}]),
        );
        if let Some(system) = request.system_prompt.as_ref() {
            body.insert(
                "systemInstruction".to_owned(),
                json!({"parts": [{"text": system}]}),
            );
        }
        let mut generation = Map::new();
        if let Some(maximum) = request.max_output_tokens {
            generation.insert("maxOutputTokens".to_owned(), json!(maximum));
        }
        if let Some(temperature) = request.temperature {
            generation.insert("temperature".to_owned(), json!(temperature));
        }
        if let Some(schema) = request.requirements.response_schema.as_ref() {
            generation.insert("responseMimeType".to_owned(), json!("application/json"));
            if let ResponseSchema::JsonSchema { schema, strict, .. } = schema {
                if *strict {
                    return Err(ProviderError::InvalidRequest(
                        "OpenCode Gemini strict JSON schema semantics are not normalized"
                            .to_owned(),
                    ));
                }
                generation.insert("responseJsonSchema".to_owned(), schema.clone());
            }
        }
        if !generation.is_empty() {
            body.insert("generationConfig".to_owned(), Value::Object(generation));
        }
        Ok(Value::Object(body))
    }

    fn url(model: &OpenCodeModel) -> String {
        match model.dialect {
            Dialect::OpenAiChat => format!("{}/chat/completions", model.api_base),
            Dialect::OpenAiResponses => format!("{}/responses", model.api_base),
            Dialect::AnthropicMessages => format!("{}/messages", model.api_base),
            Dialect::Gemini => format!("{}/models/{}:generateContent", model.api_base, model.id),
        }
    }

    fn request_headers(&self, request: &InferenceRequest) -> Result<HeaderMap, ProviderError> {
        if self.surface == OpenCodeSurface::Go && !request.agentic {
            return Err(ProviderError::InvalidRequest(
                "OpenCode Go requires an agentic coding workload".to_owned(),
            ));
        }
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_static(USER_AGENT_VALUE));
        if let Some(session_id) = request.session_id.as_deref() {
            if session_id.trim().is_empty() {
                return Err(ProviderError::InvalidRequest(
                    "OpenCode session_id must not be empty".to_owned(),
                ));
            }
            let value = HeaderValue::from_str(session_id).map_err(|_| {
                ProviderError::InvalidRequest(
                    "OpenCode session_id contains invalid HTTP header characters".to_owned(),
                )
            })?;
            headers.insert("x-opencode-session", value);
        } else if self.surface == OpenCodeSurface::Go {
            return Err(ProviderError::InvalidRequest(
                "OpenCode Go requires a stable session_id".to_owned(),
            ));
        }
        Ok(headers)
    }

    fn normalize_response(
        surface: OpenCodeSurface,
        model: &OpenCodeModel,
        response: JsonResponse,
    ) -> Result<InferenceResponse, ProviderError> {
        let JsonResponse {
            raw,
            quota_observations,
        } = response;
        let content = match model.dialect {
            Dialect::OpenAiChat => raw
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str)
                .map(str::to_owned),
            Dialect::OpenAiResponses => responses_text(&raw),
            Dialect::AnthropicMessages => anthropic_text(&raw),
            Dialect::Gemini => gemini_text(&raw),
        }
        .filter(|content| !content.is_empty())
        .ok_or_else(|| ProviderError::Execution {
            phase: ExecutionFailurePhase::ResponseReceived,
            detail: format!(
                "OpenCode {} response had no non-empty text output",
                model.dialect.endpoint_name()
            ),
            quota_observations: quota_observations.clone(),
        })?;
        let usage = match model.dialect {
            Dialect::OpenAiChat | Dialect::OpenAiResponses => {
                TokenUsage::from_openai_compatible(&raw)
            }
            Dialect::AnthropicMessages => anthropic_usage(&raw),
            Dialect::Gemini => gemini_usage(&raw),
        };
        let usage_cost_usd = match model.dialect {
            Dialect::Gemini => raw.pointer("/usageMetadata/cost").and_then(Value::as_f64),
            _ => raw.pointer("/usage/cost").and_then(Value::as_f64),
        };
        let response_model = match model.dialect {
            Dialect::Gemini => raw
                .get("modelVersion")
                .and_then(Value::as_str)
                .map(str::to_owned),
            _ => raw.get("model").and_then(Value::as_str).map(str::to_owned),
        };
        Ok(InferenceResponse {
            content,
            model: response_model.or_else(|| Some(model.id.clone())),
            provider: Some(surface.source().to_owned()),
            usage_cost_usd,
            usage,
            quota_observations,
            raw,
        })
    }
}

#[async_trait]
impl InferenceProvider for OpenCodeConsole {
    async fn endpoints(&self, model: &str) -> Result<Vec<Endpoint>, ProviderError> {
        let models = self.models().await?;
        let Some(model_metadata) = models.iter().find(|candidate| candidate.id == model) else {
            return Err(ProviderError::NoEligibleEndpoint(model.to_owned()));
        };
        Ok(vec![Endpoint {
            source: self.surface.source().to_owned(),
            credential: None,
            provider_key: self.surface.source().to_owned(),
            id: format!(
                "{}/{}",
                self.surface.source(),
                model_metadata.dialect.endpoint_name()
            ),
            provider: self.surface.display_name().to_owned(),
            model: model.to_owned(),
            route_class: self.surface.route_class(),
            display_name: model_metadata.name.clone(),
            quantization: None,
            status: 0,
            context_length: model_metadata.context_length,
            max_prompt_tokens: model_metadata.max_prompt_tokens,
            max_completion_tokens: model_metadata.max_completion_tokens,
            pricing: model_metadata.pricing.clone(),
            stats: EndpointStats::default(),
            workload_policy: self.surface.workload_policy(),
            capabilities: EndpointCapabilities {
                adapter_parameters: model_metadata.adapter_parameters(),
                supported_parameters: model_metadata.claimed_parameters(),
                verified_parameters: BTreeSet::new(),
                rejected_parameters: BTreeSet::new(),
                verified_reasoning_efforts: BTreeSet::new(),
                rejected_reasoning_efforts: BTreeSet::new(),
                reasoning: model_metadata.reasoning_support(),
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
        let Some(candidate) = plan.candidates.first() else {
            return Err(ProviderError::NoEligibleEndpoint(request.model.clone()));
        };
        if candidate.endpoint.source != self.surface.source()
            || candidate.endpoint.provider_key != self.surface.source()
        {
            return Err(ProviderError::InvalidRequest(
                "OpenCode adapter received a route for the wrong execution surface".to_owned(),
            ));
        }
        let models = self.models().await?;
        let Some(model) = models.iter().find(|model| model.id == request.model) else {
            return Err(ProviderError::NoEligibleEndpoint(request.model.clone()));
        };
        let body = Self::compile_body(model, request)?;
        let headers = self.request_headers(request)?;
        let response = execute_json_with_retry_headers(
            &self.client,
            Some(&self.api_key),
            &Self::url(model),
            &body,
            &headers,
            retry,
        )
        .await?;
        Self::normalize_response(self.surface, model, response)
    }
}

#[derive(Debug, Deserialize)]
struct LiveModelEnvelope {
    data: Vec<LiveModel>,
}

#[derive(Debug, Deserialize)]
struct LiveModel {
    id: String,
}

#[derive(Debug, Deserialize)]
struct ModelsDevRoot {
    opencode: ModelsDevProvider,
    #[serde(rename = "opencode-go")]
    opencode_go: ModelsDevProvider,
}

#[derive(Debug, Deserialize)]
struct ModelsDevProvider {
    models: BTreeMap<String, ModelsDevModel>,
}

#[derive(Debug, Deserialize)]
struct ModelsDevModel {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    canonical_model_id: Option<String>,
    #[serde(default)]
    reasoning: bool,
    #[serde(default)]
    reasoning_options: Vec<ReasoningOption>,
    #[serde(default)]
    structured_output: bool,
    #[serde(default)]
    temperature: bool,
    #[serde(default)]
    limit: ModelLimit,
    #[serde(default)]
    provider: ModelProvider,
    #[serde(default)]
    cost: Option<Cost>,
}

impl ModelsDevModel {
    fn normalize(self, id: String) -> Option<OpenCodeModel> {
        let dialect = Dialect::from_provider(&self.provider)?;
        let api_base = self.provider.api.clone()?;
        let pricing = self.cost.as_ref()?.pricing();
        let reasoning_efforts = if self.reasoning && matches!(dialect, Dialect::OpenAiResponses) {
            self.reasoning_options
                .iter()
                .flat_map(|option| option.values.iter())
                .filter_map(|effort| parse_reasoning_effort(effort))
                .collect()
        } else {
            BTreeSet::new()
        };
        Some(OpenCodeModel {
            name: self.name.unwrap_or_else(|| id.clone()),
            id,
            canonical_hint: self.canonical_model_id,
            context_length: self.limit.context,
            max_prompt_tokens: self.limit.input,
            max_completion_tokens: self.limit.output,
            pricing,
            dialect,
            temperature: self.temperature,
            structured_output: self.structured_output,
            reasoning_efforts,
            api_base,
        })
    }
}

impl Dialect {
    fn from_provider(provider: &ModelProvider) -> Option<Self> {
        match provider.npm.as_deref() {
            Some("@ai-sdk/anthropic") => Some(Self::AnthropicMessages),
            Some("@ai-sdk/google") => Some(Self::Gemini),
            Some("@ai-sdk/openai") => Some(Self::OpenAiResponses),
            _ if provider
                .api
                .as_deref()
                .is_some_and(|api| api.ends_with("/openai/v1")) =>
            {
                Some(Self::OpenAiChat)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct ModelLimit {
    #[serde(default)]
    context: Option<u64>,
    #[serde(default)]
    input: Option<u64>,
    #[serde(default)]
    output: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
struct ModelProvider {
    #[serde(default)]
    npm: Option<String>,
    #[serde(default)]
    api: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ReasoningOption {
    #[serde(default)]
    values: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
struct Cost {
    #[serde(default)]
    input: f64,
    #[serde(default)]
    output: f64,
    #[serde(default)]
    cache_read: Option<f64>,
    #[serde(default)]
    cache_write: Option<f64>,
    #[serde(default)]
    tiers: Vec<CostTier>,
}

impl Cost {
    fn is_paid(&self) -> bool {
        [
            self.input,
            self.output,
            self.cache_read.unwrap_or(0.0),
            self.cache_write.unwrap_or(0.0),
        ]
        .into_iter()
        .chain(self.tiers.iter().flat_map(CostTier::rates))
        .any(|rate| rate.is_finite() && rate > 0.0)
    }

    fn pricing(&self) -> Pricing {
        Pricing {
            prompt_per_token: self.input / PER_MILLION,
            completion_per_token: self.output / PER_MILLION,
            request: 0.0,
            discount: 0.0,
            cache_read_per_token: self.cache_read.map(|rate| rate / PER_MILLION),
            cache_write_per_token: self.cache_write.map(|rate| rate / PER_MILLION),
            overrides: self
                .tiers
                .iter()
                .filter(|tier| tier.tier.kind == "context")
                .map(|tier| PricingOverride {
                    min_prompt_tokens: tier.tier.size.saturating_add(1),
                    prompt_per_token: tier.input.map(|rate| rate / PER_MILLION),
                    completion_per_token: tier.output.map(|rate| rate / PER_MILLION),
                    cache_read_per_token: tier.cache_read.map(|rate| rate / PER_MILLION),
                    cache_write_per_token: tier.cache_write.map(|rate| rate / PER_MILLION),
                })
                .collect(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct CostTier {
    #[serde(default)]
    input: Option<f64>,
    #[serde(default)]
    output: Option<f64>,
    #[serde(default)]
    cache_read: Option<f64>,
    #[serde(default)]
    cache_write: Option<f64>,
    tier: TierSelector,
}

impl CostTier {
    fn rates(&self) -> impl Iterator<Item = f64> + '_ {
        [self.input, self.output, self.cache_read, self.cache_write]
            .into_iter()
            .flatten()
    }
}

#[derive(Debug, Deserialize)]
struct TierSelector {
    #[serde(rename = "type")]
    kind: String,
    size: u64,
}

fn parse_reasoning_effort(value: &str) -> Option<ReasoningEffort> {
    match value {
        "none" => Some(ReasoningEffort::None),
        "minimal" => Some(ReasoningEffort::Minimal),
        "low" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" => Some(ReasoningEffort::Xhigh),
        "max" => Some(ReasoningEffort::Max),
        _ => None,
    }
}

fn responses_text(raw: &Value) -> Option<String> {
    if let Some(text) = raw.get("output_text").and_then(Value::as_str) {
        return Some(text.to_owned());
    }
    let text = raw
        .get("output")?
        .as_array()?
        .iter()
        .filter_map(|item| item.get("content").and_then(Value::as_array))
        .flatten()
        .filter_map(|content| content.get("text").and_then(Value::as_str))
        .collect::<String>();
    (!text.is_empty()).then_some(text)
}

fn anthropic_text(raw: &Value) -> Option<String> {
    let text = raw
        .get("content")?
        .as_array()?
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|item| item.get("text").and_then(Value::as_str))
        .collect::<String>();
    (!text.is_empty()).then_some(text)
}

fn gemini_text(raw: &Value) -> Option<String> {
    let text = raw
        .pointer("/candidates/0/content/parts")?
        .as_array()?
        .iter()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<String>();
    (!text.is_empty()).then_some(text)
}

fn anthropic_usage(raw: &Value) -> TokenUsage {
    let input = raw.pointer("/usage/input_tokens").and_then(Value::as_u64);
    let output = raw.pointer("/usage/output_tokens").and_then(Value::as_u64);
    TokenUsage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: input
            .zip(output)
            .map(|(input, output)| input.saturating_add(output)),
        reasoning_tokens: raw
            .pointer("/usage/opencode/tokens/reasoning")
            .and_then(Value::as_u64),
    }
}

fn gemini_usage(raw: &Value) -> TokenUsage {
    TokenUsage {
        input_tokens: raw
            .pointer("/usageMetadata/promptTokenCount")
            .and_then(Value::as_u64),
        output_tokens: raw
            .pointer("/usageMetadata/candidatesTokenCount")
            .and_then(Value::as_u64),
        total_tokens: raw
            .pointer("/usageMetadata/totalTokenCount")
            .and_then(Value::as_u64),
        reasoning_tokens: raw
            .pointer("/usageMetadata/thoughtsTokenCount")
            .and_then(Value::as_u64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(value: Value) -> Result<ModelsDevModel, serde_json::Error> {
        serde_json::from_value(value)
    }

    #[test]
    fn paid_responses_model_normalizes_price_tier_and_canonical_hint()
    -> Result<(), Box<dyn std::error::Error>> {
        let model = metadata(json!({
            "name": "GPT 6 Sol",
            "canonical_model_id": "openai/gpt-6-sol",
            "reasoning": true,
            "reasoning_options": [{"type": "effort", "values": ["low", "high"]}],
            "structured_output": true,
            "temperature": false,
            "limit": {"context": 1_050_000, "input": 922_000, "output": 128_000},
            "provider": {"npm": "@ai-sdk/openai", "api": "https://opencode.ai/inference/openai/v1"},
            "cost": {
                "input": 2.0,
                "output": 10.0,
                "cache_read": 0.2,
                "cache_write": 2.5,
                "tiers": [{
                    "input": 4.0,
                    "output": 15.0,
                    "cache_read": 0.4,
                    "cache_write": 5.0,
                    "tier": {"type": "context", "size": 272_000}
                }]
            }
        }))?;
        let normalized = model
            .normalize("gpt-6-sol".to_owned())
            .ok_or("known dialect missing")?;
        assert_eq!(
            normalized.canonical_hint.as_deref(),
            Some("openai/gpt-6-sol")
        );
        assert_eq!(normalized.dialect, Dialect::OpenAiResponses);
        assert_eq!(normalized.pricing.prompt_per_token, 2.0 / PER_MILLION);
        assert_eq!(normalized.pricing.overrides[0].min_prompt_tokens, 272_001);
        assert!(
            normalized
                .reasoning_efforts
                .contains(&ReasoningEffort::High)
        );
        Ok(())
    }

    #[test]
    fn zero_cost_model_is_not_paid() -> Result<(), serde_json::Error> {
        let cost: Cost = serde_json::from_value(json!({
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0
        }))?;
        assert!(!cost.is_paid());
        Ok(())
    }

    #[test]
    fn provider_metadata_selects_all_four_wire_families() {
        let provider = |npm: Option<&str>, api: &str| ModelProvider {
            npm: npm.map(str::to_owned),
            api: Some(api.to_owned()),
        };
        assert_eq!(
            Dialect::from_provider(&provider(Some("@ai-sdk/openai"), "https://x/openai/v1")),
            Some(Dialect::OpenAiResponses)
        );
        assert_eq!(
            Dialect::from_provider(&provider(
                Some("@ai-sdk/anthropic"),
                "https://x/anthropic/v1"
            )),
            Some(Dialect::AnthropicMessages)
        );
        assert_eq!(
            Dialect::from_provider(&provider(Some("@ai-sdk/google"), "https://x/google/v1beta")),
            Some(Dialect::Gemini)
        );
        assert_eq!(
            Dialect::from_provider(&provider(None, "https://x/openai/v1")),
            Some(Dialect::OpenAiChat)
        );
    }

    #[test]
    fn responses_body_encodes_reasoning_and_strict_schema() -> Result<(), ProviderError> {
        let model = OpenCodeModel {
            id: "gpt-6-sol".to_owned(),
            name: "GPT 6 Sol".to_owned(),
            canonical_hint: None,
            context_length: None,
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            dialect: Dialect::OpenAiResponses,
            temperature: false,
            structured_output: true,
            reasoning_efforts: BTreeSet::from([ReasoningEffort::High]),
            api_base: "https://example.test/openai/v1".to_owned(),
        };
        let mut request = InferenceRequest::new("gpt-6-sol", "hi", 8);
        request.requirements.reasoning = ReasoningEffort::High;
        request.requirements.response_schema = Some(ResponseSchema::JsonSchema {
            name: "result".to_owned(),
            schema: json!({"type": "object"}),
            strict: true,
        });
        let body = OpenCodeConsole::compile_body(&model, &request)?;
        assert_eq!(
            body.pointer("/reasoning/effort").and_then(Value::as_str),
            Some("high")
        );
        assert_eq!(
            body.pointer("/text/format/type").and_then(Value::as_str),
            Some("json_schema")
        );
        assert_eq!(
            body.pointer("/text/format/strict").and_then(Value::as_bool),
            Some(true)
        );
        Ok(())
    }

    #[test]
    fn compile_body_never_silently_drops_unsupported_controls() {
        let model = OpenCodeModel {
            id: "claude-sonnet-5-5".to_owned(),
            name: "Claude Sonnet 5.5".to_owned(),
            canonical_hint: None,
            context_length: None,
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            dialect: Dialect::AnthropicMessages,
            temperature: true,
            structured_output: true,
            reasoning_efforts: BTreeSet::new(),
            api_base: "https://example.test/anthropic/v1".to_owned(),
        };
        let mut request = InferenceRequest::new("claude-sonnet-5-5", "hi", 8);
        request.requirements.reasoning = ReasoningEffort::High;
        assert!(OpenCodeConsole::compile_body(&model, &request).is_err());
        request.requirements.reasoning = ReasoningEffort::None;
        request.requirements.response_schema = Some(ResponseSchema::JsonObject);
        assert!(OpenCodeConsole::compile_body(&model, &request).is_err());
    }

    #[test]
    fn anthropic_body_always_has_required_max_tokens() {
        let request = InferenceRequest::new("claude-sonnet-5-5", "hi", 64);
        let body = OpenCodeConsole::compile_anthropic(&request);
        assert_eq!(body.get("max_tokens").and_then(Value::as_u64), Some(64));
    }

    #[test]
    fn go_requires_agentic_workload_and_stable_session_headers() -> Result<(), ProviderError> {
        let go = OpenCodeConsole::new_go("test-key");
        assert_eq!(go.surface.route_class(), ModelRouteClass::Subscription);
        assert_eq!(
            go.surface.workload_policy().agentic,
            AgenticPolicy::Required
        );
        assert!(go.surface.workload_policy().requires_stable_session);

        let mut request = InferenceRequest::new("model", "work", 8);
        assert!(go.request_headers(&request).is_err());
        request.agentic = true;
        assert!(go.request_headers(&request).is_err());
        request.session_id = Some("conversation-42".to_owned());
        let headers = go.request_headers(&request)?;
        assert_eq!(
            headers
                .get(USER_AGENT)
                .and_then(|value| value.to_str().ok()),
            Some(USER_AGENT_VALUE)
        );
        assert_eq!(
            headers
                .get("x-opencode-session")
                .and_then(|value| value.to_str().ok()),
            Some("conversation-42")
        );
        Ok(())
    }

    #[test]
    fn payg_and_go_apply_different_zero_cost_inventory_policy() -> Result<(), serde_json::Error> {
        let zero: Cost = serde_json::from_value(json!({
            "input": 0,
            "output": 0,
            "cache_read": 0
        }))?;
        assert!(!OpenCodeSurface::PayAsYouGo.accepts_cost(&zero));
        assert!(OpenCodeSurface::Go.accepts_cost(&zero));
        Ok(())
    }

    #[test]
    fn normalizes_gateway_cost_and_usage_across_response_families() -> Result<(), ProviderError> {
        let base = OpenCodeModel {
            id: "test".to_owned(),
            name: "Test".to_owned(),
            canonical_hint: None,
            context_length: None,
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            dialect: Dialect::OpenAiResponses,
            temperature: false,
            structured_output: false,
            reasoning_efforts: BTreeSet::new(),
            api_base: "https://example.test/openai/v1".to_owned(),
        };
        let response = OpenCodeConsole::normalize_response(
            OpenCodeSurface::PayAsYouGo,
            &base,
            JsonResponse {
                raw: json!({
                    "model": "test",
                    "output": [{"content": [{"type": "output_text", "text": "PONG"}]}],
                    "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12, "cost": 0.001}
                }),
                quota_observations: Vec::new(),
            },
        )?;
        assert_eq!(response.content, "PONG");
        assert_eq!(response.usage_cost_usd, Some(0.001));
        assert_eq!(response.usage.total_tokens, Some(12));

        let mut gemini = base;
        gemini.dialect = Dialect::Gemini;
        let response = OpenCodeConsole::normalize_response(
            OpenCodeSurface::PayAsYouGo,
            &gemini,
            JsonResponse {
                raw: json!({
                    "modelVersion": "gemini-test",
                    "candidates": [{"content": {"parts": [{"text": "PONG"}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 3,
                        "candidatesTokenCount": 4,
                        "totalTokenCount": 9,
                        "thoughtsTokenCount": 2,
                        "cost": 0.002
                    }
                }),
                quota_observations: Vec::new(),
            },
        )?;
        assert_eq!(response.content, "PONG");
        assert_eq!(response.usage.reasoning_tokens, Some(2));
        assert_eq!(response.usage_cost_usd, Some(0.002));
        Ok(())
    }
}
