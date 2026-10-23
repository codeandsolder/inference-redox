use async_trait::async_trait;

use super::openai_compatible::{
    OpenAiCompatibleProfile, OpenAiCompatibleProvider, ReasoningDialect, TemperatureZeroPolicy,
    TokenLimitField,
};
use crate::{
    Endpoint, InferenceProvider, InferenceRequest, InferenceResponse, ModelCatalog,
    ModelRouteClass, Pricing, ProviderError, ProviderInventory, QuotaMetric, QuotaResource,
    QuotaRouteKey, QuotaScope, ReasoningSupport, RetryPolicy, RoutePlan,
};

/// Direct Kilo AI Gateway adapter. Free routes may be used anonymously; an account key is optional.
#[derive(Debug, Clone)]
pub struct Kilo {
    inner: OpenAiCompatibleProvider,
}

impl Kilo {
    #[must_use]
    pub fn new(api_key: Option<String>) -> Self {
        let adapter_parameters = [
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
        .collect();
        let nominal_parameters = [
            "chat_completions",
            "system_role",
            "max_tokens",
            "temperature",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        Self {
            inner: OpenAiCompatibleProvider::new_optional(
                api_key,
                OpenAiCompatibleProfile {
                    source: "kilo".to_owned(),
                    provider_key: "kilo".to_owned(),
                    provider_name: "Kilo AI Gateway".to_owned(),
                    base_url: "https://ai-gateway.kilo.ai/api/v1".to_owned(),
                    route_class: ModelRouteClass::Free,
                    token_limit_field: TokenLimitField::MaxTokens,
                    reasoning_dialect: ReasoningDialect::Unsupported,
                    temperature_zero: TemperatureZeroPolicy::PassThrough,
                    adapter_parameters,
                    nominal_parameters,
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

    fn explicit_free_chat_route(model: &str) -> bool {
        model.ends_with(":free") || matches!(model, "kilo-auto/free" | "openrouter/free")
    }

    #[must_use]
    pub fn free_ip_request_resource() -> QuotaResource {
        QuotaResource::known_capacity(
            "kilo/ip/free-rph",
            QuotaScope::NetworkOrigin,
            QuotaMetric::Requests,
            200,
        )
    }

    #[must_use]
    pub fn free_quota_route(model: impl Into<String>) -> QuotaRouteKey {
        QuotaRouteKey {
            source: "kilo".to_owned(),
            credential: None,
            provider: "kilo".to_owned(),
            model: model.into(),
        }
    }

    fn zero_pricing(pricing: &Pricing) -> bool {
        pricing.prompt_per_token == 0.0
            && pricing.completion_per_token == 0.0
            && pricing.request == 0.0
    }

    /// Reconcile the complete Kilo inventory while only making explicit zero-price free chat
    /// routes executable. Other rows keep their nominal metadata for diagnostics.
    ///
    /// # Errors
    /// Returns a catalog error when the unauthenticated `/models` endpoint cannot be loaded and no
    /// stale cache is available.
    pub async fn inventory(
        &self,
        catalog: &ModelCatalog,
    ) -> Result<ProviderInventory, ProviderError> {
        let mut inventory = self.inner.inventory(catalog).await?;
        for model in &mut inventory.models {
            let executable = Self::explicit_free_chat_route(&model.native_id)
                && model.pricing.as_ref().is_some_and(Self::zero_pricing);
            model.route_class = executable.then_some(ModelRouteClass::Free);
        }
        Ok(inventory)
    }
}

#[async_trait]
impl InferenceProvider for Kilo {
    async fn endpoints(&self, model: &str) -> Result<Vec<Endpoint>, ProviderError> {
        if !Self::explicit_free_chat_route(model) {
            return Err(ProviderError::NoEligibleEndpoint(model.to_owned()));
        }
        let mut endpoints = self.inner.endpoints(model).await?;
        endpoints.retain(|endpoint| Self::zero_pricing(&endpoint.pricing));
        if endpoints.is_empty() {
            return Err(ProviderError::NoEligibleEndpoint(model.to_owned()));
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

    use super::*;

    #[test]
    fn free_classification_is_not_zero_price_classification() {
        assert!(Kilo::explicit_free_chat_route("vendor/model:free"));
        assert!(Kilo::explicit_free_chat_route("kilo-auto/free"));
        assert!(Kilo::explicit_free_chat_route("openrouter/free"));
        assert!(!Kilo::explicit_free_chat_route(
            "google/lyria-3-pro-preview"
        ));
    }

    #[tokio::test]
    async fn inventory_retains_nominal_claims_but_routes_only_explicit_free_models()
    -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let body = serde_json::json!({
            "data": [
                {
                    "id": "vendor/chat:free",
                    "name": "Free Chat",
                    "supported_parameters": ["max_tokens", "tools"],
                    "pricing": {"prompt": "0", "completion": "0"},
                    "context_length": 100_000,
                    "top_provider": {"max_completion_tokens": 32000}
                },
                {
                    "id": "vendor/paid",
                    "supported_parameters": ["max_tokens", "structured_outputs"],
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"}
                },
                {
                    "id": "google/audio-preview",
                    "supported_parameters": ["max_tokens"],
                    "pricing": {"prompt": "0", "completion": "0"}
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
        let provider = Kilo::new(None).with_base_url(format!("http://{address}"));
        let inventory = provider.inventory(&ModelCatalog::default()).await?;
        let _ = server.join();
        let free = inventory
            .models
            .iter()
            .find(|model| model.native_id == "vendor/chat:free")
            .ok_or("missing free model")?;
        assert_eq!(free.route_class, Some(ModelRouteClass::Free));
        assert!(free.claimed_parameters.contains("tools"));
        assert_eq!(free.max_completion_tokens, Some(32_000));
        let paid = inventory
            .models
            .iter()
            .find(|model| model.native_id == "vendor/paid")
            .ok_or("missing paid model")?;
        assert_eq!(paid.route_class, None);
        assert!(paid.claimed_parameters.contains("structured_outputs"));
        let zero_nonfree = inventory
            .models
            .iter()
            .find(|model| model.native_id == "google/audio-preview")
            .ok_or("missing zero-price nonfree model")?;
        assert_eq!(zero_nonfree.route_class, None);
        Ok(())
    }
}
