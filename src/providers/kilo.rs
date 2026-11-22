use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;

use super::native_catalog::NativeCatalog;
use super::openai_compatible::{base_chat_body, execute_chat_with_retry, response_format};
use crate::{
    Endpoint, EndpointCapabilities, EndpointStats, InferenceProvider, InferenceRequest,
    InferenceResponse, ModelCatalog, ModelRouteClass, NativeModelRecord, Pricing, ProviderError,
    ProviderInventory, QuotaLedger, QuotaMetric, QuotaResource, QuotaResourceId, QuotaRouteKey,
    QuotaScope, ReasoningEffort, ReasoningSupport, RetryPolicy, RoutePlan,
};

const DEFAULT_BASE_URL: &str = "https://ai-gateway.kilo.ai/api/v1";

/// Free-only Kilo Gateway adapter. Paid Kilo routes are intentionally excluded.
#[derive(Debug, Clone)]
#[must_use = "the configured provider must be retained to issue inference requests"]
pub struct KiloGateway {
    catalog: NativeCatalog<Option<String>>,
    quota_pool: Option<String>,
}

impl KiloGateway {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            catalog: NativeCatalog::new(api_key, DEFAULT_BASE_URL),
            quota_pool: None,
        }
    }

    pub fn anonymous() -> Self {
        Self::new(None)
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.catalog.set_base_url(base_url);
        self
    }

    pub fn with_quota_pool(mut self, quota_pool: impl Into<String>) -> Self {
        self.quota_pool = Some(quota_pool.into());
        self
    }

    /// Fetch Kilo's current free routes and reconcile them against canonical identity.
    ///
    /// # Errors
    /// Returns a catalog error when the keyless model inventory cannot be fetched or decoded.
    pub async fn inventory(
        &self,
        catalog: &ModelCatalog,
    ) -> Result<ProviderInventory, ProviderError> {
        Ok(ProviderInventory::reconcile(
            "kilo",
            self.native_models().await?,
            catalog,
        ))
    }

    async fn native_models(&self) -> Result<Vec<NativeModelRecord>, ProviderError> {
        self.catalog
            .models(async {
                let response = self
                    .catalog
                    .client
                    .get(format!("{}/models", self.catalog.base_url))
                    .send()
                    .await
                    .map_err(|error| ProviderError::Catalog(error.to_string()))?;
                let status = response.status();
                if !status.is_success() {
                    return Err(ProviderError::Catalog(format!("HTTP {status}")));
                }
                let envelope: KiloModelEnvelope = response
                    .json()
                    .await
                    .map_err(|error| ProviderError::Catalog(error.to_string()))?;
                Ok::<Vec<NativeModelRecord>, ProviderError>(
                    envelope
                        .data
                        .into_iter()
                        .filter(KiloModel::is_free)
                        .map(KiloModel::normalize)
                        .collect(),
                )
            })
            .await
    }

    fn compile_body(request: &InferenceRequest) -> Result<Value, ProviderError> {
        if request.requirements.reasoning != ReasoningEffort::None {
            return Err(ProviderError::InvalidRequest(
                "Kilo reasoning controls are not normalized yet".to_owned(),
            ));
        }
        let mut body = base_chat_body(request, "max_tokens");
        if let Some(schema) = request.requirements.response_schema.as_ref() {
            body.insert("response_format".to_owned(), response_format(schema));
        }
        body.insert("stream".to_owned(), Value::Bool(false));
        Ok(Value::Object(body))
    }
}

#[async_trait]
impl InferenceProvider for KiloGateway {
    async fn endpoints(&self, model: &str) -> Result<Vec<Endpoint>, ProviderError> {
        let models = self.native_models().await?;
        let Some(native) = models.iter().find(|candidate| candidate.id == model) else {
            return Err(ProviderError::NoEligibleEndpoint(model.to_owned()));
        };
        Ok(vec![Endpoint {
            source: "kilo".to_owned(),
            credential: None,
            provider_key: "kilo".to_owned(),
            id: "kilo".to_owned(),
            provider: "Kilo Gateway".to_owned(),
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
                    "max_tokens",
                    "temperature",
                    "response_format",
                    "structured_outputs",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
                supported_parameters: native.claimed_parameters.clone(),
                verified_parameters: BTreeSet::new(),
                rejected_parameters: BTreeSet::new(),
                verified_reasoning_efforts: BTreeSet::new(),
                rejected_reasoning_efforts: BTreeSet::new(),
                constraint_enforcement: BTreeMap::new(),
                verified_schema_features: BTreeMap::new(),
                rejected_schema_features: BTreeMap::new(),
                reasoning: ReasoningSupport::None,
                supports_implicit_caching: false,
            },
        }])
    }

    fn configure_quota_route(
        &self,
        route: &QuotaRouteKey,
        ledger: &mut QuotaLedger,
    ) -> Result<(), ProviderError> {
        if route.source != "kilo" || route.provider != "kilo" {
            return Err(ProviderError::InvalidRequest(
                "Kilo quota topology received a non-Kilo route".to_owned(),
            ));
        }
        let Some(pool) = self.quota_pool.as_deref() else {
            return Ok(());
        };
        let id = QuotaResourceId(format!("kilo/{pool}/free-requests"));
        if ledger.graph().resource(&id).is_none() {
            ledger.graph_mut().insert_resource(QuotaResource::unknown(
                id.0.clone(),
                QuotaScope::Provider,
                QuotaMetric::Requests,
            ));
        }
        let _ = ledger.graph_mut().bind(route.clone(), &id);
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
        if candidate.endpoint.source != "kilo" || candidate.endpoint.provider_key != "kilo" {
            return Err(ProviderError::InvalidRequest(
                "Kilo adapter received a non-Kilo route".to_owned(),
            ));
        }
        let body = Self::compile_body(request)?;
        execute_chat_with_retry(
            &self.catalog.client,
            self.catalog.api_key.as_deref(),
            &format!("{}/chat/completions", self.catalog.base_url),
            &body,
            "kilo",
            Some(0.0),
            retry,
        )
        .await
    }
}

#[derive(Debug, Deserialize)]
struct KiloModelEnvelope {
    data: Vec<KiloModel>,
}

#[derive(Debug, Deserialize)]
struct KiloTopProvider {
    #[serde(default)]
    max_completion_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct KiloModel {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    context_length: Option<u64>,
    #[serde(default)]
    top_provider: Option<KiloTopProvider>,
    #[serde(default)]
    supported_parameters: BTreeSet<String>,
    #[serde(default, rename = "isFree")]
    is_free: bool,
}

impl KiloModel {
    fn is_free(&self) -> bool {
        self.is_free || self.id == "kilo-auto/free" || self.id.ends_with(":free")
    }

    fn normalize(self) -> NativeModelRecord {
        let virtual_route = self.id.starts_with("kilo-auto/");
        NativeModelRecord {
            id: self.id,
            display_name: self.name,
            owned_by: Some("kilo".to_owned()),
            created_unix: self.created,
            context_length: self.context_length,
            max_completion_tokens: self
                .top_provider
                .and_then(|provider| provider.max_completion_tokens),
            claimed_parameters: self.supported_parameters,
            pricing: Some(Pricing::default()),
            route_class: Some(ModelRouteClass::Free),
            canonical_hint: None,
            virtual_route,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_free_catalog_routes_are_retained() -> Result<(), Box<dyn std::error::Error>> {
        let envelope: KiloModelEnvelope = serde_json::from_value(serde_json::json!({
            "data": [
                {"id":"paid/model","isFree":false,"context_length":1000},
                {"id":"free/model:free","isFree":true,"context_length":2000,
                 "top_provider":{"max_completion_tokens":500},
                 "supported_parameters":["max_tokens","response_format"]}
            ]
        }))?;
        let models = envelope
            .data
            .into_iter()
            .filter(KiloModel::is_free)
            .map(KiloModel::normalize)
            .collect::<Vec<_>>();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "free/model:free");
        assert_eq!(models[0].max_completion_tokens, Some(500));
        assert!(models[0].claimed_parameters.contains("response_format"));
        Ok(())
    }

    #[test]
    fn free_quota_pool_is_shared_across_models() -> Result<(), ProviderError> {
        let provider = KiloGateway::anonymous().with_quota_pool("egress-a");
        let route = |model: &str| QuotaRouteKey {
            source: "kilo".to_owned(),
            credential: None,
            provider: "kilo".to_owned(),
            model: model.to_owned(),
        };
        let mut ledger = QuotaLedger::new(crate::QuotaGraph::new());
        provider.configure_quota_route(&route("a:free"), &mut ledger)?;
        provider.configure_quota_route(&route("b:free"), &mut ledger)?;
        let a = ledger.evaluate(&route("a:free"), crate::QuotaDemand::default(), 1);
        let b = ledger.evaluate(&route("b:free"), crate::QuotaDemand::default(), 1);
        assert_eq!(a.checks.len(), 1);
        assert_eq!(a.checks[0].resource_id, b.checks[0].resource_id);
        assert_eq!(a.checks[0].scope, QuotaScope::Provider);
        Ok(())
    }
}
