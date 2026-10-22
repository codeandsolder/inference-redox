use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    InferenceProvider, InferenceRequest, InferenceResponse, ModelCatalog, ModelLookup,
    ModelRouteClass, ProviderError, RetryPolicy, RouteCandidate, RoutePlan, RoutingStrategy,
    build_route_plan,
};

/// Cross-provider coordinator. Catalog identity and provider execution stay separate:
/// callers name a canonical model/alias, while each adapter receives its native route ID.
#[derive(Clone, Default)]
pub struct InferencePool {
    providers: BTreeMap<String, Arc<dyn InferenceProvider>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PooledInferenceResponse {
    pub plan: RoutePlan,
    pub selected: RouteCandidate,
    pub response: InferenceResponse,
}

impl InferencePool {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<P>(&mut self, source: impl Into<String>, provider: Arc<P>)
    where
        P: InferenceProvider + 'static,
    {
        self.providers.insert(source.into(), provider);
    }

    pub fn register_dyn(
        &mut self,
        source: impl Into<String>,
        provider: Arc<dyn InferenceProvider>,
    ) {
        self.providers.insert(source.into(), provider);
    }

    /// Discover provider-native endpoints for the request's canonical model/alias.
    /// Route classes not admitted by the request are not queried.
    ///
    /// # Errors
    /// Returns an error when model identity is ambiguous/unresolved or no registered provider
    /// exposes an eligible route class for it.
    pub async fn endpoints(
        &self,
        catalog: &ModelCatalog,
        request: &InferenceRequest,
    ) -> Result<Vec<crate::Endpoint>, ProviderError> {
        let resolved = match catalog.resolve(&request.model) {
            ModelLookup::Resolved { model } => model,
            ModelLookup::Ambiguous { canonical_ids } => {
                return Err(ProviderError::InvalidRequest(format!(
                    "ambiguous model {}: {}",
                    request.model,
                    canonical_ids.join(", ")
                )));
            }
            ModelLookup::NotFound => {
                return Err(ProviderError::NoEligibleEndpoint(request.model.clone()));
            }
        };

        let mut endpoints = Vec::new();
        for route in resolved.routes {
            if !request
                .requirements
                .allowed_route_classes
                .contains(&route.route_class)
            {
                continue;
            }
            let Some(provider) = self.providers.get(&route.source) else {
                continue;
            };
            let Ok(mut discovered) = provider.endpoints(&route.id).await else {
                continue;
            };
            for endpoint in &mut discovered {
                // The catalog route is authoritative for execution class and model spelling.
                endpoint.source.clone_from(&route.source);
                endpoint.model.clone_from(&route.id);
                endpoint.route_class = route.route_class;
            }
            endpoints.extend(discovered);
        }

        if endpoints.is_empty() {
            return Err(ProviderError::NoEligibleEndpoint(request.model.clone()));
        }
        Ok(endpoints)
    }

    /// Build one cost/performance plan across every registered execution surface.
    ///
    /// # Errors
    /// Returns normal catalog/routing errors when no route can satisfy the request.
    pub async fn route(
        &self,
        catalog: &ModelCatalog,
        strategy: RoutingStrategy,
        request: &InferenceRequest,
    ) -> Result<RoutePlan, ProviderError> {
        let endpoints = self.endpoints(catalog, request).await?;
        build_route_plan(strategy, request, endpoints)
    }

    /// Execute a synchronous request across ranked standard/free routes, failing over across
    /// provider surfaces when a candidate fails. Batch is intentionally a separate async API.
    ///
    /// # Errors
    /// Returns an error for batch-admitting requests, routing failure, or when all ranked
    /// synchronous candidates fail.
    pub async fn infer(
        &self,
        catalog: &ModelCatalog,
        strategy: RoutingStrategy,
        request: &InferenceRequest,
        retry: RetryPolicy,
    ) -> Result<PooledInferenceResponse, ProviderError> {
        if request
            .requirements
            .allowed_route_classes
            .contains(&ModelRouteClass::Batch)
        {
            return Err(ProviderError::InvalidRequest(
                "synchronous infer cannot admit batch routes; use the batch API".to_owned(),
            ));
        }
        let plan = self.route(catalog, strategy, request).await?;
        let mut attempted = 0_usize;
        for candidate in plan.candidates.clone() {
            let Some(provider) = self.providers.get(&candidate.endpoint.source) else {
                continue;
            };
            attempted += 1;
            let mut native_request = request.clone();
            native_request.model.clone_from(&candidate.endpoint.model);
            let candidate_plan = RoutePlan {
                prompt_tokens: plan.prompt_tokens,
                expected_output_tokens: plan.expected_output_tokens,
                cheapest_cost_usd: candidate.expected_cost_usd,
                cost_ceiling_usd: candidate.expected_cost_usd,
                candidates: vec![candidate.clone()],
            };
            if let Ok(response) = provider
                .execute_plan(&native_request, &candidate_plan, retry)
                .await
            {
                return Ok(PooledInferenceResponse {
                    plan,
                    selected: candidate,
                    response,
                });
            }
        }
        Err(ProviderError::Request(format!(
            "all {attempted} ranked provider candidates failed"
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Mutex as StdMutex;

    use async_trait::async_trait;
    use serde_json::json;

    use super::*;
    use crate::{
        Endpoint, EndpointCapabilities, EndpointStats, ModelArchitecture, ModelBenchmarks,
        ModelCatalogEntry, ModelReasoning, ModelTopProvider, Pricing, ReasoningSupport,
    };

    struct MockProvider {
        source: &'static str,
        price: f64,
        fail_execute: bool,
        calls: Arc<StdMutex<Vec<String>>>,
    }

    #[async_trait]
    impl InferenceProvider for MockProvider {
        async fn endpoints(&self, model: &str) -> Result<Vec<Endpoint>, ProviderError> {
            self.calls
                .lock()
                .map_err(|_| ProviderError::Catalog("poisoned test mutex".to_owned()))?
                .push(model.to_owned());
            Ok(vec![Endpoint {
                source: self.source.to_owned(),
                provider_key: self.source.to_owned(),
                id: self.source.to_owned(),
                provider: self.source.to_owned(),
                model: model.to_owned(),
                route_class: ModelRouteClass::Standard,
                display_name: self.source.to_owned(),
                quantization: None,
                status: 0,
                context_length: None,
                max_prompt_tokens: None,
                max_completion_tokens: None,
                pricing: Pricing {
                    prompt_per_token: self.price,
                    completion_per_token: self.price,
                    ..Pricing::default()
                },
                stats: EndpointStats::default(),
                capabilities: EndpointCapabilities {
                    adapter_parameters: BTreeSet::from(["max_tokens".to_owned()]),
                    supported_parameters: BTreeSet::from(["max_tokens".to_owned()]),
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
            _plan: &RoutePlan,
            _retry: RetryPolicy,
        ) -> Result<InferenceResponse, ProviderError> {
            if self.fail_execute {
                return Err(ProviderError::Request("mock failure".to_owned()));
            }
            Ok(InferenceResponse {
                content: "ok".to_owned(),
                model: Some(request.model.clone()),
                provider: Some(self.source.to_owned()),
                usage_cost_usd: None,
                raw: json!({"native_model": request.model}),
            })
        }
    }

    fn entry(source: &str, id: &str, class: ModelRouteClass) -> ModelCatalogEntry {
        ModelCatalogEntry {
            id: id.to_owned(),
            route_class: class,
            canonical_id: "vendor/model-v1".to_owned(),
            name: "Model".to_owned(),
            description: None,
            aliases: BTreeSet::from([id.to_owned(), "vendor/model".to_owned()]),
            created_unix: None,
            context_length: None,
            architecture: ModelArchitecture::default(),
            pricing: Pricing::default(),
            pricing_known: true,
            top_provider: ModelTopProvider::default(),
            supported_parameters: BTreeSet::new(),
            reasoning: ModelReasoning::default(),
            benchmarks: ModelBenchmarks::default(),
            knowledge_cutoff: None,
            expiration_date: None,
            source: source.to_owned(),
        }
    }

    fn catalog() -> ModelCatalog {
        ModelCatalog {
            models: vec![
                entry("openrouter", "vendor/model", ModelRouteClass::Standard),
                entry("nvidia", "vendor/native-model", ModelRouteClass::Free),
                entry("openrouter", "vendor/model:batch", ModelRouteClass::Batch),
            ],
        }
    }

    #[tokio::test]
    async fn default_policy_queries_standard_and_free_but_not_batch() -> Result<(), ProviderError> {
        let openrouter_calls = Arc::new(StdMutex::new(Vec::new()));
        let nvidia_calls = Arc::new(StdMutex::new(Vec::new()));
        let mut pool = InferencePool::new();
        pool.register(
            "openrouter",
            Arc::new(MockProvider {
                source: "openrouter",
                price: 1e-6,
                fail_execute: false,
                calls: Arc::clone(&openrouter_calls),
            }),
        );
        pool.register(
            "nvidia",
            Arc::new(MockProvider {
                source: "nvidia",
                price: 0.0,
                fail_execute: false,
                calls: Arc::clone(&nvidia_calls),
            }),
        );
        let request = InferenceRequest::new("vendor/model", "hi", 8);
        let endpoints = pool.endpoints(&catalog(), &request).await?;
        assert_eq!(endpoints.len(), 2);
        let calls = openrouter_calls
            .lock()
            .map_err(|_| ProviderError::Catalog("poisoned test mutex".to_owned()))?;
        assert_eq!(calls.as_slice(), ["vendor/model"]);
        let calls = nvidia_calls
            .lock()
            .map_err(|_| ProviderError::Catalog("poisoned test mutex".to_owned()))?;
        assert_eq!(calls.as_slice(), ["vendor/native-model"]);
        Ok(())
    }

    #[tokio::test]
    async fn batch_only_policy_queries_only_batch_route() -> Result<(), ProviderError> {
        let calls = Arc::new(StdMutex::new(Vec::new()));
        let mut pool = InferencePool::new();
        pool.register(
            "openrouter",
            Arc::new(MockProvider {
                source: "openrouter",
                price: 1e-6,
                fail_execute: false,
                calls: Arc::clone(&calls),
            }),
        );
        let mut request = InferenceRequest::new("vendor/model", "bulk", 8);
        request.requirements.allowed_route_classes = BTreeSet::from([ModelRouteClass::Batch]);
        let endpoints = pool.endpoints(&catalog(), &request).await?;
        assert_eq!(endpoints.len(), 1);
        assert_eq!(endpoints[0].route_class, ModelRouteClass::Batch);
        let calls = calls
            .lock()
            .map_err(|_| ProviderError::Catalog("poisoned test mutex".to_owned()))?;
        assert_eq!(calls.as_slice(), ["vendor/model:batch"]);
        Ok(())
    }

    #[tokio::test]
    async fn cheapest_direct_route_executes_with_native_model_id() -> Result<(), ProviderError> {
        let calls = Arc::new(StdMutex::new(Vec::new()));
        let mut pool = InferencePool::new();
        pool.register(
            "openrouter",
            Arc::new(MockProvider {
                source: "openrouter",
                price: 1e-6,
                fail_execute: false,
                calls: Arc::clone(&calls),
            }),
        );
        pool.register(
            "nvidia",
            Arc::new(MockProvider {
                source: "nvidia",
                price: 0.0,
                fail_execute: false,
                calls,
            }),
        );
        let request = InferenceRequest::new("vendor/model", "hi", 8);
        let result = pool
            .infer(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
                RetryPolicy::default(),
            )
            .await?;
        assert_eq!(result.selected.endpoint.source, "nvidia");
        assert_eq!(result.response.provider.as_deref(), Some("nvidia"));
        assert_eq!(
            result
                .response
                .raw
                .get("native_model")
                .and_then(|v| v.as_str()),
            Some("vendor/native-model")
        );
        Ok(())
    }

    #[tokio::test]
    async fn synchronous_infer_fails_over_across_provider_surfaces() -> Result<(), ProviderError> {
        let calls = Arc::new(StdMutex::new(Vec::new()));
        let mut pool = InferencePool::new();
        pool.register(
            "nvidia",
            Arc::new(MockProvider {
                source: "nvidia",
                price: 0.0,
                fail_execute: true,
                calls: Arc::clone(&calls),
            }),
        );
        pool.register(
            "openrouter",
            Arc::new(MockProvider {
                source: "openrouter",
                price: 0.0,
                fail_execute: false,
                calls,
            }),
        );
        let request = InferenceRequest::new("vendor/model", "hi", 8);
        let result = pool
            .infer(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
                RetryPolicy::default(),
            )
            .await?;
        assert_eq!(result.selected.endpoint.source, "openrouter");
        assert_eq!(result.response.provider.as_deref(), Some("openrouter"));
        Ok(())
    }

    #[tokio::test]
    async fn synchronous_infer_rejects_batch_admission() {
        let mut request = InferenceRequest::new("vendor/model", "hi", 8);
        request
            .requirements
            .allowed_route_classes
            .insert(ModelRouteClass::Batch);
        let result = InferencePool::new()
            .infer(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
                RetryPolicy::default(),
            )
            .await;
        assert!(matches!(result, Err(ProviderError::InvalidRequest(_))));
    }
}
