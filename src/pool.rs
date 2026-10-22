use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::router::supports_request;
use crate::{
    InferenceProvider, InferenceRequest, InferenceResponse, ModelCatalog, ModelLookup,
    ModelRouteClass, ProviderError, QuotaAdmission, QuotaAdmissionReport, QuotaAwareRoutePlan,
    QuotaDemand, QuotaLedger, QuotaRouteKey, QuotaUnknownPolicy, RetryPolicy, RouteCandidate,
    RoutePlan, RoutingStrategy, build_route_plan, request_prompt_tokens,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ProviderInstanceKey {
    source: String,
    credential: Option<String>,
}

impl ProviderInstanceKey {
    fn new(source: impl Into<String>, credential: Option<String>) -> Self {
        Self {
            source: source.into(),
            credential,
        }
    }
}

/// Cross-provider coordinator. Catalog identity and provider execution stay separate:
/// callers name a canonical model/alias, while each adapter receives its native route ID.
#[derive(Clone, Default)]
pub struct InferencePool {
    providers: BTreeMap<ProviderInstanceKey, Arc<dyn InferenceProvider>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PooledInferenceResponse {
    pub plan: RoutePlan,
    pub selected: RouteCandidate,
    pub response: InferenceResponse,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuotaPooledInferenceResponse {
    pub quota_plan: QuotaAwareRoutePlan,
    pub selected: RouteCandidate,
    pub response: InferenceResponse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaExecutionContext {
    pub unknown_policy: QuotaUnknownPolicy,
    pub now_unix: u64,
}

#[derive(Debug, Error)]
pub enum QuotaRoutingError {
    #[error(transparent)]
    Provider(#[from] ProviderError),
    #[error("no quota-admissible endpoint for {model}")]
    NoQuotaAdmissible {
        model: String,
        reports: Box<Vec<QuotaAdmissionReport>>,
    },
}

#[derive(Debug, Error)]
pub enum QuotaInferenceError {
    #[error(transparent)]
    Routing(#[from] QuotaRoutingError),
    #[error("all {attempted} quota-admissible provider candidates failed")]
    AllCandidatesFailed {
        attempted: usize,
        last_error: Option<Box<ProviderError>>,
    },
    #[error("quota ledger invariant failed: {0}")]
    LedgerInvariant(String),
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
        self.register_dyn(source, provider);
    }

    /// Register another logical credential/account for one execution surface.
    /// The label is opaque routing metadata and must never contain the secret itself.
    pub fn register_credential<P>(
        &mut self,
        source: impl Into<String>,
        credential: impl Into<String>,
        provider: Arc<P>,
    ) where
        P: InferenceProvider + 'static,
    {
        self.register_dyn_credential(source, credential, provider);
    }

    pub fn register_dyn(
        &mut self,
        source: impl Into<String>,
        provider: Arc<dyn InferenceProvider>,
    ) {
        self.providers
            .insert(ProviderInstanceKey::new(source, None), provider);
    }

    pub fn register_dyn_credential(
        &mut self,
        source: impl Into<String>,
        credential: impl Into<String>,
        provider: Arc<dyn InferenceProvider>,
    ) {
        self.providers.insert(
            ProviderInstanceKey::new(source, Some(credential.into())),
            provider,
        );
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
            for (instance, provider) in self
                .providers
                .iter()
                .filter(|(instance, _)| instance.source == route.source)
            {
                let Ok(mut discovered) = provider.endpoints(&route.id).await else {
                    continue;
                };
                for endpoint in &mut discovered {
                    // Catalog identity and registry credential are authoritative over adapter hints.
                    endpoint.source.clone_from(&route.source);
                    endpoint.credential.clone_from(&instance.credential);
                    endpoint.model.clone_from(&route.id);
                    endpoint.route_class = route.route_class;
                }
                endpoints.extend(discovered);
            }
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

    /// Build a route plan after quota filtering, before computing the request cost floor.
    ///
    /// # Errors
    /// Returns provider/routing errors normally, or a quota-specific failure carrying every
    /// shape-eligible route's admission report when quota removes all candidates.
    pub async fn route_with_quota(
        &self,
        catalog: &ModelCatalog,
        strategy: RoutingStrategy,
        request: &InferenceRequest,
        ledger: &QuotaLedger,
        unknown_policy: QuotaUnknownPolicy,
        now_unix: u64,
    ) -> Result<QuotaAwareRoutePlan, QuotaRoutingError> {
        let endpoints = self.endpoints(catalog, request).await?;
        let prompt_tokens = request_prompt_tokens(request);
        let demand = QuotaDemand::for_request(request);
        let mut quota_reports = Vec::new();
        let mut admitted = Vec::new();
        for endpoint in endpoints
            .into_iter()
            .filter(|endpoint| supports_request(endpoint, request, prompt_tokens))
        {
            let report =
                ledger.evaluate(&QuotaRouteKey::from_endpoint(&endpoint), demand, now_unix);
            let keep = match report.admission {
                QuotaAdmission::Admitted => true,
                QuotaAdmission::Indeterminate => unknown_policy == QuotaUnknownPolicy::Allow,
                QuotaAdmission::Rejected => false,
            };
            if keep {
                admitted.push(endpoint);
            }
            quota_reports.push(report);
        }
        if admitted.is_empty() {
            return Err(QuotaRoutingError::NoQuotaAdmissible {
                model: request.model.clone(),
                reports: Box::new(quota_reports),
            });
        }
        let plan = build_route_plan(strategy, request, admitted)?;
        Ok(QuotaAwareRoutePlan {
            plan,
            quota_reports,
        })
    }

    /// Execute a synchronous route with quota reservation and conservative post-call accounting.
    ///
    /// Known finite quota is reserved per candidate immediately before dispatch. Successful calls
    /// commit normalized actual token usage where available. Structured execution failures commit
    /// the reserved worst case; local/non-dispatch failures release it. Quota is re-evaluated before
    /// every failover candidate so shared buckets cannot be multiplied by sibling credentials.
    ///
    /// # Errors
    /// Returns quota-routing errors, ledger invariant failures, or an aggregate failure after all
    /// currently quota-admissible synchronous candidates fail.
    pub async fn infer_with_quota(
        &self,
        catalog: &ModelCatalog,
        strategy: RoutingStrategy,
        request: &InferenceRequest,
        retry: RetryPolicy,
        ledger: &mut QuotaLedger,
        quota: QuotaExecutionContext,
    ) -> Result<QuotaPooledInferenceResponse, QuotaInferenceError> {
        if request
            .requirements
            .allowed_route_classes
            .contains(&ModelRouteClass::Batch)
        {
            return Err(QuotaRoutingError::Provider(ProviderError::InvalidRequest(
                "synchronous infer cannot admit batch routes; use the batch API".to_owned(),
            ))
            .into());
        }
        let quota_plan = self
            .route_with_quota(
                catalog,
                strategy,
                request,
                ledger,
                quota.unknown_policy,
                quota.now_unix,
            )
            .await?;
        let demand = QuotaDemand::for_request(request);
        let mut attempted = 0_usize;
        let mut last_error = None;
        for candidate in quota_plan.plan.candidates.clone() {
            let route_key = QuotaRouteKey::from_endpoint(&candidate.endpoint);
            let Ok(reservation) = ledger.reserve_with_policy(
                &route_key,
                demand,
                quota.now_unix,
                quota.unknown_policy,
            ) else {
                continue;
            };
            let instance = ProviderInstanceKey::new(
                candidate.endpoint.source.clone(),
                candidate.endpoint.credential.clone(),
            );
            let Some(provider) = self.providers.get(&instance) else {
                if !ledger.release(reservation) {
                    return Err(QuotaInferenceError::LedgerInvariant(
                        "failed to release reservation for missing provider instance".to_owned(),
                    ));
                }
                continue;
            };
            attempted += 1;
            let mut native_request = request.clone();
            native_request.model.clone_from(&candidate.endpoint.model);
            let candidate_plan = RoutePlan {
                prompt_tokens: quota_plan.plan.prompt_tokens,
                expected_output_tokens: quota_plan.plan.expected_output_tokens,
                cheapest_cost_usd: candidate.expected_cost_usd,
                cost_ceiling_usd: candidate.expected_cost_usd,
                candidates: vec![candidate.clone()],
            };
            match provider
                .execute_plan(&native_request, &candidate_plan, retry)
                .await
            {
                Ok(response) => {
                    let actual = demand.with_actual_usage(&response);
                    if !ledger.commit(reservation, Some(actual), quota.now_unix) {
                        return Err(QuotaInferenceError::LedgerInvariant(
                            "failed to commit successful inference reservation".to_owned(),
                        ));
                    }
                    return Ok(QuotaPooledInferenceResponse {
                        quota_plan,
                        selected: candidate,
                        response,
                    });
                }
                Err(error) => {
                    let accounted = if error.execution_phase().is_some() {
                        ledger.commit(reservation, None, quota.now_unix)
                    } else {
                        ledger.release(reservation)
                    };
                    if !accounted {
                        return Err(QuotaInferenceError::LedgerInvariant(
                            "failed to settle failed inference reservation".to_owned(),
                        ));
                    }
                    last_error = Some(Box::new(error));
                }
            }
        }
        Err(QuotaInferenceError::AllCandidatesFailed {
            attempted,
            last_error,
        })
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
            let instance = ProviderInstanceKey::new(
                candidate.endpoint.source.clone(),
                candidate.endpoint.credential.clone(),
            );
            let Some(provider) = self.providers.get(&instance) else {
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
                credential: None,
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
            self.calls
                .lock()
                .map_err(|_| ProviderError::Catalog("poisoned test mutex".to_owned()))?
                .push(format!("execute:{}", request.model));
            if self.fail_execute {
                return Err(ProviderError::Execution {
                    phase: crate::ExecutionFailurePhase::TransportUnknown,
                    detail: "mock failure".to_owned(),
                });
            }
            Ok(InferenceResponse {
                content: "ok".to_owned(),
                model: Some(request.model.clone()),
                provider: Some(self.source.to_owned()),
                usage_cost_usd: None,
                usage: crate::TokenUsage {
                    input_tokens: Some(10),
                    output_tokens: Some(20),
                    total_tokens: Some(30),
                    reasoning_tokens: None,
                },
                raw: json!({"native_model": request.model, "mock_price": self.price}),
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
    async fn same_source_credentials_remain_distinct_through_discovery_and_execution()
    -> Result<(), ProviderError> {
        let expensive_calls = Arc::new(StdMutex::new(Vec::new()));
        let cheap_calls = Arc::new(StdMutex::new(Vec::new()));
        let mut pool = InferencePool::new();
        pool.register_credential(
            "nvidia",
            "account-expensive",
            Arc::new(MockProvider {
                source: "nvidia",
                price: 2e-6,
                fail_execute: false,
                calls: Arc::clone(&expensive_calls),
            }),
        );
        pool.register_credential(
            "nvidia",
            "account-cheap",
            Arc::new(MockProvider {
                source: "nvidia",
                price: 0.0,
                fail_execute: false,
                calls: Arc::clone(&cheap_calls),
            }),
        );
        let request = InferenceRequest::new("vendor/model", "hi", 8);
        let endpoints = pool.endpoints(&catalog(), &request).await?;
        let credentials: BTreeSet<_> = endpoints
            .iter()
            .filter_map(|endpoint| endpoint.credential.as_deref())
            .collect();
        assert_eq!(
            credentials,
            BTreeSet::from(["account-cheap", "account-expensive"])
        );

        let result = pool
            .infer(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
                RetryPolicy::default(),
            )
            .await?;
        assert_eq!(
            result.selected.endpoint.credential.as_deref(),
            Some("account-cheap")
        );
        assert_eq!(
            result
                .response
                .raw
                .get("mock_price")
                .and_then(serde_json::Value::as_f64),
            Some(0.0)
        );
        let expensive = expensive_calls
            .lock()
            .map_err(|_| ProviderError::Catalog("poisoned test mutex".to_owned()))?;
        // Discovery happened for both credentials, but the expensive credential never executed.
        assert_eq!(
            expensive.as_slice(),
            ["vendor/native-model", "vendor/native-model"]
        );
        let cheap = cheap_calls
            .lock()
            .map_err(|_| ProviderError::Catalog("poisoned test mutex".to_owned()))?;
        assert_eq!(
            cheap.as_slice(),
            [
                "vendor/native-model",
                "vendor/native-model",
                "execute:vendor/native-model"
            ]
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
    async fn quota_filter_removes_exhausted_cheapest_before_cost_floor()
    -> Result<(), QuotaRoutingError> {
        let mut pool = InferencePool::new();
        pool.register_credential(
            "nvidia",
            "cheap-exhausted",
            Arc::new(MockProvider {
                source: "nvidia",
                price: 0.0,
                fail_execute: false,
                calls: Arc::new(StdMutex::new(Vec::new())),
            }),
        );
        pool.register_credential(
            "nvidia",
            "expensive-healthy",
            Arc::new(MockProvider {
                source: "nvidia",
                price: 1e-6,
                fail_execute: false,
                calls: Arc::new(StdMutex::new(Vec::new())),
            }),
        );

        let cheap_route = QuotaRouteKey {
            source: "nvidia".to_owned(),
            credential: Some("cheap-exhausted".to_owned()),
            provider: "nvidia".to_owned(),
            model: "vendor/native-model".to_owned(),
        };
        let expensive_route = QuotaRouteKey {
            source: "nvidia".to_owned(),
            credential: Some("expensive-healthy".to_owned()),
            provider: "nvidia".to_owned(),
            model: "vendor/native-model".to_owned(),
        };
        let cheap_resource = crate::QuotaResource::finite(
            "cheap/rpd",
            crate::QuotaScope::Account,
            crate::QuotaMetric::Requests,
            1,
            0,
        );
        let cheap_id = cheap_resource.id.clone();
        let expensive_resource = crate::QuotaResource::finite(
            "expensive/rpd",
            crate::QuotaScope::Account,
            crate::QuotaMetric::Requests,
            100,
            100,
        );
        let expensive_id = expensive_resource.id.clone();
        let mut graph = crate::QuotaGraph::new();
        graph.insert_resource(cheap_resource);
        graph.insert_resource(expensive_resource);
        assert!(graph.bind(cheap_route, &cheap_id));
        assert!(graph.bind(expensive_route, &expensive_id));
        let ledger = QuotaLedger::new(graph);
        let request = InferenceRequest::new("vendor/model", "hi", 8);

        let result = pool
            .route_with_quota(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
                &ledger,
                QuotaUnknownPolicy::Reject,
                100,
            )
            .await?;
        assert_eq!(result.plan.candidates.len(), 1);
        assert_eq!(
            result.plan.candidates[0].endpoint.credential.as_deref(),
            Some("expensive-healthy")
        );
        assert!(result.plan.cheapest_cost_usd > 0.0);
        assert!(result.quota_reports.iter().any(|report| {
            report.route.credential.as_deref() == Some("cheap-exhausted")
                && report.admission == QuotaAdmission::Rejected
        }));
        Ok(())
    }

    #[tokio::test]
    async fn unknown_quota_policy_is_explicit() -> Result<(), QuotaRoutingError> {
        let mut pool = InferencePool::new();
        pool.register_credential(
            "nvidia",
            "unmeasured",
            Arc::new(MockProvider {
                source: "nvidia",
                price: 0.0,
                fail_execute: false,
                calls: Arc::new(StdMutex::new(Vec::new())),
            }),
        );
        let request = InferenceRequest::new("vendor/model", "hi", 8);
        let ledger = QuotaLedger::new(crate::QuotaGraph::new());
        let permissive = pool
            .route_with_quota(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
                &ledger,
                QuotaUnknownPolicy::Allow,
                100,
            )
            .await?;
        assert_eq!(permissive.plan.candidates.len(), 1);
        assert_eq!(
            permissive.quota_reports[0].admission,
            QuotaAdmission::Indeterminate
        );

        let strict = pool
            .route_with_quota(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
                &ledger,
                QuotaUnknownPolicy::Reject,
                100,
            )
            .await;
        match strict {
            Err(QuotaRoutingError::NoQuotaAdmissible { reports, .. }) => {
                assert_eq!(reports.len(), 1);
                assert_eq!(reports[0].admission, QuotaAdmission::Indeterminate);
            }
            other => {
                return Err(QuotaRoutingError::Provider(ProviderError::Response(
                    format!("unexpected strict quota result: {other:?}"),
                )));
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn quota_infer_commits_normalized_actual_usage() -> Result<(), QuotaInferenceError> {
        let mut pool = InferencePool::new();
        pool.register(
            "nvidia",
            Arc::new(MockProvider {
                source: "nvidia",
                price: 0.0,
                fail_execute: false,
                calls: Arc::new(StdMutex::new(Vec::new())),
            }),
        );
        let route = QuotaRouteKey {
            source: "nvidia".to_owned(),
            credential: None,
            provider: "nvidia".to_owned(),
            model: "vendor/native-model".to_owned(),
        };
        let resource = crate::QuotaResource::finite(
            "nvidia/tpd",
            crate::QuotaScope::Account,
            crate::QuotaMetric::TotalTokens,
            1_000,
            1_000,
        );
        let resource_id = resource.id.clone();
        let mut graph = crate::QuotaGraph::new();
        graph.insert_resource(resource);
        assert!(graph.bind(route, &resource_id));
        let mut ledger = QuotaLedger::new(graph);
        let mut request = InferenceRequest::new("vendor/model", "hi", 8);
        request.prompt_tokens = Some(100);
        request.max_output_tokens = Some(200);
        let result = pool
            .infer_with_quota(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
                RetryPolicy::default(),
                &mut ledger,
                QuotaExecutionContext {
                    unknown_policy: QuotaUnknownPolicy::Reject,
                    now_unix: 100,
                },
            )
            .await?;
        assert_eq!(result.response.usage.total_tokens, Some(30));
        assert_eq!(
            ledger.graph().resource(&resource_id).map(|r| r.remaining),
            Some(crate::QuotaAmount::Finite(970))
        );
        Ok(())
    }

    #[tokio::test]
    async fn ambiguous_failure_consumes_shared_request_reservation_before_failover() {
        let failing_calls = Arc::new(StdMutex::new(Vec::new()));
        let healthy_calls = Arc::new(StdMutex::new(Vec::new()));
        let mut pool = InferencePool::new();
        pool.register_credential(
            "nvidia",
            "a-failing",
            Arc::new(MockProvider {
                source: "nvidia",
                price: 0.0,
                fail_execute: true,
                calls: Arc::clone(&failing_calls),
            }),
        );
        pool.register_credential(
            "nvidia",
            "b-healthy",
            Arc::new(MockProvider {
                source: "nvidia",
                price: 0.0,
                fail_execute: false,
                calls: Arc::clone(&healthy_calls),
            }),
        );
        let shared = crate::QuotaResource::finite(
            "nvidia/account/rpd",
            crate::QuotaScope::Account,
            crate::QuotaMetric::Requests,
            1,
            1,
        );
        let shared_id = shared.id.clone();
        let mut graph = crate::QuotaGraph::new();
        graph.insert_resource(shared);
        for credential in ["a-failing", "b-healthy"] {
            let route = QuotaRouteKey {
                source: "nvidia".to_owned(),
                credential: Some(credential.to_owned()),
                provider: "nvidia".to_owned(),
                model: "vendor/native-model".to_owned(),
            };
            assert!(graph.bind(route, &shared_id));
        }
        let mut ledger = QuotaLedger::new(graph);
        let request = InferenceRequest::new("vendor/model", "hi", 8);
        let result = pool
            .infer_with_quota(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
                RetryPolicy::default(),
                &mut ledger,
                QuotaExecutionContext {
                    unknown_policy: QuotaUnknownPolicy::Reject,
                    now_unix: 100,
                },
            )
            .await;
        assert!(matches!(
            result,
            Err(QuotaInferenceError::AllCandidatesFailed { attempted: 1, .. })
        ));
        assert_eq!(
            ledger.graph().resource(&shared_id).map(|r| r.remaining),
            Some(crate::QuotaAmount::Finite(0))
        );
        let failing = failing_calls.lock();
        assert!(failing.is_ok());
        if let Ok(calls) = failing {
            assert!(calls.iter().any(|call| call.starts_with("execute:")));
        }
        let healthy = healthy_calls.lock();
        assert!(healthy.is_ok());
        if let Ok(calls) = healthy {
            assert!(!calls.iter().any(|call| call.starts_with("execute:")));
        }
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
