use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    BenchmarkCatalog, InferencePool, InferenceRequest, ModelCatalog, ModelSuitabilityReport,
    ModelSuitabilityRequirements, ProviderError, QuotaAdmissionReport, QuotaLedger, QuotaRouteKey,
    QuotaUnknownPolicy, RequestRequirements, RouteCandidate, RoutingStrategy, SuitabilityStatus,
    build_route_plan, request_prompt_tokens,
};

/// Request shape for work whose model has not been selected yet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceRequestTemplate {
    #[serde(default)]
    pub system_prompt: Option<String>,
    pub prompt: String,
    pub expected_output_tokens: u64,
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub agentic: bool,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub requirements: RequestRequirements,
}

impl InferenceRequestTemplate {
    #[must_use]
    pub fn for_model(&self, model: impl Into<String>) -> InferenceRequest {
        InferenceRequest {
            model: model.into(),
            system_prompt: self.system_prompt.clone(),
            prompt: self.prompt.clone(),
            expected_output_tokens: self.expected_output_tokens,
            prompt_tokens: self.prompt_tokens,
            max_output_tokens: self.max_output_tokens,
            temperature: self.temperature,
            agentic: self.agentic,
            session_id: self.session_id.clone(),
            requirements: self.requirements.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuitabilityUnknownPolicy {
    Allow,
    #[default]
    Reject,
}

/// Bounded live-discovery policy for cross-model selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSelectionPolicy {
    pub unknown_suitability: SuitabilityUnknownPolicy,
    /// Maximum canonical models whose live provider endpoints may be queried.
    pub discovery_limit: usize,
    pub quota_unknown_policy: QuotaUnknownPolicy,
    pub now_unix: u64,
}

/// Evidence inputs shared by one model-selection pass.
#[derive(Debug, Clone, Copy)]
pub struct ModelSelectionEvidence<'a> {
    pub catalog: &'a ModelCatalog,
    pub benchmarks: Option<&'a BenchmarkCatalog>,
    pub suitability: &'a ModelSuitabilityRequirements,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum ModelSelectionRejection {
    NoExecutableRoute {
        canonical_id: String,
        suitability: ModelSuitabilityReport,
    },
    DiscoveryUnavailable {
        canonical_id: String,
        suitability: ModelSuitabilityReport,
    },
    RequestShapeUnsupported {
        canonical_id: String,
        suitability: ModelSuitabilityReport,
    },
    QuotaUnavailable {
        canonical_id: String,
        suitability: ModelSuitabilityReport,
        reports: Vec<QuotaAdmissionReport>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRouteSelectionCandidate {
    pub canonical_id: String,
    pub suitability: ModelSuitabilityReport,
    pub quota: QuotaAdmissionReport,
    pub route: RouteCandidate,
}

/// Inspectable model + execution-route plan for deferred or standing work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSelectionPlan {
    pub considered_models: usize,
    pub probed_models: usize,
    pub deferred_models: usize,
    pub prompt_tokens: u64,
    pub expected_output_tokens: u64,
    pub cheapest_cost_usd: Option<f64>,
    pub cost_ceiling_usd: Option<f64>,
    pub candidates: Vec<ModelRouteSelectionCandidate>,
    pub rejections: Vec<ModelSelectionRejection>,
}

#[derive(Debug)]
struct EligibleModel {
    suitability: ModelSuitabilityReport,
    catalog_cost: Option<f64>,
}

#[derive(Debug, Default)]
struct SelectionDiscovery {
    endpoints: Vec<crate::Endpoint>,
    route_evidence: BTreeMap<QuotaRouteKey, (String, ModelSuitabilityReport, QuotaAdmissionReport)>,
    rejections: Vec<ModelSelectionRejection>,
}

fn suitability_admitted(status: SuitabilityStatus, policy: SuitabilityUnknownPolicy) -> bool {
    match status {
        SuitabilityStatus::Suitable => true,
        SuitabilityStatus::Unknown => policy == SuitabilityUnknownPolicy::Allow,
        SuitabilityStatus::Unsuitable => false,
    }
}

fn suitability_rank(status: SuitabilityStatus) -> u8 {
    match status {
        SuitabilityStatus::Suitable => 0,
        SuitabilityStatus::Unknown => 1,
        SuitabilityStatus::Unsuitable => 2,
    }
}

fn optional_cost_cmp(left: Option<f64>, right: Option<f64>) -> Ordering {
    match (left, right) {
        (Some(a), Some(b)) => a.total_cmp(&b),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn catalog_cost_floor(
    catalog: &ModelCatalog,
    canonical_id: &str,
    request: &InferenceRequest,
) -> Option<f64> {
    let prompt_tokens = request_prompt_tokens(request);
    catalog
        .models
        .iter()
        .filter(|entry| entry.canonical_id == canonical_id)
        .filter(|entry| {
            request
                .requirements
                .allowed_route_classes
                .contains(&entry.route_class)
        })
        .filter(|entry| entry.pricing_known)
        .map(|entry| {
            if entry.route_class == crate::ModelRouteClass::Subscription {
                0.0
            } else {
                entry
                    .pricing
                    .expected_cost(prompt_tokens, request.expected_output_tokens)
            }
        })
        .filter(|cost| cost.is_finite() && *cost >= 0.0)
        .min_by(f64::total_cmp)
}

fn has_allowed_catalog_route(
    catalog: &ModelCatalog,
    canonical_id: &str,
    request: &InferenceRequestTemplate,
) -> bool {
    catalog.models.iter().any(|entry| {
        entry.canonical_id == canonical_id
            && request
                .requirements
                .allowed_route_classes
                .contains(&entry.route_class)
    })
}

fn eligible_models(
    evidence: ModelSelectionEvidence<'_>,
    request_template: &InferenceRequestTemplate,
    policy: ModelSelectionPolicy,
) -> Vec<EligibleModel> {
    let reports = evidence.benchmarks.map_or_else(
        || evidence.catalog.suitability_reports(evidence.suitability),
        |benchmarks| {
            evidence
                .catalog
                .suitability_reports_with_benchmarks(evidence.suitability, benchmarks)
        },
    );
    let mut eligible = reports
        .into_iter()
        .filter(|report| suitability_admitted(report.status, policy.unknown_suitability))
        .filter(|report| {
            has_allowed_catalog_route(evidence.catalog, &report.canonical_id, request_template)
        })
        .map(|suitability| {
            let request = request_template.for_model(suitability.canonical_id.clone());
            EligibleModel {
                catalog_cost: catalog_cost_floor(
                    evidence.catalog,
                    &suitability.canonical_id,
                    &request,
                ),
                suitability,
            }
        })
        .collect::<Vec<_>>();
    eligible.sort_by(|left, right| {
        suitability_rank(left.suitability.status)
            .cmp(&suitability_rank(right.suitability.status))
            .then_with(|| optional_cost_cmp(left.catalog_cost, right.catalog_cost))
            .then_with(|| {
                left.suitability
                    .canonical_id
                    .cmp(&right.suitability.canonical_id)
            })
    });
    eligible
}

fn empty_plan(
    request: &InferenceRequest,
    considered_models: usize,
    probed_models: usize,
    rejections: Vec<ModelSelectionRejection>,
) -> ModelSelectionPlan {
    ModelSelectionPlan {
        considered_models,
        probed_models,
        deferred_models: considered_models.saturating_sub(probed_models),
        prompt_tokens: request_prompt_tokens(request),
        expected_output_tokens: request.expected_output_tokens,
        cheapest_cost_usd: None,
        cost_ceiling_usd: None,
        candidates: Vec::new(),
        rejections,
    }
}

fn finish_plan(
    strategy: RoutingStrategy,
    request: &InferenceRequest,
    considered_models: usize,
    probed_models: usize,
    discovery: SelectionDiscovery,
) -> Result<ModelSelectionPlan, ProviderError> {
    if discovery.endpoints.is_empty() {
        return Ok(empty_plan(
            request,
            considered_models,
            probed_models,
            discovery.rejections,
        ));
    }
    let route_plan = build_route_plan(strategy, request, discovery.endpoints)?;
    let mut candidates = Vec::with_capacity(route_plan.candidates.len());
    for route in route_plan.candidates {
        let route_key = QuotaRouteKey::from_endpoint(&route.endpoint);
        let Some((canonical_id, suitability, quota)) = discovery.route_evidence.get(&route_key)
        else {
            return Err(ProviderError::Response(
                "model selection lost route evidence after ranking".to_owned(),
            ));
        };
        candidates.push(ModelRouteSelectionCandidate {
            canonical_id: canonical_id.clone(),
            suitability: suitability.clone(),
            quota: quota.clone(),
            route,
        });
    }
    Ok(ModelSelectionPlan {
        considered_models,
        probed_models,
        deferred_models: considered_models.saturating_sub(probed_models),
        prompt_tokens: route_plan.prompt_tokens,
        expected_output_tokens: route_plan.expected_output_tokens,
        cheapest_cost_usd: Some(route_plan.cheapest_cost_usd),
        cost_ceiling_usd: Some(route_plan.cost_ceiling_usd),
        candidates,
        rejections: discovery.rejections,
    })
}

impl InferencePool {
    async fn discover_selection_models(
        &self,
        evidence: ModelSelectionEvidence<'_>,
        request_template: &InferenceRequestTemplate,
        models: impl Iterator<Item = EligibleModel>,
        ledger: &mut QuotaLedger,
        policy: ModelSelectionPolicy,
    ) -> SelectionDiscovery {
        let mut discovery = SelectionDiscovery::default();
        for model in models {
            let canonical_id = model.suitability.canonical_id.clone();
            let request = request_template.for_model(canonical_id.clone());
            let result = match self
                .quota_admitted_endpoints(
                    evidence.catalog,
                    &request,
                    ledger,
                    policy.quota_unknown_policy,
                    policy.now_unix,
                )
                .await
            {
                Ok(result) => result,
                Err(ProviderError::NoEligibleEndpoint(_)) => {
                    discovery
                        .rejections
                        .push(ModelSelectionRejection::NoExecutableRoute {
                            canonical_id,
                            suitability: model.suitability,
                        });
                    continue;
                }
                Err(_) => {
                    discovery
                        .rejections
                        .push(ModelSelectionRejection::DiscoveryUnavailable {
                            canonical_id,
                            suitability: model.suitability,
                        });
                    continue;
                }
            };
            if result.endpoints.is_empty() {
                let rejection = if result.reports.is_empty() {
                    ModelSelectionRejection::RequestShapeUnsupported {
                        canonical_id,
                        suitability: model.suitability,
                    }
                } else {
                    ModelSelectionRejection::QuotaUnavailable {
                        canonical_id,
                        suitability: model.suitability,
                        reports: result.reports,
                    }
                };
                discovery.rejections.push(rejection);
                continue;
            }
            for report in &result.reports {
                discovery.route_evidence.insert(
                    report.route.clone(),
                    (
                        canonical_id.clone(),
                        model.suitability.clone(),
                        report.clone(),
                    ),
                );
            }
            discovery.endpoints.extend(result.endpoints);
        }
        discovery
    }

    /// Select a canonical model and concrete execution route from suitability, live provider,
    /// conformance, quota, and request-economics evidence.
    ///
    /// Catalog pricing is used only to bound endpoint discovery fan-out. Final route ordering is
    /// recomputed globally from live endpoints after quota filtering.
    ///
    /// # Errors
    /// Returns an error for an invalid strategy, zero discovery limit, or malformed request shape.
    pub async fn select_with_quota(
        &self,
        evidence: ModelSelectionEvidence<'_>,
        request_template: &InferenceRequestTemplate,
        strategy: RoutingStrategy,
        policy: ModelSelectionPolicy,
        ledger: &mut QuotaLedger,
    ) -> Result<ModelSelectionPlan, ProviderError> {
        crate::router::validate_strategy(strategy)?;
        if policy.discovery_limit == 0 {
            return Err(ProviderError::InvalidRequest(
                "model selection discovery_limit must be at least 1".to_owned(),
            ));
        }
        let request = request_template.for_model("selection/template");
        if request
            .max_output_tokens
            .is_some_and(|maximum| maximum < request.expected_output_tokens)
        {
            return Err(ProviderError::InvalidRequest(
                "max_output_tokens is smaller than expected_output_tokens".to_owned(),
            ));
        }
        let eligible = eligible_models(evidence, request_template, policy);
        let considered_models = eligible.len();
        let probed_models = considered_models.min(policy.discovery_limit);
        let discovery = self
            .discover_selection_models(
                evidence,
                request_template,
                eligible.into_iter().take(probed_models),
                ledger,
                policy,
            )
            .await;
        finish_plan(
            strategy,
            &request,
            considered_models,
            probed_models,
            discovery,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use async_trait::async_trait;
    use serde_json::json;

    use super::*;
    use crate::{
        ArtificialAnalysisBenchmarks, Endpoint, EndpointCapabilities, EndpointStats,
        InferenceProvider, InferenceResponse, ModelArchitecture, ModelBenchmarks,
        ModelCatalogEntry, ModelReasoning, ModelRouteClass, ModelTopProvider, Pricing,
        ReasoningSupport, RetryPolicy, RoutePlan, TokenUsage,
    };

    struct MockProvider {
        source: &'static str,
        price: f64,
    }

    struct FailingDiscoveryProvider;

    #[async_trait]
    impl InferenceProvider for FailingDiscoveryProvider {
        async fn endpoints(&self, _model: &str) -> Result<Vec<Endpoint>, ProviderError> {
            Err(ProviderError::Catalog(
                "temporary catalog failure".to_owned(),
            ))
        }

        async fn execute_plan(
            &self,
            _request: &InferenceRequest,
            _plan: &RoutePlan,
            _retry: RetryPolicy,
        ) -> Result<InferenceResponse, ProviderError> {
            Err(ProviderError::Request("not reached".to_owned()))
        }
    }

    #[async_trait]
    impl InferenceProvider for MockProvider {
        async fn endpoints(&self, model: &str) -> Result<Vec<Endpoint>, ProviderError> {
            Ok(vec![Endpoint {
                source: self.source.to_owned(),
                credential: None,
                provider_key: self.source.to_owned(),
                id: self.source.to_owned(),
                provider: self.source.to_owned(),
                model: model.to_owned(),
                route_class: ModelRouteClass::Standard,
                display_name: model.to_owned(),
                quantization: None,
                status: 0,
                context_length: Some(100_000),
                max_prompt_tokens: None,
                max_completion_tokens: Some(10_000),
                pricing: Pricing {
                    prompt_per_token: self.price,
                    completion_per_token: self.price,
                    ..Pricing::default()
                },
                stats: EndpointStats::default(),
                workload_policy: crate::WorkloadPolicy::default(),
                capabilities: EndpointCapabilities {
                    adapter_parameters: BTreeSet::from([
                        "max_tokens".to_owned(),
                        "temperature".to_owned(),
                    ]),
                    supported_parameters: BTreeSet::from([
                        "max_tokens".to_owned(),
                        "temperature".to_owned(),
                    ]),
                    reasoning: ReasoningSupport::None,
                    ..EndpointCapabilities::default()
                },
            }])
        }

        async fn execute_plan(
            &self,
            request: &InferenceRequest,
            _plan: &RoutePlan,
            _retry: RetryPolicy,
        ) -> Result<InferenceResponse, ProviderError> {
            Ok(InferenceResponse {
                content: "ok".to_owned(),
                model: Some(request.model.clone()),
                provider: Some(self.source.to_owned()),
                usage_cost_usd: None,
                usage: TokenUsage::default(),
                quota_observations: Vec::new(),
                raw: json!({}),
            })
        }
    }

    fn entry(
        source: &str,
        id: &str,
        canonical_id: &str,
        class: ModelRouteClass,
        coding_index: Option<f64>,
        price: f64,
    ) -> ModelCatalogEntry {
        ModelCatalogEntry {
            id: id.to_owned(),
            route_class: class,
            canonical_id: canonical_id.to_owned(),
            name: id.to_owned(),
            description: None,
            aliases: BTreeSet::from([id.to_owned(), canonical_id.to_owned()]),
            created_unix: None,
            context_length: Some(100_000),
            architecture: ModelArchitecture::default(),
            pricing: Pricing {
                prompt_per_token: price,
                completion_per_token: price,
                ..Pricing::default()
            },
            pricing_known: true,
            top_provider: ModelTopProvider::default(),
            supported_parameters: BTreeSet::new(),
            reasoning: ModelReasoning::default(),
            benchmarks: ModelBenchmarks {
                artificial_analysis: coding_index.map(|coding_index| {
                    ArtificialAnalysisBenchmarks {
                        coding_index: Some(coding_index),
                        ..ArtificialAnalysisBenchmarks::default()
                    }
                }),
                ..ModelBenchmarks::default()
            },
            knowledge_cutoff: None,
            expiration_date: None,
            source: source.to_owned(),
        }
    }

    fn request_template() -> InferenceRequestTemplate {
        InferenceRequestTemplate {
            system_prompt: None,
            prompt: "solve".to_owned(),
            expected_output_tokens: 10,
            prompt_tokens: Some(10),
            max_output_tokens: Some(20),
            temperature: None,
            agentic: false,
            session_id: None,
            requirements: RequestRequirements {
                minimum_uptime_30m: None,
                ..RequestRequirements::default()
            },
        }
    }

    fn selection(discovery_limit: usize) -> ModelSelectionPolicy {
        ModelSelectionPolicy {
            unknown_suitability: SuitabilityUnknownPolicy::Reject,
            discovery_limit,
            quota_unknown_policy: QuotaUnknownPolicy::Allow,
            now_unix: 100,
        }
    }

    fn evidence<'a>(
        catalog: &'a ModelCatalog,
        suitability: &'a ModelSuitabilityRequirements,
    ) -> ModelSelectionEvidence<'a> {
        ModelSelectionEvidence {
            catalog,
            benchmarks: None,
            suitability,
        }
    }

    #[tokio::test]
    async fn global_cost_band_is_applied_after_cross_model_discovery() -> Result<(), ProviderError>
    {
        let catalog = ModelCatalog {
            models: vec![
                entry(
                    "free",
                    "vendor/free",
                    "vendor/free",
                    ModelRouteClass::Free,
                    Some(80.0),
                    0.0,
                ),
                entry(
                    "paid",
                    "vendor/paid",
                    "vendor/paid",
                    ModelRouteClass::Standard,
                    Some(90.0),
                    1e-6,
                ),
            ],
        };
        let suitability = ModelSuitabilityRequirements {
            minimum_coding_index: Some(50.0),
            ..ModelSuitabilityRequirements::default()
        };
        let mut pool = InferencePool::new();
        pool.register(
            "free",
            Arc::new(MockProvider {
                source: "free",
                price: 0.0,
            }),
        );
        pool.register(
            "paid",
            Arc::new(MockProvider {
                source: "paid",
                price: 1e-6,
            }),
        );
        let mut ledger = QuotaLedger::new(crate::QuotaGraph::new());
        let plan = pool
            .select_with_quota(
                evidence(&catalog, &suitability),
                &request_template(),
                RoutingStrategy::FastestResponseCheap(1.5),
                selection(2),
                &mut ledger,
            )
            .await?;
        assert_eq!(plan.considered_models, 2);
        assert_eq!(plan.probed_models, 2);
        assert_eq!(plan.candidates.len(), 1);
        assert_eq!(plan.candidates[0].canonical_id, "vendor/free");
        assert_eq!(plan.cheapest_cost_usd, Some(0.0));
        Ok(())
    }

    #[tokio::test]
    async fn discovery_limit_uses_catalog_cost_only_as_fanout_bound() -> Result<(), ProviderError> {
        let catalog = ModelCatalog {
            models: vec![
                entry(
                    "free",
                    "vendor/free",
                    "vendor/free",
                    ModelRouteClass::Free,
                    Some(80.0),
                    0.0,
                ),
                entry(
                    "paid",
                    "vendor/paid",
                    "vendor/paid",
                    ModelRouteClass::Standard,
                    Some(90.0),
                    1e-6,
                ),
            ],
        };
        let suitability = ModelSuitabilityRequirements {
            minimum_coding_index: Some(50.0),
            ..ModelSuitabilityRequirements::default()
        };
        let mut pool = InferencePool::new();
        pool.register(
            "free",
            Arc::new(MockProvider {
                source: "free",
                price: 0.0,
            }),
        );
        pool.register(
            "paid",
            Arc::new(MockProvider {
                source: "paid",
                price: 1e-6,
            }),
        );
        let mut ledger = QuotaLedger::new(crate::QuotaGraph::new());
        let plan = pool
            .select_with_quota(
                evidence(&catalog, &suitability),
                &request_template(),
                RoutingStrategy::FastestResponseCheap(1.5),
                selection(1),
                &mut ledger,
            )
            .await?;
        assert_eq!(plan.considered_models, 2);
        assert_eq!(plan.probed_models, 1);
        assert_eq!(plan.deferred_models, 1);
        assert_eq!(plan.candidates[0].canonical_id, "vendor/free");
        Ok(())
    }

    #[tokio::test]
    async fn models_without_requested_route_class_do_not_spend_discovery_budget()
    -> Result<(), ProviderError> {
        let catalog = ModelCatalog {
            models: vec![
                entry(
                    "free",
                    "vendor/free",
                    "vendor/free",
                    ModelRouteClass::Free,
                    Some(90.0),
                    0.0,
                ),
                entry(
                    "paid",
                    "vendor/paid",
                    "vendor/paid",
                    ModelRouteClass::Standard,
                    Some(80.0),
                    1e-6,
                ),
            ],
        };
        let suitability = ModelSuitabilityRequirements {
            minimum_coding_index: Some(50.0),
            ..ModelSuitabilityRequirements::default()
        };
        let mut template = request_template();
        template.requirements.allowed_route_classes = BTreeSet::from([ModelRouteClass::Standard]);
        let mut pool = InferencePool::new();
        pool.register(
            "free",
            Arc::new(MockProvider {
                source: "free",
                price: 0.0,
            }),
        );
        pool.register(
            "paid",
            Arc::new(MockProvider {
                source: "paid",
                price: 1e-6,
            }),
        );
        let mut ledger = QuotaLedger::new(crate::QuotaGraph::new());
        let plan = pool
            .select_with_quota(
                evidence(&catalog, &suitability),
                &template,
                RoutingStrategy::FastestResponseCheap(1.5),
                selection(1),
                &mut ledger,
            )
            .await?;
        assert_eq!(plan.considered_models, 1);
        assert_eq!(plan.probed_models, 1);
        assert_eq!(plan.candidates.len(), 1);
        assert_eq!(plan.candidates[0].canonical_id, "vendor/paid");
        Ok(())
    }

    #[tokio::test]
    async fn transient_discovery_failure_is_not_reported_as_no_route() -> Result<(), ProviderError>
    {
        let catalog = ModelCatalog {
            models: vec![entry(
                "flaky",
                "vendor/flaky",
                "vendor/flaky",
                ModelRouteClass::Standard,
                Some(90.0),
                1e-6,
            )],
        };
        let suitability = ModelSuitabilityRequirements {
            minimum_coding_index: Some(50.0),
            ..ModelSuitabilityRequirements::default()
        };
        let mut pool = InferencePool::new();
        pool.register("flaky", Arc::new(FailingDiscoveryProvider));
        let mut ledger = QuotaLedger::new(crate::QuotaGraph::new());
        let plan = pool
            .select_with_quota(
                evidence(&catalog, &suitability),
                &request_template(),
                RoutingStrategy::FastestResponseCheap(1.5),
                selection(1),
                &mut ledger,
            )
            .await?;
        assert_eq!(plan.rejections.len(), 1);
        assert!(matches!(
            plan.rejections[0],
            ModelSelectionRejection::DiscoveryUnavailable { .. }
        ));
        Ok(())
    }

    #[tokio::test]
    async fn unknown_suitability_is_rejected_by_default_policy() -> Result<(), ProviderError> {
        let catalog = ModelCatalog {
            models: vec![entry(
                "free",
                "vendor/free",
                "vendor/free",
                ModelRouteClass::Free,
                None,
                0.0,
            )],
        };
        let suitability = ModelSuitabilityRequirements {
            minimum_coding_index: Some(50.0),
            ..ModelSuitabilityRequirements::default()
        };
        let mut pool = InferencePool::new();
        pool.register(
            "free",
            Arc::new(MockProvider {
                source: "free",
                price: 0.0,
            }),
        );
        let mut ledger = QuotaLedger::new(crate::QuotaGraph::new());
        let plan = pool
            .select_with_quota(
                evidence(&catalog, &suitability),
                &request_template(),
                RoutingStrategy::FastestResponseCheap(1.5),
                selection(1),
                &mut ledger,
            )
            .await?;
        assert_eq!(plan.considered_models, 0);
        assert_eq!(plan.candidates.len(), 0);
        Ok(())
    }
}
