use std::cmp::Ordering;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    InferencePool, InferenceRequest, ModelCatalog, ModelRouteClass, ModelSuitabilityReport,
    ModelSuitabilityRequirements, QuotaAdmissionReport, QuotaUnknownPolicy, RequestRequirements,
    RouteCandidate, RoutingStrategy, SharedQuotaLedger, SuitabilityStatus, request_prompt_tokens,
};

const fn default_max_model_probes() -> usize {
    16
}

const fn default_cost_multiplier() -> f64 {
    1.5
}

/// Provider-independent inference payload for a standing task. Model selection is intentionally
/// absent: the planner chooses a canonical model only after suitability and economics filtering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StandingWorkload {
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
    pub requirements: RequestRequirements,
}

impl StandingWorkload {
    #[must_use]
    pub fn request_for_model(&self, model: impl Into<String>) -> InferenceRequest {
        InferenceRequest {
            model: model.into(),
            system_prompt: self.system_prompt.clone(),
            prompt: self.prompt.clone(),
            expected_output_tokens: self.expected_output_tokens,
            prompt_tokens: self.prompt_tokens,
            max_output_tokens: self.max_output_tokens,
            temperature: self.temperature,
            requirements: self.requirements.clone(),
        }
    }
}

/// Policy for opportunistic standing work. Suitability is a minimum-quality gate; cost and live
/// quota decide whether a currently usable execution opportunity exists.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StandingTaskPolicy {
    #[serde(default)]
    pub suitability: ModelSuitabilityRequirements,
    /// Missing suitability evidence is a fallback only when explicitly enabled.
    #[serde(default)]
    pub allow_unknown_suitability: bool,
    /// Bound live provider discovery after cheap catalog-level filtering.
    #[serde(default = "default_max_model_probes")]
    pub max_model_probes: usize,
    /// Hard per-request expected cost ceiling. `None` means no absolute ceiling.
    #[serde(default)]
    pub max_expected_cost_usd: Option<f64>,
    /// Existing router policy applied within each canonical model.
    #[serde(default = "default_cost_multiplier")]
    pub cheapest_cost_multiplier: f64,
    /// Unknown quota may remain eligible while known exhaustion always rejects a route.
    #[serde(default)]
    pub quota_unknown_policy: QuotaUnknownPolicy,
}

impl Default for StandingTaskPolicy {
    fn default() -> Self {
        Self {
            suitability: ModelSuitabilityRequirements::default(),
            allow_unknown_suitability: false,
            max_model_probes: default_max_model_probes(),
            max_expected_cost_usd: None,
            cheapest_cost_multiplier: default_cost_multiplier(),
            quota_unknown_policy: QuotaUnknownPolicy::Allow,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StandingInferenceTask {
    pub id: String,
    pub workload: StandingWorkload,
    #[serde(default)]
    pub policy: StandingTaskPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StandingPlanStatus {
    Ready,
    Waiting,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StandingOpportunity {
    pub canonical_id: String,
    pub suitability_status: SuitabilityStatus,
    pub catalog_expected_cost_usd: f64,
    pub route: RouteCandidate,
    pub quota_reports: Vec<QuotaAdmissionReport>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StandingRouteFailure {
    pub canonical_id: String,
    pub error: String,
}

/// Explainable snapshot of whether a standing task can run now.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StandingTaskPlan {
    pub task_id: String,
    pub status: StandingPlanStatus,
    pub selected: Option<StandingOpportunity>,
    pub opportunities: Vec<StandingOpportunity>,
    pub considered_canonical_models: usize,
    pub suitability_eligible_models: usize,
    pub catalog_priced_models: usize,
    pub catalog_over_budget_models: usize,
    pub probed_models: usize,
    pub live_over_budget_routes: usize,
    pub route_failures: Vec<StandingRouteFailure>,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum StandingTaskError {
    #[error("max_model_probes must be at least 1")]
    ZeroModelProbeBudget,
    #[error("cheapest_cost_multiplier must be finite and >= 1.0, got {0}")]
    InvalidCostMultiplier(f64),
    #[error("max_expected_cost_usd must be finite and >= 0.0, got {0}")]
    InvalidCostCeiling(f64),
}

#[derive(Debug, Clone)]
struct CatalogCandidate {
    canonical_id: String,
    suitability_status: SuitabilityStatus,
    catalog_expected_cost_usd: f64,
}

impl InferencePool {
    /// Plan one standing task against current catalog, conformance and shared quota state.
    ///
    /// Cheap catalog metadata bounds provider discovery first. Provider failures are returned as
    /// waiting diagnostics rather than hard errors so a long-lived task can simply be reconsidered
    /// when conditions change.
    ///
    /// # Errors
    /// Returns only malformed task-policy errors; ordinary lack of suitable/live capacity is a
    /// successful `Waiting` plan.
    pub async fn plan_standing_task(
        &self,
        catalog: &ModelCatalog,
        task: &StandingInferenceTask,
        ledger: &SharedQuotaLedger,
        now_unix: u64,
    ) -> Result<StandingTaskPlan, StandingTaskError> {
        validate_policy(&task.policy)?;
        let reports = catalog.suitability_reports(&task.policy.suitability);
        let considered_canonical_models = reports.len();
        let accepted: Vec<_> = reports
            .into_iter()
            .filter(|report| suitability_allowed(report, &task.policy))
            .collect();
        let suitability_eligible_models = accepted.len();

        let mut catalog_over_budget_models = 0_usize;
        let mut candidates = Vec::new();
        for report in accepted {
            let request = task.workload.request_for_model(&report.canonical_id);
            let Some(cost) = catalog_minimum_known_cost(catalog, &request, &report.canonical_id)
            else {
                continue;
            };
            if task
                .policy
                .max_expected_cost_usd
                .is_some_and(|maximum| cost > maximum + f64::EPSILON)
            {
                catalog_over_budget_models += 1;
                continue;
            }
            candidates.push(CatalogCandidate {
                canonical_id: report.canonical_id,
                suitability_status: report.status,
                catalog_expected_cost_usd: cost,
            });
        }
        let catalog_priced_models = candidates.len();
        candidates.sort_by(catalog_candidate_cmp);
        candidates.truncate(task.policy.max_model_probes);
        let probed_models = candidates.len();

        let mut opportunities = Vec::new();
        let mut route_failures = Vec::new();
        let mut live_over_budget_routes = 0_usize;
        for candidate in candidates {
            let request = task.workload.request_for_model(&candidate.canonical_id);
            match self
                .route_with_shared_quota(
                    catalog,
                    RoutingStrategy::FastestResponseCheap(task.policy.cheapest_cost_multiplier),
                    &request,
                    ledger,
                    task.policy.quota_unknown_policy,
                    now_unix,
                )
                .await
            {
                Ok(quota_plan) => {
                    for route in quota_plan.plan.candidates {
                        if task
                            .policy
                            .max_expected_cost_usd
                            .is_some_and(|maximum| route.expected_cost_usd > maximum + f64::EPSILON)
                        {
                            live_over_budget_routes += 1;
                            continue;
                        }
                        opportunities.push(StandingOpportunity {
                            canonical_id: candidate.canonical_id.clone(),
                            suitability_status: candidate.suitability_status,
                            catalog_expected_cost_usd: candidate.catalog_expected_cost_usd,
                            route,
                            quota_reports: quota_plan.quota_reports.clone(),
                        });
                    }
                }
                Err(error) => route_failures.push(StandingRouteFailure {
                    canonical_id: candidate.canonical_id,
                    error: error.to_string(),
                }),
            }
        }
        opportunities.sort_by(opportunity_cmp);
        let selected = opportunities.first().cloned();
        let status = if selected.is_some() {
            StandingPlanStatus::Ready
        } else {
            StandingPlanStatus::Waiting
        };
        Ok(StandingTaskPlan {
            task_id: task.id.clone(),
            status,
            selected,
            opportunities,
            considered_canonical_models,
            suitability_eligible_models,
            catalog_priced_models,
            catalog_over_budget_models,
            probed_models,
            live_over_budget_routes,
            route_failures,
        })
    }
}

fn validate_policy(policy: &StandingTaskPolicy) -> Result<(), StandingTaskError> {
    if policy.max_model_probes == 0 {
        return Err(StandingTaskError::ZeroModelProbeBudget);
    }
    if !policy.cheapest_cost_multiplier.is_finite() || policy.cheapest_cost_multiplier < 1.0 {
        return Err(StandingTaskError::InvalidCostMultiplier(
            policy.cheapest_cost_multiplier,
        ));
    }
    if let Some(maximum) = policy.max_expected_cost_usd
        && (!maximum.is_finite() || maximum < 0.0)
    {
        return Err(StandingTaskError::InvalidCostCeiling(maximum));
    }
    Ok(())
}

fn suitability_allowed(report: &ModelSuitabilityReport, policy: &StandingTaskPolicy) -> bool {
    report.status == SuitabilityStatus::Suitable
        || (policy.allow_unknown_suitability && report.status == SuitabilityStatus::Unknown)
}

fn catalog_minimum_known_cost(
    catalog: &ModelCatalog,
    request: &InferenceRequest,
    canonical_id: &str,
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
            entry
                .pricing
                .expected_cost(prompt_tokens, request.expected_output_tokens)
        })
        .filter(|cost| cost.is_finite() && *cost >= 0.0)
        .min_by(f64::total_cmp)
}

fn suitability_rank(status: SuitabilityStatus) -> u8 {
    match status {
        SuitabilityStatus::Suitable => 0,
        SuitabilityStatus::Unknown => 1,
        SuitabilityStatus::Unsuitable => 2,
    }
}

fn route_class_rank(class: ModelRouteClass) -> u8 {
    match class {
        ModelRouteClass::Free => 0,
        ModelRouteClass::Batch => 1,
        ModelRouteClass::Standard => 2,
    }
}

fn optional_f64_cmp(left: Option<f64>, right: Option<f64>) -> Ordering {
    match (left, right) {
        (Some(a), Some(b)) => a.total_cmp(&b),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn catalog_candidate_cmp(left: &CatalogCandidate, right: &CatalogCandidate) -> Ordering {
    suitability_rank(left.suitability_status)
        .cmp(&suitability_rank(right.suitability_status))
        .then_with(|| {
            left.catalog_expected_cost_usd
                .total_cmp(&right.catalog_expected_cost_usd)
        })
        .then_with(|| left.canonical_id.cmp(&right.canonical_id))
}

fn opportunity_cmp(left: &StandingOpportunity, right: &StandingOpportunity) -> Ordering {
    suitability_rank(left.suitability_status)
        .cmp(&suitability_rank(right.suitability_status))
        .then_with(|| {
            left.route
                .expected_cost_usd
                .total_cmp(&right.route.expected_cost_usd)
        })
        .then_with(|| {
            route_class_rank(left.route.endpoint.route_class)
                .cmp(&route_class_rank(right.route.endpoint.route_class))
        })
        .then_with(|| {
            optional_f64_cmp(
                left.route.expected_response_p75_seconds,
                right.route.expected_response_p75_seconds,
            )
        })
        .then_with(|| left.canonical_id.cmp(&right.canonical_id))
        .then_with(|| left.route.endpoint.id.cmp(&right.route.endpoint.id))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use serde_json::json;

    use super::*;
    use crate::{
        ArtificialAnalysisBenchmarks, Endpoint, EndpointCapabilities, EndpointStats,
        InferenceProvider, InferenceResponse, ModelArchitecture, ModelBenchmarks,
        ModelCatalogEntry, ModelReasoning, ModelTopProvider, Pricing, ProviderError, RetryPolicy,
        RoutePlan, TokenUsage,
    };

    #[derive(Debug)]
    struct MockProvider {
        calls: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl InferenceProvider for MockProvider {
        async fn endpoints(&self, model: &str) -> Result<Vec<Endpoint>, ProviderError> {
            self.calls
                .lock()
                .map_err(|_| ProviderError::Catalog("poisoned mock mutex".to_owned()))?
                .push(model.to_owned());
            Ok(vec![Endpoint {
                source: "mock".to_owned(),
                credential: None,
                provider_key: "mock".to_owned(),
                id: model.to_owned(),
                provider: "Mock".to_owned(),
                model: model.to_owned(),
                route_class: ModelRouteClass::Free,
                display_name: model.to_owned(),
                quantization: None,
                status: 0,
                context_length: Some(1_000_000),
                max_prompt_tokens: None,
                max_completion_tokens: Some(100_000),
                pricing: Pricing::default(),
                stats: EndpointStats {
                    uptime_30m: Some(100.0),
                    ..EndpointStats::default()
                },
                capabilities: EndpointCapabilities {
                    adapter_parameters: BTreeSet::from(["max_tokens".to_owned()]),
                    supported_parameters: BTreeSet::from(["max_tokens".to_owned()]),
                    ..EndpointCapabilities::default()
                },
            }])
        }

        async fn execute_plan(
            &self,
            _request: &InferenceRequest,
            _plan: &RoutePlan,
            _retry: RetryPolicy,
        ) -> Result<InferenceResponse, ProviderError> {
            Ok(InferenceResponse {
                content: "ok".to_owned(),
                model: None,
                provider: Some("mock".to_owned()),
                usage_cost_usd: Some(0.0),
                usage: TokenUsage::default(),
                quota_observations: Vec::new(),
                raw: json!({}),
            })
        }
    }

    fn catalog_entry(
        id: &str,
        canonical_id: &str,
        coding: Option<f64>,
        price: f64,
    ) -> ModelCatalogEntry {
        ModelCatalogEntry {
            id: id.to_owned(),
            route_class: ModelRouteClass::Free,
            canonical_id: canonical_id.to_owned(),
            name: id.to_owned(),
            description: None,
            aliases: BTreeSet::from([id.to_owned(), canonical_id.to_owned()]),
            created_unix: None,
            context_length: Some(1_000_000),
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
                artificial_analysis: Some(ArtificialAnalysisBenchmarks {
                    intelligence_index: None,
                    coding_index: coding,
                    agentic_index: None,
                }),
                design_arena: Vec::new(),
            },
            knowledge_cutoff: None,
            expiration_date: None,
            source: "mock".to_owned(),
        }
    }

    fn task(minimum_coding: Option<f64>) -> StandingInferenceTask {
        let suitability = ModelSuitabilityRequirements {
            minimum_coding_index: minimum_coding,
            ..ModelSuitabilityRequirements::default()
        };
        StandingInferenceTask {
            id: "task".to_owned(),
            workload: StandingWorkload {
                system_prompt: None,
                prompt: "hello".to_owned(),
                expected_output_tokens: 8,
                prompt_tokens: Some(2),
                max_output_tokens: Some(16),
                temperature: None,
                requirements: RequestRequirements {
                    allowed_route_classes: BTreeSet::from([ModelRouteClass::Free]),
                    ..RequestRequirements::default()
                },
            },
            policy: StandingTaskPolicy {
                suitability,
                ..StandingTaskPolicy::default()
            },
        }
    }

    fn pool(calls: Arc<Mutex<Vec<String>>>) -> InferencePool {
        let mut pool = InferencePool::new();
        pool.register("mock", Arc::new(MockProvider { calls }));
        pool
    }

    #[tokio::test]
    async fn suitable_free_model_is_ready() -> Result<(), StandingTaskError> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let catalog = ModelCatalog {
            models: vec![catalog_entry("vendor/a", "vendor/a", Some(60.0), 0.0)],
        };
        let plan = pool(Arc::clone(&calls))
            .plan_standing_task(
                &catalog,
                &task(Some(50.0)),
                &SharedQuotaLedger::default(),
                100,
            )
            .await?;
        assert_eq!(plan.status, StandingPlanStatus::Ready);
        assert_eq!(
            plan.selected
                .as_ref()
                .map(|value| value.canonical_id.as_str()),
            Some("vendor/a")
        );
        assert_eq!(plan.probed_models, 1);
        Ok(())
    }

    #[tokio::test]
    async fn unknown_quality_waits_without_provider_probe() -> Result<(), StandingTaskError> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let catalog = ModelCatalog {
            models: vec![catalog_entry("vendor/a", "vendor/a", None, 0.0)],
        };
        let plan = pool(Arc::clone(&calls))
            .plan_standing_task(
                &catalog,
                &task(Some(50.0)),
                &SharedQuotaLedger::default(),
                100,
            )
            .await?;
        assert_eq!(plan.status, StandingPlanStatus::Waiting);
        assert_eq!(plan.suitability_eligible_models, 0);
        assert_eq!(calls.lock().map_or(usize::MAX, |calls| calls.len()), 0);
        Ok(())
    }

    #[tokio::test]
    async fn absolute_cost_ceiling_filters_before_provider_probe() -> Result<(), StandingTaskError>
    {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let catalog = ModelCatalog {
            models: vec![catalog_entry("vendor/a", "vendor/a", Some(60.0), 0.01)],
        };
        let mut work = task(Some(50.0));
        work.policy.max_expected_cost_usd = Some(0.001);
        let plan = pool(Arc::clone(&calls))
            .plan_standing_task(&catalog, &work, &SharedQuotaLedger::default(), 100)
            .await?;
        assert_eq!(plan.status, StandingPlanStatus::Waiting);
        assert_eq!(plan.catalog_over_budget_models, 1);
        assert_eq!(plan.probed_models, 0);
        assert_eq!(calls.lock().map_or(usize::MAX, |calls| calls.len()), 0);
        Ok(())
    }

    #[tokio::test]
    async fn probe_budget_checks_cheapest_catalog_model_first() -> Result<(), StandingTaskError> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let catalog = ModelCatalog {
            models: vec![
                catalog_entry("vendor/expensive", "vendor/expensive", Some(60.0), 0.001),
                catalog_entry("vendor/free", "vendor/free", Some(60.0), 0.0),
            ],
        };
        let mut work = task(Some(50.0));
        work.policy.max_model_probes = 1;
        let plan = pool(Arc::clone(&calls))
            .plan_standing_task(&catalog, &work, &SharedQuotaLedger::default(), 100)
            .await?;
        assert_eq!(plan.status, StandingPlanStatus::Ready);
        assert_eq!(plan.probed_models, 1);
        assert_eq!(
            calls.lock().map(|calls| calls.clone()).unwrap_or_default(),
            vec!["vendor/free".to_owned()]
        );
        Ok(())
    }
}
