use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::Mutex;

use crate::router::{lowered_strict_schema, supports_request};
use crate::{
    ConformanceProfile, InferenceProvider, InferenceRequest, InferenceRequestTemplate,
    InferenceResponse, ModelCatalog, ModelLookup, ModelRouteClass, ModelRouteSelectionCandidate,
    ProviderError, QuotaAdmission, QuotaAdmissionReport, QuotaAwareRoutePlan, QuotaDemand,
    QuotaLedger, QuotaRouteKey, QuotaUnknownPolicy, RetryPolicy, RouteCandidate, RoutePlan,
    RoutingStrategy, build_route_plan, request_prompt_tokens,
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
    conformance: Arc<Mutex<ConformanceProfile>>,
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

pub(crate) struct QuotaEndpointDiscovery {
    pub endpoints: Vec<crate::Endpoint>,
    pub reports: Vec<QuotaAdmissionReport>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaExecutionContext {
    pub unknown_policy: QuotaUnknownPolicy,
    pub now_unix: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SelectedRouteExecutionPolicy {
    pub quota_unknown_policy: QuotaUnknownPolicy,
    pub now_unix: u64,
    pub max_cost_usd: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectedRouteValidation {
    pub candidate: RouteCandidate,
    /// Worst-case request cost using `max_output_tokens` when supplied.
    pub maximum_cost_usd: f64,
    pub quota: QuotaAdmissionReport,
}

#[derive(Debug, Error)]
pub enum SelectedRouteExecutionError {
    #[error(transparent)]
    Provider(#[from] ProviderError),
    #[error("selected route is no longer available: {0}")]
    RouteUnavailable(String),
    #[error("selected route is a batch route and requires the batch lifecycle")]
    BatchRoute,
    #[error("revalidated maximum request cost {cost_usd} USD exceeds maximum {maximum_usd} USD")]
    CostExceeded { cost_usd: f64, maximum_usd: f64 },
    #[error("invalid selected-route execution policy: {0}")]
    InvalidPolicy(String),
    #[error("selected route has no quota admission: {report:?}")]
    QuotaUnavailable { report: Box<QuotaAdmissionReport> },
    #[error("quota accounting invariant failed: {0}")]
    Accounting(String),
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

fn commit_successful_quota(
    ledger: &mut QuotaLedger,
    reservation: crate::QuotaReservationId,
    route: &QuotaRouteKey,
    demand: QuotaDemand,
    response: &InferenceResponse,
    now_unix: u64,
) -> Result<(), QuotaInferenceError> {
    let actual = demand.with_actual_usage(response);
    if ledger
        .commit_with_observations(
            reservation,
            route,
            Some(actual),
            &response.quota_observations,
            now_unix,
        )
        .is_some()
    {
        Ok(())
    } else {
        Err(QuotaInferenceError::LedgerInvariant(
            "failed to commit successful inference reservation".to_owned(),
        ))
    }
}

fn settle_failed_quota(
    ledger: &mut QuotaLedger,
    reservation: crate::QuotaReservationId,
    route: &QuotaRouteKey,
    error: &ProviderError,
    now_unix: u64,
) -> Result<(), QuotaInferenceError> {
    let accounted = if error.execution_phase().is_some() {
        ledger
            .commit_with_observations(
                reservation,
                route,
                None,
                error.quota_observations(),
                now_unix,
            )
            .is_some()
    } else {
        ledger.release(reservation)
    };
    if accounted {
        Ok(())
    } else {
        Err(QuotaInferenceError::LedgerInvariant(
            "failed to settle failed inference reservation".to_owned(),
        ))
    }
}

impl InferencePool {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_conformance_profile(mut self, profile: ConformanceProfile) -> Self {
        self.conformance = Arc::new(Mutex::new(profile));
        self
    }

    /// Share mutable conformance evidence with a long-lived caller such as the stdio bridge.
    #[must_use]
    pub fn with_shared_conformance_profile(
        mut self,
        profile: Arc<Mutex<ConformanceProfile>>,
    ) -> Self {
        self.conformance = profile;
        self
    }

    /// Snapshot the current in-process conformance evidence.
    pub async fn conformance_profile(&self) -> ConformanceProfile {
        self.conformance.lock().await.clone()
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
    /// Returns an error when model identity is ambiguous/unresolved, every matching provider
    /// discovery fails, or no registered provider exposes an eligible route class for it.
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

        let conformance = self.conformance.lock().await.clone();
        let mut endpoints = Vec::new();
        let mut discovery_error = None;
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
                let mut discovered = match provider.endpoints(&route.id).await {
                    Ok(discovered) => discovered,
                    Err(ProviderError::NoEligibleEndpoint(_)) => continue,
                    Err(error) => {
                        discovery_error = Some(error);
                        continue;
                    }
                };
                for endpoint in &mut discovered {
                    // Catalog identity and registry credential are authoritative over adapter hints.
                    endpoint.source.clone_from(&route.source);
                    endpoint.credential.clone_from(&instance.credential);
                    endpoint.model.clone_from(&route.id);
                    endpoint.route_class = route.route_class;
                    conformance.apply_to_endpoint(endpoint);
                }
                endpoints.extend(discovered);
            }
        }

        if endpoints.is_empty() {
            return Err(discovery_error
                .unwrap_or_else(|| ProviderError::NoEligibleEndpoint(request.model.clone())));
        }
        Ok(endpoints)
    }

    async fn record_runtime_schema_failure(
        &self,
        endpoint: &crate::Endpoint,
        request: &InferenceRequest,
        failure: &crate::StructuredOutputFailure,
    ) {
        let Some((transport, features)) = runtime_schema_transport(endpoint, request, failure)
        else {
            return;
        };
        self.conformance
            .lock()
            .await
            .record_runtime_schema_failure(endpoint, transport, &features, failure);
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

    pub(crate) async fn quota_admitted_endpoints(
        &self,
        catalog: &ModelCatalog,
        request: &InferenceRequest,
        ledger: &mut QuotaLedger,
        unknown_policy: QuotaUnknownPolicy,
        now_unix: u64,
    ) -> Result<QuotaEndpointDiscovery, ProviderError> {
        let endpoints = self.endpoints(catalog, request).await?;
        let prompt_tokens = request_prompt_tokens(request);
        let lowered_strict_schema = lowered_strict_schema(request);
        let demand = QuotaDemand::for_request(request);
        let mut reports = Vec::new();
        let mut admitted = Vec::new();
        for endpoint in endpoints.into_iter().filter(|endpoint| {
            supports_request(
                endpoint,
                request,
                prompt_tokens,
                lowered_strict_schema.as_ref(),
            )
        }) {
            let route_key = QuotaRouteKey::from_endpoint(&endpoint);
            let instance =
                ProviderInstanceKey::new(endpoint.source.clone(), endpoint.credential.clone());
            if let Some(provider) = self.providers.get(&instance) {
                provider.configure_quota_route(&route_key, ledger)?;
            }
            let report = ledger.evaluate(&route_key, demand, now_unix);
            let keep = match report.admission {
                QuotaAdmission::Admitted => true,
                QuotaAdmission::Indeterminate => unknown_policy == QuotaUnknownPolicy::Allow,
                QuotaAdmission::Rejected => false,
            };
            if keep {
                admitted.push(endpoint);
            }
            reports.push(report);
        }
        Ok(QuotaEndpointDiscovery {
            endpoints: admitted,
            reports,
        })
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
        ledger: &mut QuotaLedger,
        unknown_policy: QuotaUnknownPolicy,
        now_unix: u64,
    ) -> Result<QuotaAwareRoutePlan, QuotaRoutingError> {
        let discovery = self
            .quota_admitted_endpoints(catalog, request, ledger, unknown_policy, now_unix)
            .await?;
        if discovery.endpoints.is_empty() {
            return Err(QuotaRoutingError::NoQuotaAdmissible {
                model: request.model.clone(),
                reports: Box::new(discovery.reports),
            });
        }
        let plan = build_route_plan(strategy, request, discovery.endpoints)?;
        Ok(QuotaAwareRoutePlan {
            plan,
            quota_reports: discovery.reports,
        })
    }

    /// Revalidate one previously selected concrete route against current endpoint, capability,
    /// pricing, and quota evidence.
    ///
    /// # Errors
    /// Returns an error when the exact route disappeared, the request shape no longer fits,
    /// the current expected cost exceeds policy, or quota does not admit the route.
    pub async fn revalidate_selected_with_quota(
        &self,
        catalog: &ModelCatalog,
        selected: &ModelRouteSelectionCandidate,
        request_template: &InferenceRequestTemplate,
        ledger: &mut QuotaLedger,
        policy: SelectedRouteExecutionPolicy,
    ) -> Result<SelectedRouteValidation, SelectedRouteExecutionError> {
        if !policy.max_cost_usd.is_finite() || policy.max_cost_usd < 0.0 {
            return Err(SelectedRouteExecutionError::InvalidPolicy(
                "max_cost_usd must be finite and non-negative".to_owned(),
            ));
        }
        let request = request_template.for_model(selected.canonical_id.clone());
        let endpoints = self.endpoints(catalog, &request).await?;
        let wanted = &selected.route.endpoint;
        let endpoint = endpoints
            .into_iter()
            .find(|endpoint| {
                endpoint.source == wanted.source
                    && endpoint.credential == wanted.credential
                    && endpoint.provider_key == wanted.provider_key
                    && endpoint.id == wanted.id
                    && endpoint.model == wanted.model
                    && endpoint.route_class == wanted.route_class
            })
            .ok_or_else(|| {
                SelectedRouteExecutionError::RouteUnavailable(format!(
                    "{}/{}/{}",
                    wanted.source, wanted.provider_key, wanted.model
                ))
            })?;
        let current_plan = build_route_plan(
            RoutingStrategy::FastestResponseCheap(1.0),
            &request,
            vec![endpoint],
        )?;
        let Some(candidate) = current_plan.candidates.into_iter().next() else {
            return Err(SelectedRouteExecutionError::RouteUnavailable(
                selected.canonical_id.clone(),
            ));
        };
        let maximum_output_tokens = request
            .max_output_tokens
            .unwrap_or(request.expected_output_tokens);
        let maximum_cost_usd = candidate
            .endpoint
            .pricing
            .expected_cost(current_plan.prompt_tokens, maximum_output_tokens);
        if !maximum_cost_usd.is_finite() || maximum_cost_usd < 0.0 {
            return Err(SelectedRouteExecutionError::InvalidPolicy(
                "revalidated route produced a non-finite or negative maximum cost".to_owned(),
            ));
        }
        if maximum_cost_usd > policy.max_cost_usd + f64::EPSILON {
            return Err(SelectedRouteExecutionError::CostExceeded {
                cost_usd: maximum_cost_usd,
                maximum_usd: policy.max_cost_usd,
            });
        }
        let route = QuotaRouteKey::from_endpoint(&candidate.endpoint);
        let instance = ProviderInstanceKey::new(
            candidate.endpoint.source.clone(),
            candidate.endpoint.credential.clone(),
        );
        let provider = self.providers.get(&instance).ok_or_else(|| {
            SelectedRouteExecutionError::RouteUnavailable(selected.canonical_id.clone())
        })?;
        provider.configure_quota_route(&route, ledger)?;
        let report = ledger.evaluate(&route, QuotaDemand::for_request(&request), policy.now_unix);
        let admitted = match report.admission {
            QuotaAdmission::Admitted => true,
            QuotaAdmission::Indeterminate => {
                policy.quota_unknown_policy == QuotaUnknownPolicy::Allow
            }
            QuotaAdmission::Rejected => false,
        };
        if !admitted {
            return Err(SelectedRouteExecutionError::QuotaUnavailable {
                report: Box::new(report),
            });
        }
        Ok(SelectedRouteValidation {
            candidate,
            maximum_cost_usd,
            quota: report,
        })
    }

    /// Execute one exact previously selected synchronous route after fresh revalidation.
    ///
    /// # Errors
    /// Returns revalidation, quota-accounting, provider-execution, or batch-lifecycle errors.
    pub async fn infer_selected_with_quota(
        &self,
        catalog: &ModelCatalog,
        selected: &ModelRouteSelectionCandidate,
        request_template: &InferenceRequestTemplate,
        retry: RetryPolicy,
        ledger: &mut QuotaLedger,
        policy: SelectedRouteExecutionPolicy,
    ) -> Result<QuotaPooledInferenceResponse, SelectedRouteExecutionError> {
        if selected.route.endpoint.route_class == ModelRouteClass::Batch {
            return Err(SelectedRouteExecutionError::BatchRoute);
        }
        let validation = self
            .revalidate_selected_with_quota(catalog, selected, request_template, ledger, policy)
            .await?;
        let candidate = validation.candidate.clone();
        let route = QuotaRouteKey::from_endpoint(&candidate.endpoint);
        let request = request_template.for_model(candidate.endpoint.model.clone());
        let demand = QuotaDemand::for_request(&request);
        let reservation = ledger
            .reserve_with_policy(&route, demand, policy.now_unix, policy.quota_unknown_policy)
            .map_err(|report| SelectedRouteExecutionError::QuotaUnavailable { report })?;
        let instance = ProviderInstanceKey::new(
            candidate.endpoint.source.clone(),
            candidate.endpoint.credential.clone(),
        );
        let provider = self.providers.get(&instance).ok_or_else(|| {
            SelectedRouteExecutionError::RouteUnavailable(selected.canonical_id.clone())
        })?;
        let plan = RoutePlan {
            prompt_tokens: request_prompt_tokens(&request),
            expected_output_tokens: request.expected_output_tokens,
            cheapest_cost_usd: candidate.expected_cost_usd,
            cost_ceiling_usd: candidate.expected_cost_usd,
            candidates: vec![candidate.clone()],
        };
        match provider.execute_plan(&request, &plan, retry).await {
            Ok(response) => {
                commit_successful_quota(
                    ledger,
                    reservation,
                    &route,
                    demand,
                    &response,
                    policy.now_unix,
                )
                .map_err(|error| SelectedRouteExecutionError::Accounting(error.to_string()))?;
                match crate::structured_output::validate_inference_response(&request, response) {
                    Ok(response) => Ok(QuotaPooledInferenceResponse {
                        quota_plan: QuotaAwareRoutePlan {
                            plan,
                            quota_reports: vec![validation.quota],
                        },
                        selected: candidate,
                        response,
                    }),
                    Err(failure) => {
                        self.record_runtime_schema_failure(&candidate.endpoint, &request, &failure)
                            .await;
                        Err(SelectedRouteExecutionError::Provider(
                            ProviderError::Response(failure.to_string()),
                        ))
                    }
                }
            }
            Err(error) => {
                settle_failed_quota(ledger, reservation, &route, &error, policy.now_unix).map_err(
                    |accounting| SelectedRouteExecutionError::Accounting(accounting.to_string()),
                )?;
                Err(SelectedRouteExecutionError::Provider(error))
            }
        }
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
                    commit_successful_quota(
                        ledger,
                        reservation,
                        &route_key,
                        demand,
                        &response,
                        quota.now_unix,
                    )?;
                    match crate::structured_output::validate_inference_response(request, response) {
                        Ok(response) => {
                            return Ok(QuotaPooledInferenceResponse {
                                quota_plan,
                                selected: candidate,
                                response,
                            });
                        }
                        Err(failure) => {
                            self.record_runtime_schema_failure(
                                &candidate.endpoint,
                                request,
                                &failure,
                            )
                            .await;
                            last_error =
                                Some(Box::new(ProviderError::Response(failure.to_string())));
                        }
                    }
                }
                Err(error) => {
                    settle_failed_quota(ledger, reservation, &route_key, &error, quota.now_unix)?;
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
        let mut last_error = None;
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
            match provider
                .execute_plan(&native_request, &candidate_plan, retry)
                .await
            {
                Ok(response) => {
                    match crate::structured_output::validate_inference_response(request, response) {
                        Ok(response) => {
                            return Ok(PooledInferenceResponse {
                                plan,
                                selected: candidate,
                                response,
                            });
                        }
                        Err(failure) => {
                            self.record_runtime_schema_failure(
                                &candidate.endpoint,
                                request,
                                &failure,
                            )
                            .await;
                            last_error = Some(ProviderError::Response(failure.to_string()));
                        }
                    }
                }
                Err(error) => last_error = Some(error),
            }
        }
        Err(last_error.unwrap_or_else(|| {
            ProviderError::Request(format!("all {attempted} ranked provider candidates failed"))
        }))
    }
}

fn runtime_schema_transport(
    endpoint: &crate::Endpoint,
    request: &InferenceRequest,
    failure: &crate::StructuredOutputFailure,
) -> Option<(
    crate::SchemaFeatureTransport,
    BTreeSet<crate::SchemaFeature>,
)> {
    let tool_failure = failure.violations.iter().any(|violation| {
        violation.instance_path.starts_with("/tool_calls")
            || matches!(
                violation.keyword.as_str(),
                "tool_choice" | "tool_name" | "parallel_tool_calls"
            )
    });
    if tool_failure && !request.tools.is_empty() {
        return runtime_tool_schema_transport(endpoint, request);
    }
    match request.requirements.response_schema.as_ref() {
        Some(crate::ResponseSchema::JsonObject) => Some((
            crate::SchemaFeatureTransport::StructuredOutputs,
            BTreeSet::new(),
        )),
        Some(crate::ResponseSchema::JsonSchema {
            schema,
            strict: true,
            ..
        }) => {
            let lowered = crate::lower_json_schema(schema);
            let transport = endpoint
                .capabilities
                .strict_output_transport_for_lowered_schema_with_policy(
                    &lowered,
                    request.requirements.schema_evidence,
                )?;
            match transport {
                crate::StrictOutputTransport::Native => Some((
                    crate::SchemaFeatureTransport::StructuredOutputs,
                    crate::schema_feature_requirements(&lowered),
                )),
                crate::StrictOutputTransport::ForcedTool => {
                    let (tool_schema, _) =
                        crate::model::forced_tool_parameter_schema_from_lowered(&lowered);
                    Some((
                        crate::SchemaFeatureTransport::Tools,
                        crate::schema_feature_requirements(&tool_schema),
                    ))
                }
            }
        }
        Some(crate::ResponseSchema::JsonSchema { strict: false, .. }) | None => {
            if request.tools.is_empty() {
                None
            } else {
                runtime_tool_schema_transport(endpoint, request)
            }
        }
    }
}

fn runtime_tool_schema_transport(
    endpoint: &crate::Endpoint,
    request: &InferenceRequest,
) -> Option<(
    crate::SchemaFeatureTransport,
    BTreeSet<crate::SchemaFeature>,
)> {
    match endpoint.capabilities.tool_transport(request)? {
        crate::ToolTransport::Native => {
            let features = request.tools.iter().filter(|tool| tool.strict).fold(
                BTreeSet::new(),
                |mut features, tool| {
                    let lowered = crate::lower_json_schema(&tool.parameters);
                    features.extend(crate::schema_feature_requirements(&lowered));
                    features
                },
            );
            Some((crate::SchemaFeatureTransport::Tools, features))
        }
        crate::ToolTransport::StructuredOutput => request
            .lowered_structured_tool_response_schema()
            .map(|(_, features)| (crate::SchemaFeatureTransport::StructuredOutputs, features)),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
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
                workload_policy: crate::WorkloadPolicy::default(),
                capabilities: EndpointCapabilities {
                    adapter_parameters: BTreeSet::from(["max_tokens".to_owned()]),
                    supported_parameters: BTreeSet::from(["max_tokens".to_owned()]),
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
                    quota_observations: Vec::new(),
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
                tool_calls: Vec::new(),
                output_repairs: Vec::new(),
                quota_observations: Vec::new(),
                raw: json!({"native_model": request.model, "mock_price": self.price}),
            })
        }
    }

    struct StructuredProvider {
        content: &'static str,
    }

    #[async_trait]
    impl InferenceProvider for StructuredProvider {
        async fn endpoints(&self, model: &str) -> Result<Vec<Endpoint>, ProviderError> {
            Ok(vec![Endpoint {
                source: "openrouter".to_owned(),
                credential: None,
                provider_key: "structured".to_owned(),
                id: "structured/fp8".to_owned(),
                provider: "Structured".to_owned(),
                model: model.to_owned(),
                route_class: ModelRouteClass::Standard,
                display_name: "Structured".to_owned(),
                quantization: None,
                status: 0,
                context_length: Some(100_000),
                max_prompt_tokens: None,
                max_completion_tokens: None,
                pricing: Pricing::default(),
                stats: EndpointStats {
                    uptime_30m: Some(100.0),
                    ..EndpointStats::default()
                },
                workload_policy: crate::WorkloadPolicy::default(),
                capabilities: EndpointCapabilities {
                    adapter_parameters: BTreeSet::from([
                        "max_tokens".to_owned(),
                        "response_format".to_owned(),
                        "structured_outputs".to_owned(),
                    ]),
                    supported_parameters: BTreeSet::from([
                        "max_tokens".to_owned(),
                        "response_format".to_owned(),
                        "structured_outputs".to_owned(),
                    ]),
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

        async fn execute_plan(
            &self,
            request: &InferenceRequest,
            _plan: &RoutePlan,
            _retry: RetryPolicy,
        ) -> Result<InferenceResponse, ProviderError> {
            Ok(InferenceResponse {
                content: self.content.to_owned(),
                model: Some(request.model.clone()),
                provider: Some("Structured".to_owned()),
                usage_cost_usd: Some(0.0),
                usage: crate::TokenUsage::default(),
                tool_calls: Vec::new(),
                output_repairs: Vec::new(),
                quota_observations: Vec::new(),
                raw: json!({}),
            })
        }
    }

    struct ObservedQuotaFailureProvider {
        price: f64,
        calls: Arc<StdMutex<Vec<String>>>,
    }

    #[async_trait]
    impl InferenceProvider for ObservedQuotaFailureProvider {
        async fn endpoints(&self, model: &str) -> Result<Vec<Endpoint>, ProviderError> {
            self.calls
                .lock()
                .map_err(|_| ProviderError::Catalog("poisoned test mutex".to_owned()))?
                .push(model.to_owned());
            Ok(vec![Endpoint {
                source: "nvidia".to_owned(),
                credential: None,
                provider_key: "nvidia".to_owned(),
                id: "nvidia".to_owned(),
                provider: "NVIDIA".to_owned(),
                model: model.to_owned(),
                route_class: ModelRouteClass::Free,
                display_name: "NVIDIA".to_owned(),
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
                workload_policy: crate::WorkloadPolicy::default(),
                capabilities: EndpointCapabilities {
                    adapter_parameters: BTreeSet::from(["max_tokens".to_owned()]),
                    supported_parameters: BTreeSet::from(["max_tokens".to_owned()]),
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
            Err(ProviderError::Execution {
                phase: crate::ExecutionFailurePhase::ResponseReceived,
                detail: "rate limited".to_owned(),
                quota_observations: vec![crate::QuotaObservation {
                    bucket: "rate_limit.requests".to_owned(),
                    metric: Some(crate::QuotaMetric::Requests),
                    limit: Some(100),
                    remaining: Some(0),
                    reset_after_millis: Some(1_000),
                }],
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
            aliases: BTreeSet::from([
                id.to_owned(),
                "vendor/model".to_owned(),
                "vendor/model-v1".to_owned(),
            ]),
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

    fn selected_candidate(
        class: ModelRouteClass,
        expected_cost_usd: f64,
    ) -> ModelRouteSelectionCandidate {
        let model = if class == ModelRouteClass::Batch {
            "vendor/model:batch"
        } else {
            "vendor/model"
        };
        let endpoint = Endpoint {
            source: "openrouter".to_owned(),
            credential: None,
            provider_key: "openrouter".to_owned(),
            id: "openrouter".to_owned(),
            provider: "openrouter".to_owned(),
            model: model.to_owned(),
            route_class: class,
            display_name: model.to_owned(),
            quantization: None,
            status: 0,
            context_length: None,
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            stats: EndpointStats::default(),
            workload_policy: crate::WorkloadPolicy::default(),
            capabilities: EndpointCapabilities::default(),
        };
        let route = QuotaRouteKey::from_endpoint(&endpoint);
        ModelRouteSelectionCandidate {
            canonical_id: "vendor/model-v1".to_owned(),
            suitability: crate::ModelSuitabilityReport {
                canonical_id: "vendor/model-v1".to_owned(),
                metadata_source: "openrouter".to_owned(),
                metadata_route_id: "vendor/model".to_owned(),
                status: crate::SuitabilityStatus::Suitable,
                checks: Vec::new(),
            },
            quota: QuotaAdmissionReport {
                route,
                admission: QuotaAdmission::Indeterminate,
                checks: Vec::new(),
            },
            route: RouteCandidate {
                endpoint,
                expected_cost_usd,
                expected_response_p75_seconds: None,
                expected_response_p95_seconds: None,
            },
        }
    }

    fn selected_template() -> InferenceRequestTemplate {
        InferenceRequestTemplate {
            system_prompt: None,
            prompt: "hi".to_owned(),
            expected_output_tokens: 8,
            prompt_tokens: Some(10),
            max_output_tokens: Some(20),
            temperature: None,
            include_reasoning: false,
            tools: Vec::new(),
            tool_choice: None,
            parallel_tool_calls: None,
            agentic: false,
            session_id: None,
            requirements: crate::RequestRequirements {
                minimum_uptime_30m: None,
                ..crate::RequestRequirements::default()
            },
        }
    }

    #[tokio::test]
    async fn selected_route_revalidation_blocks_cost_drift()
    -> Result<(), Box<dyn std::error::Error>> {
        let calls = Arc::new(StdMutex::new(Vec::new()));
        let mut pool = InferencePool::new();
        pool.register(
            "openrouter",
            Arc::new(MockProvider {
                source: "openrouter",
                price: 1e-6,
                fail_execute: false,
                calls,
            }),
        );
        let mut ledger = QuotaLedger::new(crate::QuotaGraph::new());
        let error = pool
            .revalidate_selected_with_quota(
                &catalog(),
                &selected_candidate(ModelRouteClass::Standard, 0.0),
                &selected_template(),
                &mut ledger,
                SelectedRouteExecutionPolicy {
                    quota_unknown_policy: QuotaUnknownPolicy::Allow,
                    now_unix: 100,
                    max_cost_usd: 20e-6,
                },
            )
            .await
            .err()
            .ok_or("expected cost drift rejection")?;
        assert!(matches!(
            error,
            SelectedRouteExecutionError::CostExceeded { .. }
        ));
        Ok(())
    }

    #[tokio::test]
    async fn selected_sync_route_is_revalidated_then_executed_exactly()
    -> Result<(), Box<dyn std::error::Error>> {
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
        let mut ledger = QuotaLedger::new(crate::QuotaGraph::new());
        let result = pool
            .infer_selected_with_quota(
                &catalog(),
                &selected_candidate(ModelRouteClass::Standard, 1e-6),
                &selected_template(),
                RetryPolicy::default(),
                &mut ledger,
                SelectedRouteExecutionPolicy {
                    quota_unknown_policy: QuotaUnknownPolicy::Allow,
                    now_unix: 100,
                    max_cost_usd: 1.0,
                },
            )
            .await?;
        assert_eq!(result.response.content, "ok");
        assert_eq!(result.selected.endpoint.id, "openrouter");
        let calls = calls.lock().map_err(|_| "poisoned test mutex")?;
        assert!(calls.iter().any(|call| call == "execute:vendor/model"));
        Ok(())
    }

    #[tokio::test]
    async fn selected_batch_route_never_enters_sync_execution()
    -> Result<(), Box<dyn std::error::Error>> {
        let pool = InferencePool::new();
        let mut ledger = QuotaLedger::new(crate::QuotaGraph::new());
        let error = pool
            .infer_selected_with_quota(
                &catalog(),
                &selected_candidate(ModelRouteClass::Batch, 0.0),
                &selected_template(),
                RetryPolicy::default(),
                &mut ledger,
                SelectedRouteExecutionPolicy {
                    quota_unknown_policy: QuotaUnknownPolicy::Allow,
                    now_unix: 100,
                    max_cost_usd: 1.0,
                },
            )
            .await
            .err()
            .ok_or("expected batch lifecycle rejection")?;
        assert!(matches!(error, SelectedRouteExecutionError::BatchRoute));
        Ok(())
    }

    #[tokio::test]
    async fn discovered_endpoints_apply_conformance_profile() -> Result<(), ProviderError> {
        let calls = Arc::new(StdMutex::new(Vec::new()));
        let profile = ConformanceProfile {
            observations: vec![crate::ConformanceObservation {
                source: "nvidia".to_owned(),
                credential: None,
                provider_key: "nvidia".to_owned(),
                endpoint_id: None,
                native_model_id: "vendor/native-model".to_owned(),
                canonical_id: Some("vendor/model-v1".to_owned()),
                parameter: "max_tokens".to_owned(),
                reasoning_effort: None,
                reasoning_tokens: None,
                completion_tokens: None,
                reasoning_chars: None,
                reasoning_trace: None,
                response_content: None,
                tool_arguments: None,
                puzzle_correct: None,
                finish_reason: None,
                constraint_satisfied: None,
                test_id: "max-tokens-test".to_owned(),
                test_version: 1,
                outcome: crate::ConformanceOutcome::Failed,
                observed_unix_ms: 1,
                detail: None,
            }],
        };
        let mut pool = InferencePool::new().with_conformance_profile(profile);
        pool.register(
            "nvidia",
            Arc::new(MockProvider {
                source: "nvidia",
                price: 0.0,
                fail_execute: false,
                calls,
            }),
        );
        let request = InferenceRequest::new("vendor/native-model", "hi", 8);
        let endpoints = pool.endpoints(&catalog(), &request).await?;
        let endpoint = endpoints
            .iter()
            .find(|endpoint| endpoint.source == "nvidia")
            .ok_or_else(|| ProviderError::Response("missing nvidia endpoint".to_owned()))?;
        assert!(
            endpoint
                .capabilities
                .rejected_parameters
                .contains("max_tokens")
        );
        assert!(!endpoint.capabilities.supports_parameter("max_tokens"));
        Ok(())
    }

    #[tokio::test]
    async fn default_policy_queries_non_batch_fixture_routes() -> Result<(), ProviderError> {
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
    async fn standard_only_policy_executes_paid_openrouter_route() -> Result<(), ProviderError> {
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
        let mut request = InferenceRequest::new("vendor/model", "paid", 8);
        request.requirements.allowed_route_classes = BTreeSet::from([ModelRouteClass::Standard]);
        let result = pool
            .infer(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
                RetryPolicy::default(),
            )
            .await?;
        assert_eq!(result.selected.endpoint.source, "openrouter");
        assert_eq!(
            result.selected.endpoint.route_class,
            ModelRouteClass::Standard
        );
        assert!(result.selected.expected_cost_usd > 0.0);
        let calls = calls
            .lock()
            .map_err(|_| ProviderError::Catalog("poisoned test mutex".to_owned()))?;
        assert!(calls.iter().any(|call| call == "execute:vendor/model"));
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
    async fn learn_mode_repairs_unambiguous_real_response() -> Result<(), ProviderError> {
        let mut pool = InferencePool::new();
        pool.register(
            "openrouter",
            Arc::new(StructuredProvider {
                content: r#"{"n":"1"}"#,
            }),
        );
        let mut request = InferenceRequest::new("vendor/model", "return one", 8);
        request.requirements.schema_evidence = crate::SchemaEvidencePolicy::Learn;
        request.requirements.response_schema = Some(crate::ResponseSchema::JsonSchema {
            name: "answer".to_owned(),
            schema: json!({
                "type":"object",
                "properties":{"n":{"type":"integer"}},
                "required":["n"],
                "additionalProperties":false
            }),
            strict: true,
        });
        let result = pool
            .infer(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
                RetryPolicy::default(),
            )
            .await?;
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&result.response.content).ok(),
            Some(json!({"n":1}))
        );
        assert_eq!(result.response.output_repairs.len(), 1);
        assert_eq!(
            result.response.output_repairs[0].kind,
            crate::OutputRepairKind::StringToInteger
        );
        assert_eq!(pool.conformance_profile().await.observations, Vec::new());
        Ok(())
    }

    #[tokio::test]
    async fn learn_mode_baseline_failure_demotes_strict_transport() {
        let mut pool = InferencePool::new();
        pool.register(
            "openrouter",
            Arc::new(StructuredProvider {
                content: "not-json",
            }),
        );
        let mut request = InferenceRequest::new("vendor/model", "return n", 8);
        request.requirements.schema_evidence = crate::SchemaEvidencePolicy::Learn;
        request.requirements.response_schema = Some(crate::ResponseSchema::JsonSchema {
            name: "answer".to_owned(),
            schema: json!({
                "type":"object",
                "properties":{"n":{"type":"integer"}},
                "required":["n"],
                "additionalProperties":false
            }),
            strict: true,
        });
        let first = pool
            .infer(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
                RetryPolicy::default(),
            )
            .await;
        assert!(matches!(first, Err(ProviderError::Response(_))));
        let profile = pool.conformance_profile().await;
        assert!(profile.observations.iter().any(|observation| {
            observation.test_id == "strict-json-schema"
                && observation.outcome == crate::ConformanceOutcome::Passed
                && observation.constraint_satisfied == Some(false)
        }));
        assert!(matches!(
            pool.route(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
            )
            .await,
            Err(ProviderError::NoEligibleEndpoint(_))
        ));
    }

    #[tokio::test]
    async fn learn_mode_negative_evidence_blocks_same_wire_feature_next_time() {
        let mut pool = InferencePool::new();
        pool.register(
            "openrouter",
            Arc::new(StructuredProvider { content: r#""c""# }),
        );
        let mut request = InferenceRequest::new("vendor/model", "return a or b", 8);
        request.requirements.schema_evidence = crate::SchemaEvidencePolicy::Learn;
        request.requirements.response_schema = Some(crate::ResponseSchema::JsonSchema {
            name: "answer".to_owned(),
            schema: json!({"oneOf":[{"const":"a"},{"const":"b"}]}),
            strict: true,
        });
        let first = pool
            .infer(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
                RetryPolicy::default(),
            )
            .await;
        assert!(matches!(first, Err(ProviderError::Response(_))));

        let profile = pool.conformance_profile().await;
        assert!(profile.observations.iter().any(|observation| {
            observation.test_id == "schema-feature-structured_outputs-enum"
                && observation.outcome == crate::ConformanceOutcome::Failed
        }));
        assert!(matches!(
            pool.route(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
            )
            .await,
            Err(ProviderError::NoEligibleEndpoint(_))
        ));
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
    async fn synchronous_infer_preserves_last_provider_error() -> Result<(), ProviderError> {
        let calls = Arc::new(StdMutex::new(Vec::new()));
        let mut pool = InferencePool::new();
        pool.register(
            "openrouter",
            Arc::new(MockProvider {
                source: "openrouter",
                price: 0.0,
                fail_execute: true,
                calls,
            }),
        );
        let request = InferenceRequest::new("vendor/model", "hi", 8);
        let error = pool
            .infer(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
                RetryPolicy::default(),
            )
            .await
            .err()
            .ok_or_else(|| ProviderError::Request("expected provider failure".to_owned()))?;
        assert!(matches!(
            error,
            ProviderError::Execution {
                detail,
                ..
            } if detail == "mock failure"
        ));
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
        let mut ledger = QuotaLedger::new(graph);
        let request = InferenceRequest::new("vendor/model", "hi", 8);

        let result = pool
            .route_with_quota(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
                &mut ledger,
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
        let mut ledger = QuotaLedger::new(crate::QuotaGraph::new());
        let permissive = pool
            .route_with_quota(
                &catalog(),
                RoutingStrategy::FastestResponseCheap(1.5),
                &request,
                &mut ledger,
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
                &mut ledger,
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
    async fn response_quota_snapshot_blocks_shared_sibling_before_failover() {
        let failing_calls = Arc::new(StdMutex::new(Vec::new()));
        let healthy_calls = Arc::new(StdMutex::new(Vec::new()));
        let mut pool = InferencePool::new();
        pool.register_credential(
            "nvidia",
            "a-failing",
            Arc::new(ObservedQuotaFailureProvider {
                price: 1e-6,
                calls: Arc::clone(&failing_calls),
            }),
        );
        pool.register_credential(
            "nvidia",
            "b-healthy",
            Arc::new(MockProvider {
                source: "nvidia",
                price: 1.1e-6,
                fail_execute: false,
                calls: Arc::clone(&healthy_calls),
            }),
        );
        let shared = crate::QuotaResource::finite(
            "nvidia/account/rpd",
            crate::QuotaScope::Account,
            crate::QuotaMetric::Requests,
            100,
            10,
        );
        let shared_id = shared.id.clone();
        let mut graph = crate::QuotaGraph::new();
        graph.insert_resource(shared);
        let mut observation_map = crate::QuotaObservationMap::new();
        for credential in ["a-failing", "b-healthy"] {
            let route = QuotaRouteKey {
                source: "nvidia".to_owned(),
                credential: Some(credential.to_owned()),
                provider: "nvidia".to_owned(),
                model: "vendor/native-model".to_owned(),
            };
            assert!(graph.bind(route.clone(), &shared_id));
            observation_map.bind(route, "rate_limit.requests", shared_id.clone());
        }
        let mut ledger = QuotaLedger::with_observation_map(graph, observation_map);
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
            ledger
                .graph()
                .resource(&shared_id)
                .map(|resource| resource.remaining),
            Some(crate::QuotaAmount::Finite(0))
        );
        let healthy = healthy_calls.lock();
        assert!(healthy.is_ok());
        if let Ok(calls) = healthy {
            assert!(!calls.iter().any(|call| call.starts_with("execute:")));
        }
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
