use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use tokio::sync::{Mutex, MutexGuard};

use serde::{Deserialize, Serialize};

use crate::{Endpoint, InferenceRequest};

/// Stable logical identifier for one quota bucket. Never put an API key in this value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct QuotaResourceId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuotaScope {
    Credential,
    Account,
    Provider,
    ModelFamily,
    Model,
    Route,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuotaMetric {
    Requests,
    InputTokens,
    OutputTokens,
    TotalTokens,
    Credits,
    ConcurrentRequests,
}

/// One normalized quota/rate-limit measurement reported by an upstream surface.
///
/// `bucket` is a stable provider-local observation name such as `rate_limit.requests`; it does not
/// imply credential/account/model ownership. Ownership is assigned separately by
/// `QuotaObservationMap`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaObservation {
    pub bucket: String,
    pub metric: Option<QuotaMetric>,
    pub limit: Option<u64>,
    pub remaining: Option<u64>,
    pub reset_after_millis: Option<u64>,
}

/// `Unknown` and `Unlimited` are deliberately different states.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum QuotaAmount {
    #[default]
    Unknown,
    Unlimited,
    Finite(u64),
}

/// One independently enforced upstream quota bucket.
///
/// A passed reset timestamp makes a finite observation stale/unknown until refreshed; the core
/// never guesses that a provider's "reset" means an immediate full refill.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaResource {
    pub id: QuotaResourceId,
    pub scope: QuotaScope,
    pub metric: QuotaMetric,
    pub capacity: QuotaAmount,
    pub remaining: QuotaAmount,
    pub reset_at_unix: Option<u64>,
    pub observed_at_unix: Option<u64>,
}

impl QuotaResource {
    #[must_use]
    pub fn finite(
        id: impl Into<String>,
        scope: QuotaScope,
        metric: QuotaMetric,
        capacity: u64,
        remaining: u64,
    ) -> Self {
        Self {
            id: QuotaResourceId(id.into()),
            scope,
            metric,
            capacity: QuotaAmount::Finite(capacity),
            remaining: QuotaAmount::Finite(remaining),
            reset_at_unix: None,
            observed_at_unix: None,
        }
    }

    #[must_use]
    pub fn unknown(id: impl Into<String>, scope: QuotaScope, metric: QuotaMetric) -> Self {
        Self {
            id: QuotaResourceId(id.into()),
            scope,
            metric,
            capacity: QuotaAmount::Unknown,
            remaining: QuotaAmount::Unknown,
            reset_at_unix: None,
            observed_at_unix: None,
        }
    }

    fn is_stale(&self, now_unix: u64) -> bool {
        self.reset_at_unix.is_some_and(|reset| reset <= now_unix)
    }
}

/// Quota identity of one executable route.
///
/// `credential` is an opaque local label, never the secret itself. Separate credential routes can
/// bind to the same account-scoped resource, representing "N keys, one real quota" correctly.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QuotaRouteKey {
    pub source: String,
    pub credential: Option<String>,
    pub provider: String,
    pub model: String,
}

impl QuotaRouteKey {
    #[must_use]
    pub fn from_endpoint(endpoint: &Endpoint) -> Self {
        Self {
            source: endpoint.source.clone(),
            credential: endpoint.credential.clone(),
            provider: endpoint.provider_key.clone(),
            model: endpoint.model.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaDemand {
    pub requests: u64,
    /// Known prompt-token demand. `None` means the router only has an estimate, not a safe bound.
    pub input_tokens: Option<u64>,
    /// Known output-token upper bound. `None` means only an expectation is available.
    pub output_tokens: Option<u64>,
    /// Provider-native integer credit units; adapters own scaling/conversion semantics.
    pub credits: Option<u64>,
    pub concurrent_requests: u64,
}

impl QuotaDemand {
    #[must_use]
    pub fn for_request(request: &InferenceRequest) -> Self {
        Self {
            requests: 1,
            input_tokens: request.prompt_tokens,
            output_tokens: request.max_output_tokens,
            credits: None,
            concurrent_requests: 1,
        }
    }

    #[must_use]
    pub fn with_credits(mut self, credits: u64) -> Self {
        self.credits = Some(credits);
        self
    }

    /// Replace request estimates with provider-reported token counts where they are available.
    #[must_use]
    pub fn with_actual_usage(mut self, response: &crate::InferenceResponse) -> Self {
        if let Some(tokens) = response.usage.input_tokens {
            self.input_tokens = Some(tokens);
        }
        if let Some(tokens) = response.usage.output_tokens {
            self.output_tokens = Some(tokens);
        }
        self
    }

    fn amount(self, metric: QuotaMetric) -> Option<u64> {
        match metric {
            QuotaMetric::Requests => Some(self.requests),
            QuotaMetric::InputTokens => self.input_tokens,
            QuotaMetric::OutputTokens => self.output_tokens,
            QuotaMetric::TotalTokens => self
                .input_tokens
                .zip(self.output_tokens)
                .map(|(input, output)| input.saturating_add(output)),
            QuotaMetric::Credits => self.credits,
            QuotaMetric::ConcurrentRequests => Some(self.concurrent_requests),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuotaCheckStatus {
    Satisfied,
    Unknown,
    Exhausted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaCheck {
    pub resource_id: QuotaResourceId,
    pub scope: QuotaScope,
    pub metric: QuotaMetric,
    pub demand: Option<u64>,
    pub available: QuotaAmount,
    pub stale: bool,
    pub status: QuotaCheckStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuotaAdmission {
    Admitted,
    Indeterminate,
    Rejected,
}

/// How routing treats routes whose quota is not currently knowable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuotaUnknownPolicy {
    /// Preserve legacy/experimental behavior while still excluding known-exhausted routes.
    #[default]
    Allow,
    /// Require every bound quota resource to have known sufficient capacity.
    Reject,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaAdmissionReport {
    pub route: QuotaRouteKey,
    pub admission: QuotaAdmission,
    pub checks: Vec<QuotaCheck>,
}

/// Ranked route plan plus the quota evidence considered before the cost floor was computed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuotaAwareRoutePlan {
    pub plan: crate::RoutePlan,
    pub quota_reports: Vec<QuotaAdmissionReport>,
}

/// Provider-independent quota topology. Sharing one resource ID between routes expresses a
/// provider/account/model-family pool without multiplying it by key or model count.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaGraph {
    resources: BTreeMap<QuotaResourceId, QuotaResource>,
    bindings: BTreeMap<QuotaRouteKey, BTreeSet<QuotaResourceId>>,
}

impl QuotaGraph {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_resource(&mut self, resource: QuotaResource) {
        self.resources.insert(resource.id.clone(), resource);
    }

    #[must_use]
    pub fn resource(&self, id: &QuotaResourceId) -> Option<&QuotaResource> {
        self.resources.get(id)
    }

    pub fn resource_mut(&mut self, id: &QuotaResourceId) -> Option<&mut QuotaResource> {
        self.resources.get_mut(id)
    }

    /// Bind to an already-declared resource. Missing IDs fail instead of becoming fake capacity.
    pub fn bind(&mut self, route: QuotaRouteKey, resource_id: &QuotaResourceId) -> bool {
        if !self.resources.contains_key(resource_id) {
            return false;
        }
        self.bindings
            .entry(route)
            .or_default()
            .insert(resource_id.clone());
        true
    }

    fn bound_resources(&self, route: &QuotaRouteKey) -> Vec<&QuotaResource> {
        self.bindings
            .get(route)
            .into_iter()
            .flatten()
            .filter_map(|id| self.resources.get(id))
            .collect()
    }
}

/// Explicit mapping from provider-local observation buckets to logical quota resources.
///
/// Bindings are route-specific, which lets two credential instances map the same observed header
/// to either one shared account resource or two independent resources without changing adapters.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaObservationMap {
    bindings: BTreeMap<(QuotaRouteKey, String), QuotaResourceId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuotaObservationApplyStatus {
    Applied,
    Unbound,
    MissingResource,
    MetricMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaObservationApplyResult {
    pub bucket: String,
    pub status: QuotaObservationApplyStatus,
    pub resource_id: Option<QuotaResourceId>,
    /// True only when this observation replaced the post-request remaining value.
    pub remaining_refreshed: bool,
}

impl QuotaObservationMap {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bind(
        &mut self,
        route: QuotaRouteKey,
        bucket: impl Into<String>,
        resource_id: QuotaResourceId,
    ) {
        self.bindings.insert((route, bucket.into()), resource_id);
    }

    pub fn apply(
        &self,
        graph: &mut QuotaGraph,
        route: &QuotaRouteKey,
        observations: &[QuotaObservation],
        observed_at_unix: u64,
    ) -> Vec<QuotaObservationApplyResult> {
        observations
            .iter()
            .map(|observation| {
                let key = (route.clone(), observation.bucket.clone());
                let Some(resource_id) = self.bindings.get(&key).cloned() else {
                    return QuotaObservationApplyResult {
                        bucket: observation.bucket.clone(),
                        status: QuotaObservationApplyStatus::Unbound,
                        resource_id: None,
                        remaining_refreshed: false,
                    };
                };
                let Some(resource) = graph.resource_mut(&resource_id) else {
                    return QuotaObservationApplyResult {
                        bucket: observation.bucket.clone(),
                        status: QuotaObservationApplyStatus::MissingResource,
                        resource_id: Some(resource_id),
                        remaining_refreshed: false,
                    };
                };
                if observation
                    .metric
                    .is_some_and(|metric| metric != resource.metric)
                {
                    return QuotaObservationApplyResult {
                        bucket: observation.bucket.clone(),
                        status: QuotaObservationApplyStatus::MetricMismatch,
                        resource_id: Some(resource_id),
                        remaining_refreshed: false,
                    };
                }
                if let Some(limit) = observation.limit {
                    resource.capacity = QuotaAmount::Finite(limit);
                }
                if let Some(remaining) = observation.remaining {
                    resource.remaining = QuotaAmount::Finite(remaining);
                }
                if let Some(reset_after_millis) = observation.reset_after_millis {
                    let reset_after_seconds = reset_after_millis.saturating_add(999) / 1_000;
                    resource.reset_at_unix =
                        Some(observed_at_unix.saturating_add(reset_after_seconds));
                }
                resource.observed_at_unix = Some(observed_at_unix);
                QuotaObservationApplyResult {
                    bucket: observation.bucket.clone(),
                    status: QuotaObservationApplyStatus::Applied,
                    resource_id: Some(resource_id),
                    remaining_refreshed: observation.remaining.is_some(),
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct QuotaReservationId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ActiveReservation {
    charges: BTreeMap<QuotaResourceId, u64>,
    resources: BTreeSet<QuotaResourceId>,
}

/// Local reservation/consumption ledger over provider observations.
#[derive(Debug, Clone, Default, Serialize)]
pub struct QuotaLedger {
    graph: QuotaGraph,
    observation_map: QuotaObservationMap,
    reserved: BTreeMap<QuotaResourceId, u64>,
    active: BTreeMap<QuotaReservationId, ActiveReservation>,
    next_reservation: u64,
}

/// Concurrency-safe quota state shared by asynchronous routing/execution tasks.
///
/// The mutex protects only short ledger mutations. Provider I/O must never run while this lock is
/// held; `InferencePool::infer_with_shared_quota` reserves before dispatch and settles afterwards.
#[derive(Debug, Clone)]
pub struct SharedQuotaLedger {
    inner: Arc<Mutex<QuotaLedger>>,
}

impl SharedQuotaLedger {
    #[must_use]
    pub fn new(ledger: QuotaLedger) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ledger)),
        }
    }

    /// Clone the current ledger state for diagnostics or persistence.
    pub async fn snapshot(&self) -> QuotaLedger {
        self.inner.lock().await.clone()
    }

    pub(crate) async fn lock(&self) -> MutexGuard<'_, QuotaLedger> {
        self.inner.lock().await
    }
}

impl Default for SharedQuotaLedger {
    fn default() -> Self {
        Self::new(QuotaLedger::default())
    }
}

impl QuotaLedger {
    #[must_use]
    pub fn new(graph: QuotaGraph) -> Self {
        Self {
            graph,
            observation_map: QuotaObservationMap::new(),
            reserved: BTreeMap::new(),
            active: BTreeMap::new(),
            next_reservation: 1,
        }
    }

    #[must_use]
    pub fn with_observation_map(graph: QuotaGraph, observation_map: QuotaObservationMap) -> Self {
        Self {
            graph,
            observation_map,
            reserved: BTreeMap::new(),
            active: BTreeMap::new(),
            next_reservation: 1,
        }
    }

    #[must_use]
    pub fn graph(&self) -> &QuotaGraph {
        &self.graph
    }

    pub fn graph_mut(&mut self) -> &mut QuotaGraph {
        &mut self.graph
    }

    #[must_use]
    pub fn observation_map(&self) -> &QuotaObservationMap {
        &self.observation_map
    }

    pub fn observation_map_mut(&mut self) -> &mut QuotaObservationMap {
        &mut self.observation_map
    }

    #[must_use]
    pub fn evaluate(
        &self,
        route: &QuotaRouteKey,
        demand: QuotaDemand,
        now_unix: u64,
    ) -> QuotaAdmissionReport {
        let resources = self.graph.bound_resources(route);
        if resources.is_empty() {
            return QuotaAdmissionReport {
                route: route.clone(),
                admission: QuotaAdmission::Indeterminate,
                checks: Vec::new(),
            };
        }
        let mut checks = Vec::with_capacity(resources.len());
        for resource in resources {
            let stale = resource.is_stale(now_unix);
            let reserved = self.reserved.get(&resource.id).copied().unwrap_or(0);
            let demand_amount = demand.amount(resource.metric);
            let available = if stale {
                QuotaAmount::Unknown
            } else {
                match resource.remaining {
                    QuotaAmount::Finite(remaining) => {
                        QuotaAmount::Finite(remaining.saturating_sub(reserved))
                    }
                    other => other,
                }
            };
            let status = match (available, demand_amount) {
                (QuotaAmount::Unlimited, _) => QuotaCheckStatus::Satisfied,
                (QuotaAmount::Unknown, _) | (QuotaAmount::Finite(_), None) => {
                    QuotaCheckStatus::Unknown
                }
                (QuotaAmount::Finite(left), Some(needed)) if needed <= left => {
                    QuotaCheckStatus::Satisfied
                }
                (QuotaAmount::Finite(_), Some(_)) => QuotaCheckStatus::Exhausted,
            };
            checks.push(QuotaCheck {
                resource_id: resource.id.clone(),
                scope: resource.scope,
                metric: resource.metric,
                demand: demand_amount,
                available,
                stale,
                status,
            });
        }
        let admission = if checks
            .iter()
            .any(|check| check.status == QuotaCheckStatus::Exhausted)
        {
            QuotaAdmission::Rejected
        } else if checks
            .iter()
            .any(|check| check.status == QuotaCheckStatus::Unknown)
        {
            QuotaAdmission::Indeterminate
        } else {
            QuotaAdmission::Admitted
        };
        QuotaAdmissionReport {
            route: route.clone(),
            admission,
            checks,
        }
    }

    /// Reserve finite headroom. Unknown/untracked resources fail closed.
    ///
    /// # Errors
    /// Returns the explainable admission report when any bound quota is exhausted or unknown.
    pub fn reserve(
        &mut self,
        route: &QuotaRouteKey,
        demand: QuotaDemand,
        now_unix: u64,
    ) -> Result<QuotaReservationId, Box<QuotaAdmissionReport>> {
        self.reserve_with_policy(route, demand, now_unix, QuotaUnknownPolicy::Reject)
    }

    /// Reserve every known finite demand while optionally allowing unknown buckets to proceed.
    /// Unknown bound resources are retained in the reservation so post-call accounting can update
    /// them from actual usage or invalidate stale finite observations after ambiguous failures.
    ///
    /// # Errors
    /// Returns the explainable admission report on known exhaustion, or on any unknown check when
    /// `unknown_policy` is `Reject`.
    pub fn reserve_with_policy(
        &mut self,
        route: &QuotaRouteKey,
        demand: QuotaDemand,
        now_unix: u64,
        unknown_policy: QuotaUnknownPolicy,
    ) -> Result<QuotaReservationId, Box<QuotaAdmissionReport>> {
        let report = self.evaluate(route, demand, now_unix);
        let rejected = report.admission == QuotaAdmission::Rejected
            || (report.admission == QuotaAdmission::Indeterminate
                && unknown_policy == QuotaUnknownPolicy::Reject);
        if rejected {
            return Err(Box::new(report));
        }
        let resources = report
            .checks
            .iter()
            .map(|check| check.resource_id.clone())
            .collect();
        let mut charges = BTreeMap::new();
        for check in &report.checks {
            if check.status == QuotaCheckStatus::Satisfied
                && matches!(check.available, QuotaAmount::Finite(_))
                && let Some(amount) = check.demand
            {
                charges.insert(check.resource_id.clone(), amount);
                *self.reserved.entry(check.resource_id.clone()).or_default() += amount;
            }
        }
        let id = QuotaReservationId(self.next_reservation);
        self.next_reservation = self.next_reservation.saturating_add(1);
        self.active
            .insert(id, ActiveReservation { charges, resources });
        Ok(id)
    }

    /// Release without consuming quota, e.g. if no upstream request was accepted.
    pub fn release(&mut self, reservation: QuotaReservationId) -> bool {
        let Some(active) = self.active.remove(&reservation) else {
            return false;
        };
        self.unreserve(&active.charges);
        true
    }

    /// Commit an accepted request. Provider-reported actual usage may refine the reservation.
    pub fn commit(
        &mut self,
        reservation: QuotaReservationId,
        actual: Option<QuotaDemand>,
        now_unix: u64,
    ) -> bool {
        let Some(active) = self.active.remove(&reservation) else {
            return false;
        };
        self.unreserve(&active.charges);
        self.consume_unobserved(active, actual, now_unix, &BTreeSet::new());
        true
    }

    /// Commit an accepted request while applying provider-reported post-request quota state.
    /// Mapped `remaining` snapshots are authoritative for that resource and therefore suppress
    /// local consumption for the same request; capacity/reset-only observations do not.
    pub fn commit_with_observations(
        &mut self,
        reservation: QuotaReservationId,
        route: &QuotaRouteKey,
        actual: Option<QuotaDemand>,
        observations: &[QuotaObservation],
        now_unix: u64,
    ) -> Option<Vec<QuotaObservationApplyResult>> {
        let active = self.active.remove(&reservation)?;
        self.unreserve(&active.charges);
        let results = self
            .observation_map
            .apply(&mut self.graph, route, observations, now_unix);
        let refreshed_remaining = results
            .iter()
            .filter(|result| {
                result.status == QuotaObservationApplyStatus::Applied && result.remaining_refreshed
            })
            .filter_map(|result| result.resource_id.clone())
            .collect();
        self.consume_unobserved(active, actual, now_unix, &refreshed_remaining);
        Some(results)
    }

    fn consume_unobserved(
        &mut self,
        active: ActiveReservation,
        actual: Option<QuotaDemand>,
        now_unix: u64,
        refreshed_remaining: &BTreeSet<QuotaResourceId>,
    ) {
        for resource_id in active.resources {
            if refreshed_remaining.contains(&resource_id) {
                continue;
            }
            let reserved_amount = active.charges.get(&resource_id).copied();
            let Some(resource) = self.graph.resource_mut(&resource_id) else {
                continue;
            };
            if resource.metric == QuotaMetric::ConcurrentRequests {
                continue;
            }
            if resource.is_stale(now_unix) {
                resource.remaining = QuotaAmount::Unknown;
                continue;
            }
            let consumed = actual
                .and_then(|demand| demand.amount(resource.metric))
                .or(reserved_amount);
            match (resource.remaining, consumed) {
                (QuotaAmount::Finite(remaining), Some(consumed)) => {
                    resource.remaining = QuotaAmount::Finite(remaining.saturating_sub(consumed));
                }
                (QuotaAmount::Finite(_), None) => {
                    resource.remaining = QuotaAmount::Unknown;
                }
                (QuotaAmount::Unknown | QuotaAmount::Unlimited, _) => {}
            }
        }
    }

    fn unreserve(&mut self, charges: &BTreeMap<QuotaResourceId, u64>) {
        for (resource_id, amount) in charges {
            let Some(total) = self.reserved.get_mut(resource_id) else {
                continue;
            };
            *total = total.saturating_sub(*amount);
            if *total == 0 {
                self.reserved.remove(resource_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route(credential: &str) -> QuotaRouteKey {
        QuotaRouteKey {
            source: "provider".to_owned(),
            credential: Some(credential.to_owned()),
            provider: "provider".to_owned(),
            model: "model".to_owned(),
        }
    }

    #[test]
    fn observation_binding_separates_measurement_from_ownership() {
        let route_a = route("key-a");
        let route_b = route("key-b");
        let resource = QuotaResource::finite(
            "shared-account/rpd",
            QuotaScope::Account,
            QuotaMetric::Requests,
            100,
            100,
        );
        let resource_id = resource.id.clone();
        let mut graph = QuotaGraph::new();
        graph.insert_resource(resource);
        let mut map = QuotaObservationMap::new();
        map.bind(route_a.clone(), "rate_limit.requests", resource_id.clone());
        map.bind(route_b, "rate_limit.requests", resource_id.clone());
        let result = map.apply(
            &mut graph,
            &route_a,
            &[QuotaObservation {
                bucket: "rate_limit.requests".to_owned(),
                metric: Some(QuotaMetric::Requests),
                limit: Some(1_000),
                remaining: Some(731),
                reset_after_millis: Some(2_500),
            }],
            100,
        );
        assert_eq!(result[0].status, QuotaObservationApplyStatus::Applied);
        let updated = graph.resource(&resource_id);
        assert_eq!(
            updated.map(|resource| resource.capacity),
            Some(QuotaAmount::Finite(1_000))
        );
        assert_eq!(
            updated.map(|resource| resource.remaining),
            Some(QuotaAmount::Finite(731))
        );
        assert_eq!(
            updated.and_then(|resource| resource.reset_at_unix),
            Some(103)
        );
        assert_eq!(
            updated.and_then(|resource| resource.observed_at_unix),
            Some(100)
        );
    }

    #[test]
    fn unbound_or_mismatched_observation_does_not_mutate_resource() {
        let route = route("key");
        let resource = QuotaResource::finite(
            "account/rpd",
            QuotaScope::Account,
            QuotaMetric::Requests,
            100,
            100,
        );
        let resource_id = resource.id.clone();
        let mut graph = QuotaGraph::new();
        graph.insert_resource(resource);
        let mut map = QuotaObservationMap::new();
        map.bind(route.clone(), "rate_limit.tokens", resource_id.clone());
        let mismatch = map.apply(
            &mut graph,
            &route,
            &[QuotaObservation {
                bucket: "rate_limit.tokens".to_owned(),
                metric: Some(QuotaMetric::TotalTokens),
                limit: Some(1),
                remaining: Some(0),
                reset_after_millis: None,
            }],
            100,
        );
        assert_eq!(
            mismatch[0].status,
            QuotaObservationApplyStatus::MetricMismatch
        );
        assert_eq!(
            graph
                .resource(&resource_id)
                .map(|resource| resource.remaining),
            Some(QuotaAmount::Finite(100))
        );
        let unbound = map.apply(
            &mut graph,
            &route,
            &[QuotaObservation {
                bucket: "other".to_owned(),
                metric: None,
                limit: None,
                remaining: Some(0),
                reset_after_millis: None,
            }],
            100,
        );
        assert_eq!(unbound[0].status, QuotaObservationApplyStatus::Unbound);
    }

    #[test]
    fn observed_remaining_is_not_locally_subtracted_twice() {
        let route = route("key");
        let requests = QuotaResource::finite(
            "provider/rpd",
            QuotaScope::Account,
            QuotaMetric::Requests,
            100,
            100,
        );
        let request_id = requests.id.clone();
        let tokens = QuotaResource::finite(
            "provider/tpm",
            QuotaScope::Account,
            QuotaMetric::TotalTokens,
            1_000,
            1_000,
        );
        let token_id = tokens.id.clone();
        let mut graph = QuotaGraph::new();
        graph.insert_resource(requests);
        graph.insert_resource(tokens);
        assert!(graph.bind(route.clone(), &request_id));
        assert!(graph.bind(route.clone(), &token_id));
        let mut observation_map = QuotaObservationMap::new();
        observation_map.bind(route.clone(), "rate_limit.requests", request_id.clone());
        observation_map.bind(route.clone(), "rate_limit.tokens", token_id.clone());
        let mut ledger = QuotaLedger::with_observation_map(graph, observation_map);
        let demand = QuotaDemand {
            requests: 1,
            input_tokens: Some(10),
            output_tokens: Some(20),
            concurrent_requests: 1,
            ..QuotaDemand::default()
        };
        let reservation = match ledger.reserve(&route, demand, 100) {
            Ok(id) => id,
            Err(report) => return assert_eq!(report.admission, QuotaAdmission::Admitted),
        };
        let observations = [
            QuotaObservation {
                bucket: "rate_limit.requests".to_owned(),
                metric: Some(QuotaMetric::Requests),
                limit: Some(100),
                remaining: Some(90),
                reset_after_millis: None,
            },
            QuotaObservation {
                bucket: "rate_limit.tokens".to_owned(),
                metric: Some(QuotaMetric::TotalTokens),
                limit: Some(2_000),
                remaining: None,
                reset_after_millis: Some(1_000),
            },
        ];
        let results =
            ledger.commit_with_observations(reservation, &route, Some(demand), &observations, 100);
        assert!(results.is_some(), "reservation unexpectedly disappeared");
        if let Some(results) = results {
            assert!(results[0].remaining_refreshed);
            assert!(!results[1].remaining_refreshed);
        }
        assert_eq!(
            ledger.graph().resource(&request_id).map(|r| r.remaining),
            Some(QuotaAmount::Finite(90))
        );
        assert_eq!(
            ledger.graph().resource(&token_id).map(|r| r.capacity),
            Some(QuotaAmount::Finite(2_000))
        );
        assert_eq!(
            ledger.graph().resource(&token_id).map(|r| r.remaining),
            Some(QuotaAmount::Finite(970))
        );
    }

    #[test]
    fn endpoint_credential_becomes_quota_route_identity() {
        let endpoint = Endpoint {
            source: "nvidia".to_owned(),
            credential: Some("account-a".to_owned()),
            provider_key: "nvidia".to_owned(),
            id: "nvidia".to_owned(),
            provider: "NVIDIA NIM".to_owned(),
            model: "vendor/model".to_owned(),
            route_class: crate::ModelRouteClass::Free,
            display_name: "model".to_owned(),
            quantization: None,
            status: 0,
            context_length: None,
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: crate::Pricing::default(),
            stats: crate::EndpointStats::default(),
            capabilities: crate::EndpointCapabilities::default(),
        };
        let key = QuotaRouteKey::from_endpoint(&endpoint);
        assert_eq!(key.source, "nvidia");
        assert_eq!(key.credential.as_deref(), Some("account-a"));
        assert_eq!(key.provider, "nvidia");
        assert_eq!(key.model, "vendor/model");
    }

    #[test]
    fn shared_account_bucket_means_two_keys_do_not_double_capacity() {
        let route_a = route("key-a");
        let route_b = route("key-b");
        let account = QuotaResource::finite(
            "account/rpd",
            QuotaScope::Account,
            QuotaMetric::Requests,
            1,
            1,
        );
        let account_id = account.id.clone();
        let mut graph = QuotaGraph::new();
        graph.insert_resource(account);
        assert!(graph.bind(route_a.clone(), &account_id));
        assert!(graph.bind(route_b.clone(), &account_id));
        let mut ledger = QuotaLedger::new(graph);
        let demand = QuotaDemand {
            requests: 1,
            ..QuotaDemand::default()
        };
        let reservation = match ledger.reserve(&route_a, demand, 100) {
            Ok(id) => id,
            Err(report) => return assert_eq!(report.admission, QuotaAdmission::Admitted),
        };
        assert!(ledger.commit(reservation, Some(demand), 100));
        assert_eq!(
            ledger.evaluate(&route_b, demand, 100).admission,
            QuotaAdmission::Rejected
        );
    }

    #[test]
    fn unknown_credit_sanity_valve_is_not_infinite() {
        let route = route("nim-key");
        let requests = QuotaResource::finite(
            "nim/rpm",
            QuotaScope::Account,
            QuotaMetric::Requests,
            40,
            40,
        );
        let requests_id = requests.id.clone();
        let credits =
            QuotaResource::unknown("nim/credits", QuotaScope::Account, QuotaMetric::Credits);
        let credits_id = credits.id.clone();
        let mut graph = QuotaGraph::new();
        graph.insert_resource(requests);
        graph.insert_resource(credits);
        assert!(graph.bind(route.clone(), &requests_id));
        assert!(graph.bind(route.clone(), &credits_id));
        let report = QuotaLedger::new(graph).evaluate(
            &route,
            QuotaDemand {
                requests: 1,
                credits: Some(1),
                ..QuotaDemand::default()
            },
            100,
        );
        assert_eq!(report.admission, QuotaAdmission::Indeterminate);
    }

    #[test]
    fn token_quota_requires_explicit_bounds_not_router_estimates() {
        let route = route("key");
        let resource = QuotaResource::finite(
            "provider/tpm",
            QuotaScope::Account,
            QuotaMetric::TotalTokens,
            1_000,
            1_000,
        );
        let id = resource.id.clone();
        let mut graph = QuotaGraph::new();
        graph.insert_resource(resource);
        assert!(graph.bind(route.clone(), &id));
        let ledger = QuotaLedger::new(graph);

        let request = InferenceRequest::new("model", "tiny prompt", 8);
        let unknown = QuotaDemand::for_request(&request);
        assert_eq!(
            ledger.evaluate(&route, unknown, 100).admission,
            QuotaAdmission::Indeterminate
        );

        let mut bounded = request;
        bounded.prompt_tokens = Some(21);
        bounded.max_output_tokens = Some(256);
        assert_eq!(
            ledger
                .evaluate(&route, QuotaDemand::for_request(&bounded), 100)
                .admission,
            QuotaAdmission::Admitted
        );
    }

    #[test]
    fn permissive_reservation_tracks_unknown_bucket_for_post_call_accounting() {
        let route = route("key");
        let requests = QuotaResource::finite(
            "provider/rpd",
            QuotaScope::Account,
            QuotaMetric::Requests,
            10,
            10,
        );
        let requests_id = requests.id.clone();
        let tokens = QuotaResource::finite(
            "provider/tpd",
            QuotaScope::Account,
            QuotaMetric::TotalTokens,
            1_000,
            1_000,
        );
        let tokens_id = tokens.id.clone();
        let mut graph = QuotaGraph::new();
        graph.insert_resource(requests);
        graph.insert_resource(tokens);
        assert!(graph.bind(route.clone(), &requests_id));
        assert!(graph.bind(route.clone(), &tokens_id));
        let mut ledger = QuotaLedger::new(graph);
        let demand = QuotaDemand {
            requests: 1,
            concurrent_requests: 1,
            ..QuotaDemand::default()
        };
        let reservation =
            match ledger.reserve_with_policy(&route, demand, 100, QuotaUnknownPolicy::Allow) {
                Ok(id) => id,
                Err(report) => return assert_eq!(report.admission, QuotaAdmission::Admitted),
            };
        assert!(ledger.commit(reservation, None, 100));
        assert_eq!(
            ledger.graph().resource(&requests_id).map(|r| r.remaining),
            Some(QuotaAmount::Finite(9))
        );
        assert_eq!(
            ledger.graph().resource(&tokens_id).map(|r| r.remaining),
            Some(QuotaAmount::Unknown)
        );
    }

    #[test]
    fn successful_actual_usage_updates_previously_unbounded_token_demand() {
        let route = route("key");
        let tokens = QuotaResource::finite(
            "provider/tpd",
            QuotaScope::Account,
            QuotaMetric::TotalTokens,
            1_000,
            1_000,
        );
        let tokens_id = tokens.id.clone();
        let mut graph = QuotaGraph::new();
        graph.insert_resource(tokens);
        assert!(graph.bind(route.clone(), &tokens_id));
        let mut ledger = QuotaLedger::new(graph);
        let request = InferenceRequest::new("model", "prompt", 8);
        let demand = QuotaDemand::for_request(&request);
        let reservation =
            match ledger.reserve_with_policy(&route, demand, 100, QuotaUnknownPolicy::Allow) {
                Ok(id) => id,
                Err(report) => return assert_eq!(report.admission, QuotaAdmission::Admitted),
            };
        let response = crate::InferenceResponse {
            content: "ok".to_owned(),
            model: None,
            provider: None,
            usage_cost_usd: None,
            usage: crate::TokenUsage {
                input_tokens: Some(21),
                output_tokens: Some(60),
                total_tokens: Some(81),
                reasoning_tokens: Some(56),
            },
            quota_observations: Vec::new(),
            raw: serde_json::Value::Null,
        };
        assert!(ledger.commit(reservation, Some(demand.with_actual_usage(&response)), 100));
        assert_eq!(
            ledger.graph().resource(&tokens_id).map(|r| r.remaining),
            Some(QuotaAmount::Finite(919))
        );
    }

    #[test]
    fn passed_reset_timestamp_makes_observation_unknown() {
        let route = route("key");
        let mut resource = QuotaResource::finite(
            "provider/tpm",
            QuotaScope::Account,
            QuotaMetric::TotalTokens,
            10_000,
            9_000,
        );
        resource.reset_at_unix = Some(200);
        let id = resource.id.clone();
        let mut graph = QuotaGraph::new();
        graph.insert_resource(resource);
        assert!(graph.bind(route.clone(), &id));
        let demand = QuotaDemand {
            input_tokens: Some(100),
            output_tokens: Some(100),
            ..QuotaDemand::default()
        };
        let ledger = QuotaLedger::new(graph);
        assert_eq!(
            ledger.evaluate(&route, demand, 199).admission,
            QuotaAdmission::Admitted
        );
        assert_eq!(
            ledger.evaluate(&route, demand, 200).admission,
            QuotaAdmission::Indeterminate
        );
    }

    #[test]
    fn releasing_one_reservation_preserves_other_held_capacity() {
        let route = route("key");
        let resource = QuotaResource::finite(
            "provider/concurrency",
            QuotaScope::Account,
            QuotaMetric::ConcurrentRequests,
            2,
            2,
        );
        let id = resource.id.clone();
        let mut graph = QuotaGraph::new();
        graph.insert_resource(resource);
        assert!(graph.bind(route.clone(), &id));
        let demand = QuotaDemand {
            concurrent_requests: 1,
            ..QuotaDemand::default()
        };
        let mut ledger = QuotaLedger::new(graph);
        let first = match ledger.reserve(&route, demand, 100) {
            Ok(id) => id,
            Err(report) => return assert_eq!(report.admission, QuotaAdmission::Admitted),
        };
        let _second = match ledger.reserve(&route, demand, 100) {
            Ok(id) => id,
            Err(report) => return assert_eq!(report.admission, QuotaAdmission::Admitted),
        };
        assert!(ledger.release(first));
        assert_eq!(
            ledger
                .evaluate(
                    &route,
                    QuotaDemand {
                        concurrent_requests: 2,
                        ..QuotaDemand::default()
                    },
                    100,
                )
                .admission,
            QuotaAdmission::Rejected
        );
        assert_eq!(
            ledger.evaluate(&route, demand, 100).admission,
            QuotaAdmission::Admitted
        );
    }

    #[test]
    fn concurrency_reservation_blocks_until_release() {
        let route = route("key");
        let resource = QuotaResource::finite(
            "provider/concurrency",
            QuotaScope::Account,
            QuotaMetric::ConcurrentRequests,
            1,
            1,
        );
        let id = resource.id.clone();
        let mut graph = QuotaGraph::new();
        graph.insert_resource(resource);
        assert!(graph.bind(route.clone(), &id));
        let demand = QuotaDemand {
            concurrent_requests: 1,
            ..QuotaDemand::default()
        };
        let mut ledger = QuotaLedger::new(graph);
        let reservation = match ledger.reserve(&route, demand, 100) {
            Ok(id) => id,
            Err(report) => return assert_eq!(report.admission, QuotaAdmission::Admitted),
        };
        assert_eq!(
            ledger.evaluate(&route, demand, 100).admission,
            QuotaAdmission::Rejected
        );
        assert!(ledger.release(reservation));
        assert_eq!(
            ledger.evaluate(&route, demand, 100).admission,
            QuotaAdmission::Admitted
        );
    }

    #[test]
    fn commit_uses_actual_tokens_but_not_concurrency() {
        let route = route("key");
        let tokens = QuotaResource::finite(
            "provider/tpd",
            QuotaScope::Account,
            QuotaMetric::TotalTokens,
            1_000,
            1_000,
        );
        let tokens_id = tokens.id.clone();
        let concurrent = QuotaResource::finite(
            "provider/concurrency",
            QuotaScope::Account,
            QuotaMetric::ConcurrentRequests,
            1,
            1,
        );
        let concurrent_id = concurrent.id.clone();
        let mut graph = QuotaGraph::new();
        graph.insert_resource(tokens);
        graph.insert_resource(concurrent);
        assert!(graph.bind(route.clone(), &tokens_id));
        assert!(graph.bind(route.clone(), &concurrent_id));
        let mut ledger = QuotaLedger::new(graph);
        let reserved = QuotaDemand {
            input_tokens: Some(100),
            output_tokens: Some(100),
            concurrent_requests: 1,
            ..QuotaDemand::default()
        };
        let reservation = match ledger.reserve(&route, reserved, 100) {
            Ok(id) => id,
            Err(report) => return assert_eq!(report.admission, QuotaAdmission::Admitted),
        };
        let actual = QuotaDemand {
            input_tokens: Some(80),
            output_tokens: Some(20),
            concurrent_requests: 1,
            ..QuotaDemand::default()
        };
        assert!(ledger.commit(reservation, Some(actual), 100));
        assert_eq!(
            ledger.graph().resource(&tokens_id).map(|r| r.remaining),
            Some(QuotaAmount::Finite(900))
        );
        assert_eq!(
            ledger
                .evaluate(
                    &route,
                    QuotaDemand {
                        input_tokens: Some(0),
                        output_tokens: Some(0),
                        concurrent_requests: 1,
                        ..QuotaDemand::default()
                    },
                    100,
                )
                .admission,
            QuotaAdmission::Admitted
        );
    }
}
