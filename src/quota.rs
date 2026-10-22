use std::collections::{BTreeMap, BTreeSet};

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
    pub fn from_endpoint(endpoint: &Endpoint, credential: Option<String>) -> Self {
        Self {
            source: endpoint.source.clone(),
            credential,
            provider: endpoint.provider_key.clone(),
            model: endpoint.model.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaDemand {
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Provider-native integer credit units; adapters own scaling/conversion semantics.
    pub credits: Option<u64>,
    pub concurrent_requests: u64,
}

impl QuotaDemand {
    #[must_use]
    pub fn for_request(request: &InferenceRequest, prompt_tokens: u64) -> Self {
        Self {
            requests: 1,
            input_tokens: prompt_tokens,
            output_tokens: request
                .max_output_tokens
                .unwrap_or(request.expected_output_tokens),
            credits: None,
            concurrent_requests: 1,
        }
    }

    #[must_use]
    pub fn with_credits(mut self, credits: u64) -> Self {
        self.credits = Some(credits);
        self
    }

    fn amount(self, metric: QuotaMetric) -> Option<u64> {
        match metric {
            QuotaMetric::Requests => Some(self.requests),
            QuotaMetric::InputTokens => Some(self.input_tokens),
            QuotaMetric::OutputTokens => Some(self.output_tokens),
            QuotaMetric::TotalTokens => Some(self.input_tokens.saturating_add(self.output_tokens)),
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaAdmissionReport {
    pub route: QuotaRouteKey,
    pub admission: QuotaAdmission,
    pub checks: Vec<QuotaCheck>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct QuotaReservationId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActiveReservation {
    charges: BTreeMap<QuotaResourceId, u64>,
}

/// Local reservation/consumption ledger over provider observations.
#[derive(Debug, Clone, Default)]
pub struct QuotaLedger {
    graph: QuotaGraph,
    reserved: BTreeMap<QuotaResourceId, u64>,
    active: BTreeMap<QuotaReservationId, ActiveReservation>,
    next_reservation: u64,
}

impl QuotaLedger {
    #[must_use]
    pub fn new(graph: QuotaGraph) -> Self {
        Self {
            graph,
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
        let report = self.evaluate(route, demand, now_unix);
        if report.admission != QuotaAdmission::Admitted {
            return Err(Box::new(report));
        }
        let mut charges = BTreeMap::new();
        for check in &report.checks {
            if matches!(check.available, QuotaAmount::Finite(_))
                && let Some(amount) = check.demand
            {
                charges.insert(check.resource_id.clone(), amount);
                *self.reserved.entry(check.resource_id.clone()).or_default() += amount;
            }
        }
        let id = QuotaReservationId(self.next_reservation);
        self.next_reservation = self.next_reservation.saturating_add(1);
        self.active.insert(id, ActiveReservation { charges });
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
        for (resource_id, reserved_amount) in active.charges {
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
                .unwrap_or(reserved_amount);
            if let QuotaAmount::Finite(remaining) = resource.remaining {
                resource.remaining = QuotaAmount::Finite(remaining.saturating_sub(consumed));
            }
        }
        true
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
            input_tokens: 100,
            output_tokens: 100,
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
            input_tokens: 100,
            output_tokens: 100,
            concurrent_requests: 1,
            ..QuotaDemand::default()
        };
        let reservation = match ledger.reserve(&route, reserved, 100) {
            Ok(id) => id,
            Err(report) => return assert_eq!(report.admission, QuotaAdmission::Admitted),
        };
        let actual = QuotaDemand {
            input_tokens: 80,
            output_tokens: 20,
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
