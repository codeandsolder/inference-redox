//! Shared quota ledger with short synchronous critical sections and cancellation-safe dispatch.
//!
//! Provider discovery/inference must never hold the ledger mutex. A dispatched request
//! owns a reservation lease that conservatively commits on cancellation or panic.
use std::sync::{Arc, Mutex};

use thiserror::Error;

use crate::{
    InferenceResponse, QuotaDemand, QuotaLedger, QuotaObservation, QuotaReservationId,
    QuotaRouteKey,
};

#[derive(Debug, Error)]
pub enum SharedQuotaError {
    #[error("shared quota ledger mutex was poisoned")]
    Poisoned,
    #[error("quota reservation accounting invariant failed")]
    Accounting,
}

/// Shared mutable quota state; operations on the ledger are synchronous and bounded.
#[derive(Debug, Clone)]
pub struct SharedQuotaLedger {
    inner: Arc<Mutex<QuotaLedger>>,
}

impl SharedQuotaLedger {
    #[must_use = "the returned shared quota wrapper owns the ledger"]
    pub fn new(ledger: QuotaLedger) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ledger)),
        }
    }

    /// Execute a short, synchronous atomic ledger transaction.
    ///
    /// Do not perform blocking I/O or network work in the closure. Its return value
    /// cannot borrow from the ledger, and the lock is released before returning.
    ///
    /// # Errors
    /// Fails if another transaction panicked while holding the ledger.
    pub fn with_mut<T>(
        &self,
        transaction: impl FnOnce(&mut QuotaLedger) -> T,
    ) -> Result<T, SharedQuotaError> {
        let mut ledger = self.inner.lock().map_err(|_| SharedQuotaError::Poisoned)?;
        Ok(transaction(&mut ledger))
    }

    /// Own a previously reserved route until explicit accounting or cancellation.
    pub(crate) fn lease(
        &self,
        reservation: QuotaReservationId,
        route: QuotaRouteKey,
        demand: QuotaDemand,
        now_unix: u64,
    ) -> QuotaReservationLease {
        QuotaReservationLease {
            ledger: self.clone(),
            reservation: Some(reservation),
            route,
            demand,
            now_unix,
        }
    }
}

/// In-flight reservation. Dropping the future conservatively charges the reservation,
/// since cancellation cannot prove whether the upstream has begun processing.
pub struct QuotaReservationLease {
    ledger: SharedQuotaLedger,
    reservation: Option<QuotaReservationId>,
    route: QuotaRouteKey,
    demand: QuotaDemand,
    now_unix: u64,
}

impl QuotaReservationLease {
    pub fn commit_success(&mut self, response: &InferenceResponse) -> Result<(), SharedQuotaError> {
        let actual = self.demand.with_actual_usage(response);
        self.settle(Some(actual), &response.quota_observations)
    }

    pub fn commit_ambiguous_failure(
        &mut self,
        observations: &[QuotaObservation],
    ) -> Result<(), SharedQuotaError> {
        self.settle(None, observations)
    }

    pub fn release_before_dispatch(&mut self) -> Result<(), SharedQuotaError> {
        let Some(reservation) = self.reservation else {
            return Err(SharedQuotaError::Accounting);
        };
        let released = self.ledger.with_mut(|ledger| ledger.release(reservation))?;
        if released {
            self.reservation = None;
            Ok(())
        } else {
            Err(SharedQuotaError::Accounting)
        }
    }

    fn settle(
        &mut self,
        actual: Option<QuotaDemand>,
        observations: &[QuotaObservation],
    ) -> Result<(), SharedQuotaError> {
        let Some(reservation) = self.reservation else {
            return Err(SharedQuotaError::Accounting);
        };
        let committed = self.ledger.with_mut(|ledger| {
            ledger.commit_with_observations(
                reservation,
                &self.route,
                actual,
                observations,
                self.now_unix,
            )
        })?;
        if committed.is_some() {
            self.reservation = None;
            Ok(())
        } else {
            Err(SharedQuotaError::Accounting)
        }
    }
}

impl Drop for QuotaReservationLease {
    fn drop(&mut self) {
        let Some(reservation) = self.reservation.take() else {
            return;
        };
        // This is a short, synchronous accounting transaction. Recover a poisoned
        // mutex to avoid leaving the reserved capacity permanently unavailable.
        let mut ledger = match self.ledger.inner.lock() {
            Ok(ledger) => ledger,
            Err(poisoned) => poisoned.into_inner(),
        };
        let _ = ledger.commit_with_observations(
            reservation,
            &self.route,
            Some(self.demand),
            &[],
            self.now_unix,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        QuotaAmount, QuotaGraph, QuotaMetric, QuotaResource, QuotaResourceId, QuotaScope,
        QuotaUnknownPolicy,
    };

    fn fixture() -> (SharedQuotaLedger, QuotaRouteKey, QuotaResourceId) {
        let route = QuotaRouteKey {
            source: "demo".to_owned(),
            credential: None,
            provider: "a".to_owned(),
            model: "model".to_owned(),
        };
        let resource = QuotaResource::finite(
            "demo/requests",
            QuotaScope::Account,
            QuotaMetric::Requests,
            10,
            2,
        );
        let id = resource.id.clone();
        let mut graph = QuotaGraph::new();
        graph.insert_resource(resource);
        assert!(graph.bind(route.clone(), &id));
        (SharedQuotaLedger::new(QuotaLedger::new(graph)), route, id)
    }

    #[test]
    fn cancellation_commits_conservative_request_usage_and_unreserves()
    -> Result<(), Box<dyn std::error::Error>> {
        let (ledger, route, resource_id) = fixture();
        let demand = QuotaDemand {
            requests: 1,
            concurrent_requests: 1,
            ..QuotaDemand::default()
        };
        let id = ledger
            .with_mut(|state| {
                state.reserve_with_policy(&route, demand, 10, QuotaUnknownPolicy::Reject)
            })?
            .map_err(|report| format!("{report:?}"))?;
        let lease = ledger.lease(id, route.clone(), demand, 10);
        let before = ledger.with_mut(|state| state.evaluate(&route, demand, 10))?;
        assert_eq!(before.checks[0].available, QuotaAmount::Finite(1));
        drop(lease);
        let after = ledger.with_mut(|state| state.evaluate(&route, demand, 10))?;
        assert_eq!(after.checks[0].available, QuotaAmount::Finite(1));
        let remaining = ledger.with_mut(|state| {
            state
                .graph()
                .resource(&resource_id)
                .map(|value| value.remaining)
        })?;
        assert_eq!(remaining, Some(QuotaAmount::Finite(1)));
        Ok(())
    }

    #[test]
    fn predispatch_release_preserves_capacity() -> Result<(), Box<dyn std::error::Error>> {
        let (ledger, route, resource_id) = fixture();
        let demand = QuotaDemand {
            requests: 1,
            ..QuotaDemand::default()
        };
        let reservation = ledger
            .with_mut(|state| {
                state.reserve_with_policy(&route, demand, 10, QuotaUnknownPolicy::Reject)
            })?
            .map_err(|report| format!("{report:?}"))?;
        let mut lease = ledger.lease(reservation, route, demand, 10);
        lease.release_before_dispatch()?;
        drop(lease);
        assert_eq!(
            ledger.with_mut(|state| state
                .graph()
                .resource(&resource_id)
                .map(|value| value.remaining))?,
            Some(QuotaAmount::Finite(2)),
        );
        Ok(())
    }
}
