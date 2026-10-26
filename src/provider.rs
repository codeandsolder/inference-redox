use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    Endpoint, InferenceRequest, InferenceResponse, QuotaLedger, QuotaObservation, QuotaRouteKey,
    RetryPolicy, RoutePlan, RoutingStrategy,
};

/// How far an inference attempt definitely progressed before failing.
///
/// `TransportUnknown` means bytes may already have left the client, so quota reservations cannot
/// be safely released as if nothing happened. `ResponseReceived` means the provider/gateway
/// definitely answered, although provider-specific rules still decide which buckets were charged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionFailurePhase {
    TransportUnknown,
    ResponseReceived,
}

/// Errors surfaced by provider adapters and routing.
#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("no eligible endpoint: {0}")]
    NoEligibleEndpoint(String),
    #[error("invalid inference request: {0}")]
    InvalidRequest(String),
    #[error("invalid routing strategy: {0}")]
    InvalidStrategy(String),
    #[error("provider catalog error: {0}")]
    Catalog(String),
    #[error("provider request error: {0}")]
    Request(String),
    #[error("provider execution error ({phase:?}): {detail}")]
    Execution {
        phase: ExecutionFailurePhase,
        detail: String,
        quota_observations: Vec<QuotaObservation>,
    },
    #[error("provider response error: {0}")]
    Response(String),
}

impl ProviderError {
    #[must_use]
    pub fn execution_phase(&self) -> Option<ExecutionFailurePhase> {
        match self {
            Self::Execution { phase, .. } => Some(*phase),
            _ => None,
        }
    }

    #[must_use]
    pub fn quota_observations(&self) -> &[QuotaObservation] {
        match self {
            Self::Execution {
                quota_observations, ..
            } => quota_observations,
            _ => &[],
        }
    }
}

/// Provider adapter. Each provider owns its wire grammar while exposing normalized routing data.
#[async_trait]
pub trait InferenceProvider: Send + Sync {
    async fn endpoints(&self, model: &str) -> Result<Vec<Endpoint>, ProviderError>;

    /// Declare logical quota resources for one executable route before quota admission.
    /// Providers that do not know their quota topology leave the route unbound/indeterminate.
    ///
    /// # Errors
    /// Returns an invalid-route/provider error when an adapter is asked to configure a route it does not own.
    fn configure_quota_route(
        &self,
        _route: &QuotaRouteKey,
        _ledger: &mut QuotaLedger,
    ) -> Result<(), ProviderError> {
        Ok(())
    }

    async fn execute_plan(
        &self,
        request: &InferenceRequest,
        plan: &RoutePlan,
        retry: RetryPolicy,
    ) -> Result<InferenceResponse, ProviderError>;

    async fn route_prompt(
        &self,
        model: &str,
        strategy: RoutingStrategy,
        prompt: &str,
        expected_output_tokens: u64,
        requirements: crate::RequestRequirements,
    ) -> Result<RoutePlan, ProviderError> {
        let mut request = InferenceRequest::new(model, prompt, expected_output_tokens);
        request.requirements = requirements;
        let endpoints = self.endpoints(model).await?;
        crate::build_route_plan(strategy, &request, endpoints)
    }

    /// Route and execute one endpoint at a time in ranked order. Provider adapters never own
    /// failover policy: every dispatch receives a one-candidate plan, matching `InferencePool`
    /// semantics and preventing gateways from silently choosing a different upstream.
    async fn infer(
        &self,
        strategy: RoutingStrategy,
        request: &InferenceRequest,
        retry: RetryPolicy,
    ) -> Result<(RoutePlan, InferenceResponse), ProviderError> {
        let endpoints = self.endpoints(&request.model).await?;
        let plan = crate::build_route_plan(strategy, request, endpoints)?;
        let mut attempted = 0_usize;
        let mut last_error = None;
        for candidate in plan.candidates.clone() {
            attempted += 1;
            let candidate_plan = RoutePlan {
                prompt_tokens: plan.prompt_tokens,
                expected_output_tokens: plan.expected_output_tokens,
                cheapest_cost_usd: candidate.expected_cost_usd,
                cost_ceiling_usd: candidate.expected_cost_usd,
                candidates: vec![candidate],
            };
            match self.execute_plan(request, &candidate_plan, retry).await {
                Ok(response) => {
                    match crate::structured_output::validate_inference_response(request, response) {
                        Ok(response) => return Ok((plan, response)),
                        Err(failure) => {
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
