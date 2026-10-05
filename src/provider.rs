use async_trait::async_trait;
use thiserror::Error;

use crate::{
    Endpoint, InferenceRequest, InferenceResponse, RetryPolicy, RoutePlan, RoutingStrategy,
};

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
    #[error("provider response error: {0}")]
    Response(String),
}

/// Provider adapter. Each provider owns its wire grammar while exposing normalized routing data.
#[async_trait]
pub trait InferenceProvider: Send + Sync {
    async fn endpoints(&self, model: &str) -> Result<Vec<Endpoint>, ProviderError>;

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

    async fn infer(
        &self,
        strategy: RoutingStrategy,
        request: &InferenceRequest,
        retry: RetryPolicy,
    ) -> Result<(RoutePlan, InferenceResponse), ProviderError> {
        let endpoints = self.endpoints(&request.model).await?;
        let plan = crate::build_route_plan(strategy, request, endpoints)?;
        let response = self.execute_plan(request, &plan, retry).await?;
        Ok((plan, response))
    }
}
