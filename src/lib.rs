#![doc = "Provider-agnostic inference routing with shape-aware endpoint selection."]

mod model;
mod provider;
mod router;

pub mod providers;

pub use model::{
    Endpoint, EndpointCapabilities, EndpointStats, InferenceRequest, InferenceResponse,
    Percentiles, Pricing, ReasoningEffort, ReasoningSupport, RequestRequirements, ResponseSchema,
    RetryPolicy, RouteCandidate, RoutePlan, RoutingStrategy,
};
pub use provider::{InferenceProvider, ProviderError};
pub use router::{approximate_prompt_tokens, build_route_plan};
