#![doc = "Provider-agnostic inference routing with shape-aware endpoint selection."]

mod catalog;
mod model;
mod provider;
mod router;

pub mod providers;

pub use catalog::{
    ArtificialAnalysisBenchmarks, DesignArenaBenchmark, ModelArchitecture, ModelBenchmarks,
    ModelCatalog, ModelCatalogEntry, ModelLookup, ModelMatchKind, ModelReasoning, ModelRouteClass,
    ModelTopProvider, ResolvedModel, ResolvedModelRoute, normalize_model_name,
    normalize_model_slug,
};
pub use model::{
    EffectivePricing, Endpoint, EndpointCapabilities, EndpointStats, InferenceRequest,
    InferenceResponse, Percentiles, Pricing, PricingOverride, ReasoningEffort, ReasoningSupport,
    RequestRequirements, ResponseSchema, RetryPolicy, RouteCandidate, RoutePlan, RoutingStrategy,
};
pub use provider::{InferenceProvider, ProviderError};
pub use router::{approximate_prompt_tokens, build_route_plan};
