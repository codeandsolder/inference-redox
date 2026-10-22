#![doc = "Provider-agnostic inference routing with shape-aware endpoint selection."]

mod batch;
mod catalog;
mod conformance;
mod inventory;
mod model;
mod pool;
mod provider;
mod router;

pub mod providers;

pub use batch::{BatchInferenceItem, BatchJob};
pub use catalog::{
    ArtificialAnalysisBenchmarks, DesignArenaBenchmark, ModelArchitecture, ModelBenchmarks,
    ModelCatalog, ModelCatalogEntry, ModelLookup, ModelMatchKind, ModelReasoning, ModelRouteClass,
    ModelTopProvider, ResolvedModel, ResolvedModelRoute, normalize_model_name,
    normalize_model_slug, provisional_canonical_id,
};
pub use conformance::{
    ConformanceObservation, ConformanceOutcome, ConformanceProbe, ConformanceProfile,
};
pub use inventory::{
    InventoryResolution, NativeModelRecord, ProviderInventory, ProviderModelObservation,
};
pub use model::{
    EffectivePricing, Endpoint, EndpointCapabilities, EndpointStats, InferenceRequest,
    InferenceResponse, Percentiles, Pricing, PricingOverride, ReasoningEffort, ReasoningSupport,
    RequestRequirements, ResponseSchema, RetryPolicy, RouteCandidate, RoutePlan, RoutingStrategy,
};
pub use pool::{InferencePool, PooledInferenceResponse};
pub use provider::{InferenceProvider, ProviderError};
pub use router::{approximate_prompt_tokens, build_route_plan};
