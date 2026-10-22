#![doc = "Provider-agnostic inference routing with shape-aware endpoint selection."]

mod batch;
mod catalog;
mod conformance;
mod inventory;
mod model;
mod pool;
mod provider;
mod quota;
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
pub use pool::{InferencePool, PooledInferenceResponse, QuotaRoutingError};
pub use provider::{ExecutionFailurePhase, InferenceProvider, ProviderError};
pub use quota::{
    QuotaAdmission, QuotaAdmissionReport, QuotaAmount, QuotaAwareRoutePlan, QuotaCheck,
    QuotaCheckStatus, QuotaDemand, QuotaGraph, QuotaLedger, QuotaMetric, QuotaReservationId,
    QuotaResource, QuotaResourceId, QuotaRouteKey, QuotaScope, QuotaUnknownPolicy,
};
pub use router::{approximate_prompt_tokens, build_route_plan, request_prompt_tokens};
