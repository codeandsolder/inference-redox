#![doc = "Provider-agnostic inference routing with shape-aware endpoint selection."]

mod batch;
mod benchmarks;
mod catalog;
mod conformance;
mod inventory;
mod model;
mod pool;
mod provider;
mod quota;
mod quota_headers;
mod router;
mod selection;
mod suitability;

pub mod providers;

pub use batch::{BatchInferenceItem, BatchJob};
pub use benchmarks::{
    BenchmarkCatalog, BenchmarkObservation, BenchmarkSnapshot, BenchmarkSnapshotMeta,
    BenchmarkSource, BenchmarkTaskType, BenchmarkUnit,
};
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
    AgenticPolicy, EffectivePricing, Endpoint, EndpointCapabilities, EndpointStats,
    InferenceRequest, InferenceResponse, Percentiles, Pricing, PricingOverride, ReasoningEffort,
    ReasoningSupport, RequestRequirements, ResponseSchema, RetryPolicy, RouteCandidate, RoutePlan,
    RoutingStrategy, TokenUsage, WorkloadPolicy,
};
pub use pool::{
    InferencePool, PooledInferenceResponse, QuotaExecutionContext, QuotaInferenceError,
    QuotaPooledInferenceResponse, QuotaRoutingError, SelectedRouteExecutionError,
    SelectedRouteExecutionPolicy, SelectedRouteValidation,
};
pub use provider::{ExecutionFailurePhase, InferenceProvider, ProviderError};
pub use quota::{
    QuotaAdmission, QuotaAdmissionReport, QuotaAmount, QuotaAwareRoutePlan, QuotaCheck,
    QuotaCheckStatus, QuotaDemand, QuotaGraph, QuotaGraphSnapshot, QuotaLedger, QuotaMetric,
    QuotaObservation, QuotaObservationApplyResult, QuotaObservationApplyStatus,
    QuotaObservationBinding, QuotaObservationMap, QuotaReservationId, QuotaResource,
    QuotaResourceId, QuotaRouteBinding, QuotaRouteKey, QuotaScope, QuotaUnknownPolicy,
};
pub use router::{approximate_prompt_tokens, build_route_plan, request_prompt_tokens};
pub use selection::{
    InferenceRequestTemplate, ModelRouteSelectionCandidate, ModelSelectionEvidence,
    ModelSelectionPlan, ModelSelectionPolicy, ModelSelectionRejection, SuitabilityUnknownPolicy,
};
pub use suitability::{
    BenchmarkAggregation, BenchmarkKind, BenchmarkRequirement, ModelSuitabilityReport,
    ModelSuitabilityRequirements, SuitabilityCheck, SuitabilityStatus,
};
