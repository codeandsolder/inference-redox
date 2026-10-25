# inference-redox

Provider-agnostic inference routing utilities. Providers keep their exact wire grammar behind adapters; routing consumes normalized endpoint capability, price and performance data.

## OpenRouter

OpenRouter is the first backend. Its endpoint catalog is normalized into:

- prompt/completion/request pricing, including prompt-length and provider time-of-day pricing overrides plus cache read/write rates;
- p50/p75/p90/p95/p99 latency and throughput samples (p95 is interpolated when the source only publishes p90/p99);
- recent uptime;
- context and output limits;
- schema/structured-output support;
- reasoning support and normalized reasoning efforts;
- implicit-cache capability (recorded now; warmed-cache-aware routing is intentionally deferred).

`FastestResponseCheap(x)` first filters incompatible/unhealthy endpoints, finds the cheapest endpoint for the *actual request shape*, admits endpoints costing at most `x × cheapest`, then ranks those by predicted response time:

`latency + expected_output_tokens / throughput`

The primary response-time estimate uses p75 TTFT plus median output throughput; OpenRouter latency telemetry is normalized when the live per-model API reports milliseconds rather than documented seconds; p95 TTFT is the tail guard, then expected cost breaks ties. Throughput uses p50 because OpenRouter endpoint surfaces have shown inconsistent throughput-percentile direction/semantics; median throughput avoids turning that presentation detail into a routing assumption. If endpoint telemetry is unavailable, the OpenRouter adapter keeps the cost/capability filter and delegates speed ordering back to OpenRouter (`latency` for short responses, `throughput` for larger ones). Tiered endpoints such as Flex are emitted using their native `service_tier` grammar and fall back across route batches automatically. Transport, HTTP 408/429/5xx, and routable endpoint failures are retried with `Retry-After` support.

OpenRouter routing controls are compiled from catalog endpoint tags to provider slugs (`deepinfra/fp8` → `deepinfra`, `openai/flex` → `openai`) before emitting `provider.only` / `provider.order`; tier and quantization suffixes are endpoint metadata, not provider identifiers.

`expected_output_tokens` is deliberately an estimate used for price and response-time selection, not a generation cap. Set `InferenceRequest::max_output_tokens` separately when a hard limit is actually desired.

```rust,no_run
use inference_redox::{
    InferenceProvider, InferenceRequest, ReasoningEffort, RequestRequirements,
    ResponseSchema, RetryPolicy, RoutingStrategy,
};
use inference_redox::providers::openrouter::OpenRouter;
use serde_json::json;

# async fn demo() -> Result<(), Box<dyn std::error::Error>> {
let provider = OpenRouter::new(std::env::var("OPENROUTER_API_KEY")?);
let mut request = InferenceRequest::new(
    "openai/gpt-6-luna",
    "Return a compact structural search plan.",
    256,
);
request.requirements = RequestRequirements {
    reasoning: ReasoningEffort::Low,
    response_schema: Some(ResponseSchema::JsonSchema {
        name: "plan".into(),
        strict: true,
        schema: json!({"type":"object", "additionalProperties": false}),
    }),
    ..RequestRequirements::default()
};

let route_only = provider
    .route_prompt(
        "openai/gpt-6-luna",
        RoutingStrategy::FastestResponseCheap(1.5),
        &request.prompt,
        request.expected_output_tokens,
        request.requirements.clone(),
    )
    .await?;
assert!(!route_only.candidates.is_empty());

let (route, response) = provider
    .infer(
        RoutingStrategy::FastestResponseCheap(1.5),
        &request,
        RetryPolicy::default(),
    )
    .await?;
println!("{} via {} candidates", response.content, route.candidates.len());
# Ok(())
# }
```

## Long-lived stdio bridge

`inference-redox-stdio` exposes the same normalized request and routing semantics to non-Rust callers as multiplexed JSONL. It keeps provider instances, endpoint/model/benchmark caches, and a live conformance profile warm, supports concurrent in-flight requests keyed by caller IDs, and drains outstanding requests before shutdown. Credentials may come from the process environment or `--env-file <path>`.

The bridge is deliberately thin: provider-specific request grammar, capability filtering, request-shape cost calculation, service-tier handling, fallback and retry policy remain in the Rust adapter rather than being reimplemented by callers.

## Model catalog and identity

Model identity is deliberately separate from provider endpoints. OpenRouter's `/models` catalog is the first metadata authority: `canonical_slug` groups source entries under one underlying model, with the ordinary route, `:free`, and `:batch` retained as typed catalog route classes. Batch is first-class because it has distinct cost/latency semantics useful for deferred bulk work; modifiers such as `:floor` or `:nitro` remain request/routing controls rather than model identities. Catalog entries also retain architecture/modalities, tokenizer, context and output limits, model-level nominal parameter claims, reasoning modes, pricing, and the benchmark summaries OpenRouter publishes (Artificial Analysis and Design Arena when present).

Rich benchmark evidence is a separate layer rather than being flattened into those model summaries. `OpenRouter::benchmarks` normalizes Artificial Analysis, Design Arena, and OpenRouter first-party eval snapshots into `BenchmarkObservation`s keyed by permanent model slug while preserving source, task type, metric, unit, evaluated configuration, qualifiers, sample counts, uncertainty, observed cost/timing, and source timestamps. `BenchmarkCatalog` can replace snapshots source/task-wise and join observations back to canonical models explicitly. The bridge exposes the same data through `benchmarks` with `benchmark_source` and `benchmark_task_type`. Source/task availability remains an upstream fact: empty or unsupported combinations are surfaced rather than synthesized.

Benchmark snapshots use a separate one-hour cache because the upstream feed is rate-limited and changes much more slowly than endpoint telemetry. A refresh failure may return the retained snapshot with `meta.stale=true`; successful fetches and TTL-valid cache hits have `stale=false`. Configuration-specific rows intentionally do not overwrite `ModelCatalog`'s coarse suitability summaries: a score for one reasoning mode or arena slice is evidence about that configuration, not automatically a score for every route of the canonical model.

Name resolution is conservative: exact aliases first, normalized full names second, then a basename match only when it identifies exactly one canonical model. That lets a provider-native name such as `stepfun-ai/step-5-preview` resolve to an OpenRouter spelling such as `stepfun/step-5-preview` without maintaining a vendor-renaming table; ambiguous basenames are returned as ambiguous rather than guessed.

Provider-native inventories are a separate input. A direct provider route is attached to an existing OpenRouter-backed canonical identity when resolution succeeds; otherwise its native `author/model` ID becomes a provisional canonical identity. Routes carry both `source` and `id`, so a direct route is distinct from the same model string through OpenRouter. NVIDIA NIM, GroqCloud, and the free-only Kilo Gateway surface are native inventory/execution adapters. Every native inventory is reconciled independently against the same base OpenRouter catalog before routes are merged, so canonical identity cannot depend on provider ingestion order. A provider's `/models` listing is only an advertisement, not proof that the route is callable.

This is intentionally not an availability model. A provider can advertise a model that is currently unhealthy or not entitled for a particular account, and multiple providers can expose the same canonical model under different names. Provider-local availability, health, quota and price observations belong to routing state, not model identity.

Capabilities have three independent evidence layers. `adapter_parameters` says what inference-redox can actually encode on the wire; `supported_parameters` is the provider/catalog's nominal claim; versioned conformance observations can verify or reject claims for one execution-surface/provider/model scope. Effective support requires adapter support and either a provider claim or successful probe, while a failed probe vetoes both. This prevents optimistic provider metadata from advertising features the adapter cannot even send yet. Transient HTTP errors remain inconclusive rather than becoming negative capability evidence.

The stdio `conformance` operation runs selected versioned probes (`basic_chat`, `system_role`, `json_object`, `strict_json_schema`, `reasoning_low`) against one explicitly selected source/model route. Optional `route_id` and `provider_key` selectors make ambiguous gateway routes fail closed instead of guessing. Probe observations persist to the XDG cache directory (falling back to `~/.cache/inference-redox/conformance-v1.json`) and are scoped by execution surface, opaque credential label, exact endpoint, native model, test ID, and test version. Current-version observations—including inconclusive ones—are reused without spending quota until `refresh: true` explicitly requests a new observation; changing a probe definition requires incrementing its test version. Subsequent pool endpoint discovery applies the cached profile before capability filtering. `INFERENCE_REDOX_CONFORMANCE_CACHE` overrides the cache path. Because stdio requests are multiplexed concurrently, callers that need freshly refreshed probe evidence to affect a later request must wait for the conformance response before sending the dependent request.

The stdio bridge exposes `models` and `resolve_model` alongside the existing OpenRouter-only `route` / `infer`. Experimental `route_pool` / `infer_pool` operations resolve canonical identity into native OpenRouter/NVIDIA routes and rank them together; keeping these opt-in avoids changing production callers before direct-provider health/quota learning is mature. Interactive requests admit standard + free routes by default. Batch is opt-in and cannot pass through synchronous `infer_pool`.


## Batch execution

Batch is a first-class asynchronous execution class, not a slow synchronous model variant. OpenRouter `:batch` catalog routes participate in cost/capability planning only when the request explicitly admits `batch`. `OpenRouter::submit_batch` strips the catalog suffix to the underlying model, submits independent custom-ID requests to `/batches`, and exposes status/result fetch plus deletion. The stdio bridge mirrors this as `submit_batch`, `batch_status`, and `delete_batch`. Inputs are compiled through the same normalized chat/system/schema/reasoning request model, but synchronous inference rejects batch admission rather than pretending an asynchronous job is a completion.

## Quota/resource accounting

Quota is modeled as shared resources, not as `API key × model = faucet`. A `QuotaRouteKey` can bind to any number of credential-, account-, provider-, model-family-, model-, or route-scoped buckets. Multiple credentials bind to the same resource ID when the upstream limit is actually shared, so adding keys cannot multiply account-scoped RPD/TPD by construction.

Execution instances are keyed separately by provider source and an opaque credential/account label. This lets the pool hold multiple accounts for the same provider without putting secrets into routing metadata; selected endpoints retain that label, so execution dispatch and quota binding refer to the same logical account. `register(source, ...)` remains the default-credential shorthand while `register_credential(source, label, ...)` adds explicit instances.

Finite, unknown, and explicitly unlimited quota are separate states. Unknown limits or unknown credit cost produce an `Indeterminate` admission result rather than being treated as infinity. Provider reset timestamps are observation boundaries: once a reported reset time passes, the observation becomes unknown until refreshed instead of assuming every provider refills the same way. `QuotaLedger` reserves finite headroom for concurrent work, releases unaccepted requests, and commits actual token/credit usage when available.

`InferencePool::route_with_quota` is the first routing integration. It applies ordinary model/capability/shape filtering, then quota admission, and only then computes the cheapest-cost band. Known-exhausted routes therefore cannot pin the cost floor; an exhausted $0 route is removed before a healthy paid route is compared. `QuotaUnknownPolicy::Allow` preserves experimental/unmeasured routes while excluding known exhaustion, whereas `Reject` requires known sufficient quota and returns the per-route admission reports when nothing survives.

Token-quota admission distinguishes estimates from safe bounds. Router token estimates remain useful for cost/ranking, but they are not treated as quota truth: input-token demand is known only when `InferenceRequest::prompt_tokens` is supplied, and output-token demand is known only when a hard `max_output_tokens` is present. A live NIM smoke made the reason concrete: the router estimated 6 prompt tokens for `Reply with exactly PONG`, while NVIDIA reported 21 input tokens; expected output was 8, while the model used 60 completion tokens, 56 of them reasoning, under the 256-token hard cap. Without explicit bounds, token buckets are therefore `Indeterminate`, not silently under-reserved.

`infer_with_quota` adds conservative execution accounting without changing ordinary `infer_pool`. Before each ranked candidate it re-evaluates and reserves every known finite bound resource; `QuotaUnknownPolicy::Allow` may proceed through unknown buckets but still remembers those resources for post-call accounting. Success commits normalized actual token usage when the provider reports it. A structured execution failure (`TransportUnknown` or `ResponseReceived`) commits the known reservation as a conservative worst case; if a finite bound resource may have changed by an unquantifiable amount, its remaining observation becomes `Unknown`. Local/non-dispatch errors release the reservation. Failover then re-evaluates quota, so sibling credentials sharing one account bucket cannot manufacture another request after the first ambiguous attempt consumed the shared reservation.

Execution failures carry `ExecutionFailurePhase`: `TransportUnknown` means a request may already have left the client, while `ResponseReceived` means the provider/gateway definitely answered. This is a certainty boundary, not a billing claim—provider-specific observations still decide what a 429/400/500 actually consumed. Local fake-server tests cover both phases, and conformance diagnostics continue to redact response bodies.

## Provider architecture

Implement `InferenceProvider` for another service. The adapter owns endpoint discovery and exact request/response grammar; core routing stays independent of OpenAI-compatible assumptions. OpenRouter, direct NVIDIA NIM, direct GroqCloud, and free-only Kilo Gateway implement the trait. NVIDIA and Groq share a small internal OpenAI-compatible chat/retry/usage normalizer while retaining provider-specific capability and request dialects. NVIDIA currently normalizes plain chat/system prompts, max tokens, temperature and JSON response formats; reasoning/tool dialects remain deliberately unavailable until normalized and conformance-tested. Groq additionally normalizes model-specific reasoning effort and strict structured-output support for the models where Groq documents those features.

## Cache-aware routing

The normalized model already records endpoint cache support and cache pricing. A future cache-affinity layer can add "where is this prefix warm?" observations without changing `RoutingStrategy` or provider request types.

Provider responses may carry normalized quota observations (currently the standard request/token `x-ratelimit-*` family used by services such as Groq). Adapters preserve only typed limit/remaining/reset facts, never raw headers. Observation bucket names do not imply ownership: `QuotaObservationMap` explicitly maps a bucket on a specific execution-surface/credential/provider/model route to a logical quota resource, so organization-wide and key-local limits cannot be conflated by header parsing alone.

The long-lived stdio bridge exposes persistent quota-aware routing as `route_pool_quota` and `infer_pool_quota`. Both require an explicit `quota_unknown_policy` (`allow` or `reject`) instead of silently choosing how to treat unmeasured capacity. A single in-process `QuotaLedger` is shared across these calls, so provider-declared topology and response observations affect later admissions. `quota_state` exposes the current typed graph and observation bindings for inspection. Ordinary `route_pool` / `infer_pool` remain unchanged and do not implicitly opt into quota enforcement.

Quota-aware execution can now consume mapped provider snapshots during failover. `QuotaLedger` owns both the quota graph and its `QuotaObservationMap`; `infer_with_quota` settles local usage, then applies authoritative mapped `remaining` snapshots before considering the next candidate. That means a response from one credential can immediately exhaust a shared account bucket and suppress sibling credentials, while unbound/mismatched observations remain diagnostic-only. A regression also covers concurrent reservations so releasing one request cannot erase another request\x27s held capacity.

Providers can now declare route quota topology through `InferenceProvider::configure_quota_route` immediately before quota admission. Groq uses this to bind its standard request/token headers to shared model-scoped resources under an explicit opaque quota-pool label (organization/project identity, never the API secret). Two Groq credentials assigned the same pool therefore observe and consume the same logical resources. Providers with unknown topology leave routes unbound, preserving `Indeterminate` rather than fabricating ownership.

Kilo's free adapter deliberately excludes paid catalog rows. Its live `/models` feed is keyless; authenticated execution uses only the current `KILO_API_KEY` spelling, while absent credentials fall back to documented anonymous free access. `kilo-auto/*` entries are marked as virtual routes so dynamic routers cannot be merged into an unrelated canonical model through basename matching. The documented 200-requests/hour free pool is per IP and shared across free models, so Kilo declares one shared provider-scoped request resource for an opaque egress-pool label; because the gateway does not currently return normalized limit headers, its remaining capacity stays `Unknown` rather than being guessed from the published ceiling.

## Canonical model suitability

`ModelSuitabilityRequirements` is a canonical-model prefilter for future standing/background tasks. Its coarse checks evaluate Artificial Analysis summary thresholds, declared input/output modalities, nominal model-level parameters, and reasoning metadata as explicit `suitable` / `unsuitable` / `unknown` results. Missing metadata is unknown rather than failure or success; any known failed requirement makes the model unsuitable. One metadata representative is chosen per canonical identity, preferring OpenRouter standard metadata over free/batch variants and provider-native fallbacks. This layer deliberately does not decide route context, price, health, quota, or provider feature reliability: provider-route conformance and normal routing remain authoritative for execution.

Rich `benchmark_requirements` can additionally name an exact benchmark source, task, metric, unit, optional evaluated-configuration label, and qualifier subset. Aggregation is explicit: `unique` (the safe default) requires exactly one matching observation, while `best` and `worst` intentionally choose an extreme across matching configurations. A stale benchmark snapshot yields `unknown` unless that requirement explicitly sets `allow_stale=true`. This prevents a standing task from accidentally treating the best reasoning configuration as the canonical score for every route.

The stdio bridge exposes canonical suitability as the read-only `suitability` operation. When rich benchmark requirements are present, it fetches only the unique source/task snapshots those requirements need, using the shared benchmark cache, and evaluates them alongside the model catalog. Live merged-catalog smoke on 2026-10-22 with `minimum_coding_index=50` produced 448 canonical reports: 55 suitable, 62 known-unsuitable, and 331 unknown. Of the 55 known-suitable models, 5 currently have a free catalog route, 24 have batch routes, and 3 also map to direct NVIDIA inventory. This is intentionally evidence coverage, not availability: direct inventory and nominal free routes still require health/quota/conformance checks before execution.

Paid OpenRouter execution is first-class rather than a separate adapter. Ordinary OpenRouter model IDs normalize to `ModelRouteClass::Standard`, retain live endpoint pricing, and are admitted by the default interactive policy alongside free routes. A request can set `allowed_route_classes` to only `standard` when it explicitly wants paid OpenRouter execution instead of a free sibling route.
