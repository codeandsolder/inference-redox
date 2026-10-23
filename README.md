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

`inference-redox-stdio` exposes the same normalized request and routing semantics to non-Rust callers as multiplexed JSONL. It keeps one provider instance and endpoint-catalog cache warm, supports concurrent in-flight `route` and `infer` requests keyed by caller IDs, and drains outstanding requests before shutdown. Credentials may come from the process environment or `--env-file <path>`.

The bridge is deliberately thin: provider-specific request grammar, capability filtering, request-shape cost calculation, service-tier handling, fallback and retry policy remain in the Rust adapter rather than being reimplemented by callers.

## Model catalog and identity

Model identity is deliberately separate from provider endpoints. OpenRouter's `/models` catalog is the first metadata authority: `canonical_slug` groups source entries under one underlying model, with the ordinary route, `:free`, and `:batch` retained as typed catalog route classes. Batch is first-class because it has distinct cost/latency semantics useful for deferred bulk work; modifiers such as `:floor` or `:nitro` remain request/routing controls rather than model identities. Catalog entries also retain architecture/modalities, tokenizer, context and output limits, model-level nominal parameter claims, reasoning modes, pricing, and the benchmark summaries OpenRouter publishes (Artificial Analysis and Design Arena when present).

Name resolution is conservative: exact aliases first, normalized full names second, then a basename match only when it identifies exactly one canonical model. That lets a provider-native name such as `stepfun-ai/step-5-preview` resolve to an OpenRouter spelling such as `stepfun/step-5-preview` without maintaining a vendor-renaming table; ambiguous basenames are returned as ambiguous rather than guessed.

Provider-native inventories are a separate input. A direct provider route is attached to an existing OpenRouter-backed canonical identity when resolution succeeds; otherwise its native `author/model` ID becomes a provisional canonical identity. Routes carry both `source` and `id`, so `nvidia/nemotron-...` through direct NIM is distinct from the same model string through OpenRouter. NVIDIA NIM and Groq are native inventory adapters. A provider's `/models` listing is only an advertisement, not proof that the route is callable or free for the configured account; Groq's free-plan adapter therefore marks only its explicitly classified free text-generation routes as executable zero-cost routes and leaves other discovered models economically unknown.

This is intentionally not an availability model. A provider can advertise a model that is currently unhealthy or not entitled for a particular account, and multiple providers can expose the same canonical model under different names. Provider-local availability, health, quota and price observations belong to routing state, not model identity.

Capabilities have three independent evidence layers. `adapter_parameters` says what inference-redox can actually encode on the wire; `supported_parameters` is the provider/catalog's nominal claim; versioned conformance observations can verify or reject claims for one execution-surface/provider/model scope. Effective support requires adapter support and either a provider claim or successful probe, while a failed probe vetoes both. This prevents optimistic provider metadata from advertising features the adapter cannot even send yet. Transient HTTP errors remain inconclusive rather than becoming negative capability evidence.

The stdio bridge exposes `models`, `provider_inventory`, `conformance_probe`, `conformance_state`, `quota_state`, and `resolve_model` alongside the existing OpenRouter-only `route` / `infer`. `conformance_probe` executes one explicitly requested probe against one direct provider/model route and returns versioned pass/fail/inconclusive evidence; it deliberately does not auto-run a suite or consume quota implicitly. The bridge retains those observations for its lifetime and applies them to subsequent pooled endpoint discovery, so verified failures veto optimistic provider claims and verified successes can fill missing claims. `conformance_state` returns the currently retained evidence without running any probe. `quota_state` returns a read-only snapshot of the process-lifetime logical quota graph, observation bindings, current reservations, and learned remaining/reset state. `provider_inventory` is a direct-provider diagnostic surface: it returns NVIDIA/Groq native inventory or the provider error instead of silently omitting an unavailable optional provider from the merged catalog. Experimental `route_pool` / `infer_pool` operations resolve canonical identity into native OpenRouter/NVIDIA/Groq/Kilo routes and rank them together; keeping these opt-in avoids changing production callers before direct-provider health/quota learning is mature. The pooled path now keeps one process-lifetime shared quota ledger: known exhausted routes are filtered, unknown/unmeasured routes remain eligible, and provider observations learned by one request are visible to later concurrent requests. `NVIDIA_API_KEY` (including the historical mixed-case spelling), `GROQ_API_KEY`, and `KILOCODE_API_KEY` are optional. Kilo free routes remain available anonymously when its key is absent; NVIDIA/Groq direct routes require their keys. `GROQ_QUOTA_POOL_LABEL` is optional non-secret routing metadata (default `default`) for the effective Groq quota pool represented by response headers. Interactive requests admit standard + free routes by default. Batch is opt-in and cannot pass through synchronous `infer_pool`.


## Batch execution

Batch is a first-class asynchronous execution class, not a slow synchronous model variant. OpenRouter `:batch` catalog routes participate in cost/capability planning only when the request explicitly admits `batch`. `OpenRouter::submit_batch` strips the catalog suffix to the underlying model, submits independent custom-ID requests to `/batches`, and exposes status/result fetch plus deletion. The stdio bridge mirrors this as `submit_batch`, `batch_status`, and `delete_batch`. Inputs are compiled through the same normalized chat/system/schema/reasoning request model, but synchronous inference rejects batch admission rather than pretending an asynchronous job is a completion.

## Quota/resource accounting

Quota is modeled as shared resources, not as `API key × model = faucet`. A `QuotaRouteKey` can bind to any number of credential-, account-, provider-, model-family-, model-, or route-scoped buckets. Multiple credentials bind to the same resource ID when the upstream limit is actually shared, so adding keys cannot multiply account-scoped RPD/TPD by construction.

Execution instances are keyed separately by provider source and an opaque credential/account label. This lets the pool hold multiple accounts for the same provider without putting secrets into routing metadata; selected endpoints retain that label, so execution dispatch and quota binding refer to the same logical account. `register(source, ...)` remains the default-credential shorthand while `register_credential(source, label, ...)` adds explicit instances.

Finite, unknown, and explicitly unlimited quota are separate states. Unknown limits or unknown credit cost produce an `Indeterminate` admission result rather than being treated as infinity. Provider reset timestamps are observation boundaries: once a reported reset time passes, the observation becomes unknown until refreshed instead of assuming every provider refills the same way. `QuotaLedger` reserves finite headroom for concurrent work, releases unaccepted requests, and commits actual token/credit usage when available.

`InferencePool::route_with_quota` is the first routing integration. It applies ordinary model/capability/shape filtering, then quota admission, and only then computes the cheapest-cost band. Known-exhausted routes therefore cannot pin the cost floor; an exhausted $0 route is removed before a healthy paid route is compared. `QuotaUnknownPolicy::Allow` preserves experimental/unmeasured routes while excluding known exhaustion, whereas `Reject` requires known sufficient quota and returns the per-route admission reports when nothing survives.

Token-quota admission distinguishes estimates from safe bounds. Router token estimates remain useful for cost/ranking, but they are not treated as quota truth: input-token demand is known only when `InferenceRequest::prompt_tokens` is supplied, and output-token demand is known only when a hard `max_output_tokens` is present. A live NIM smoke made the reason concrete: the router estimated 6 prompt tokens for `Reply with exactly PONG`, while NVIDIA reported 21 input tokens; expected output was 8, while the model used 60 completion tokens, 56 of them reasoning, under the 256-token hard cap. Without explicit bounds, token buckets are therefore `Indeterminate`, not silently under-reserved.

`infer_with_quota` adds conservative execution accounting for a caller-owned ledger. Before each ranked candidate it re-evaluates and reserves every known finite bound resource; `QuotaUnknownPolicy::Allow` may proceed through unknown buckets but still remembers those resources for post-call accounting. Success commits normalized actual token usage when the provider reports it. A structured execution failure (`TransportUnknown` or `ResponseReceived`) commits the known reservation as a conservative worst case; if a finite bound resource may have changed by an unquantifiable amount, its remaining observation becomes `Unknown`. Local/non-dispatch errors release the reservation. Failover then re-evaluates quota, so sibling credentials sharing one account bucket cannot manufacture another request after the first ambiguous attempt consumed the shared reservation.

Execution failures carry `ExecutionFailurePhase`: `TransportUnknown` means a request may already have left the client, while `ResponseReceived` means the provider/gateway definitely answered. This is a certainty boundary, not a billing claim—provider-specific observations still decide what a 429/400/500 actually consumed. Local fake-server tests cover both phases, and conformance diagnostics continue to redact response bodies.

Kilo is also normalized as a direct provider. Its `/models` inventory is available without authentication, and only explicit free chat-route IDs (`:free`, `kilo-auto/free`, `openrouter/free`) with zero reported pricing enter routing. Other Kilo rows remain visible through `provider_inventory` with their nominal `supported_parameters` and pricing but do not become executable catalog routes. A Kilo account key is optional for the free path. Its documented 200 free requests/hour limit is modeled as one `NetworkOrigin` resource shared by all Kilo free routes on the process's egress IP. Capacity is known but remaining/reset phase start as unknown because successful Kilo responses do not expose rate-limit headers; routing therefore does not invent a fresh 200-request window on process start.

## Provider architecture

Implement `InferenceProvider` for another service. The adapter owns endpoint discovery and exact request/response grammar; core routing stays independent of OpenAI-compatible assumptions. OpenRouter, direct NVIDIA NIM and direct Groq all implement the trait. NVIDIA and Groq now share a small OpenAI-compatible transport profile instead of copy-pasted provider adapters: the profile owns wire differences such as `max_tokens` versus `max_completion_tokens`, temperature-zero behavior and reasoning-effort encoding, while provider/model wrappers own capability claims and economics. NVIDIA still deliberately withholds reasoning/tool claims until conformance-tested. Groq's free-plan wrapper exposes only classified free text-generation routes and adds model-specific JSON-schema, reasoning and implicit-cache claims.

## Cache-aware routing

The normalized model already records endpoint cache support and cache pricing. A future cache-affinity layer can add "where is this prefix warm?" observations without changing `RoutingStrategy` or provider request types.

Provider responses may carry normalized quota observations (currently the standard request/token `x-ratelimit-*` family used by services such as Groq). Adapters preserve only typed limit/remaining/reset facts, never raw headers. Observation bucket names do not imply ownership: `QuotaObservationMap` explicitly maps a bucket on a specific execution-surface/credential/provider/model route to a logical quota resource, so organization-wide and key-local limits cannot be conflated by header parsing alone.

For Groq free-plan routes, `Groq::bind_free_plan_rate_limits` can bootstrap the response-observable quota topology without pretending a fresh process knows current usage. It creates per-model account/pool-scoped RPD and TPM resources with the published capacity but `remaining = Unknown`, and maps Groq's request/token headers onto them. The caller-supplied quota-pool label names the effective pool represented by those headers; credentials should share a label only when they are known to observe the same pool. Groq can additionally apply stricter project limits beneath its organization ceiling, so a single header family is not treated as proof of both ownership layers. RPM and TPD are intentionally omitted from this bootstrap because the standard Groq response headers expose RPD and TPM snapshots, not current RPM/TPD headroom; those constraints remain upstream-enforced until a trustworthy observation source exists.

Quota-aware execution consumes mapped provider snapshots during failover. `QuotaLedger` owns both the quota graph and its `QuotaObservationMap`; `infer_with_quota` settles local usage, then applies authoritative mapped `remaining` snapshots before considering the next candidate. `SharedQuotaLedger` wraps that state for concurrent async callers, and `infer_with_shared_quota` holds its mutex only for admission/reservation/settlement—not while provider I/O is in flight. That means a response from one credential can immediately exhaust a shared account bucket and suppress sibling credentials, while unrelated requests still overlap upstream. Regressions cover both sibling suppression and the concurrency case where request A holds the only slot inside provider I/O while request B is rejected without waiting for A to finish.

## Canonical model suitability

`ModelSuitabilityRequirements` is a coarse canonical-model prefilter for future standing/background tasks. It evaluates Artificial Analysis intelligence/coding/agentic thresholds, declared input/output modalities, nominal model-level parameters, and reasoning metadata as explicit `suitable` / `unsuitable` / `unknown` checks. Missing metadata is unknown rather than failure or success; any known failed requirement makes the model unsuitable. One metadata representative is chosen per canonical identity, preferring OpenRouter standard metadata over free/batch variants and provider-native fallbacks. This layer deliberately does not decide route context, price, health, quota, or provider feature reliability: provider-route conformance and normal routing remain authoritative for execution.

The stdio bridge exposes canonical suitability as the read-only `suitability` operation. Live merged-catalog smoke on 2026-10-22 with `minimum_coding_index=50` produced 448 canonical reports: 55 suitable, 62 known-unsuitable, and 331 unknown. Of the 55 known-suitable models, 5 currently have a free catalog route, 24 have batch routes, and 3 also map to direct NVIDIA inventory. This is intentionally evidence coverage, not availability: direct inventory and nominal free routes still require health/quota/conformance checks before execution.
