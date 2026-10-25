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

The primary response-time estimate uses p75 TTFT plus median output throughput; OpenRouter latency telemetry is normalized when the live per-model API reports milliseconds rather than documented seconds; p95 TTFT is the tail guard, then expected cost breaks ties. Throughput uses p50 because OpenRouter endpoint surfaces have shown inconsistent throughput-percentile direction/semantics; median throughput avoids turning that presentation detail into a routing assumption. Missing telemetry degrades the local ranking signal; it never delegates provider selection back to OpenRouter. Tiered endpoints such as Flex retain their native `service_tier` grammar. Transport, HTTP 408/429/5xx, and routable endpoint failures are retried with `Retry-After` support.

Every OpenRouter execution is pinned to exactly one full endpoint tag (`deepinfra/turbo`, `near-ai/fp4`, `google-vertex/global`, etc.) with `provider.only=[endpoint]` and `allow_fallbacks=false`. OpenRouter `sort`, `order`, Auto Exacto, and provider failover are therefore not part of inference-redox routing. The provider-independent router owns failover across ranked candidates even when several candidates happen to share the same OpenRouter gateway. `provider.require_parameters` is deliberately left false on pinned calls: endpoint declarations are useful discovery/probe hints, not capability authority. `provider.max_price` remains a fail-closed billing guard rather than a selector.

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

Capabilities have three independent evidence layers. `adapter_parameters` says what inference-redox can actually encode on the wire; `supported_parameters` is the provider/catalog's nominal claim; versioned conformance observations verify or reject behavior for one execution-surface/provider/model/endpoint scope. Nominal metadata is still useful for cheap discovery and expensive-to-probe facts such as context limits, but behaviorally important schema/tool controls fail closed on empirical evidence: ordinary response formats and tools require a successful route-local probe, strict tool arguments require observed enforcement, and a strict whole-response schema requires either an enforced native schema transport or an enforced forced-tool transport. A failed probe vetoes optimistic metadata. Transient/provider failures and completion-cap truncation remain inconclusive rather than becoming negative capability evidence.

Structured-output/tool conformance distinguishes four states. A wire/API rejection is unsupported; a successfully accepted control that loses even an easy contradictory constraint is `ineffective`; easy success followed by failure on a long-nonce adversarial stress constraint is `hint_only`; surviving the stress constraint is `enforced`. This is deliberately a behavioral label: reasoning traces show that some providers expose the schema to the model, so a passing probe does not prove grammar-constrained decoding. Tool probes additionally require the exact forced function to appear in `tool_calls`, so accepting `tools` while ignoring `tool_choice` is not counted as support. `InferenceRequest` has provider-independent function tools and `ToolChoice` (`auto`, `required`, `none`, or an exact function), and `InferenceResponse` normalizes returned function calls independently from provider raw payloads. Translation is bidirectional when semantics are lossless: a strict response schema can become an exact forced synthetic `__inference_redox_output` tool on a tools-only enforced route, while an exact forced function (or a one-tool `required` request) can become an enforced native JSON-schema response and is reconstructed as a normalized `ToolCall`. Native transports are preferred. `auto` and multi-tool `required` are not rewritten through a single structured response because doing so would silently narrow their semantics.

The stdio `conformance` operation runs selected versioned probes (`basic_chat`, `system_role`, `json_object`, `json_schema_loose`, `strict_json_schema`, `strict_json_schema_stress`, `tool_schema_loose`, `tool_schema_strict`, `tool_schema_strict_stress`, plus `reasoning_off` / `reasoning_minimal` / `reasoning_low` / `reasoning_medium` / `reasoning_high` / `reasoning_xhigh` / `reasoning_max` and the independent `reasoning_budget_256`) against one explicitly selected source/model route. Reasoning probes deliberately challenge the real backend rather than trusting advertised effort lists: adapter wire support and provider/catalog claims remain separate evidence, so an unadvertised effort may still be tested and accepted or rejected upstream. Reasoning observations are effort-scoped: a failed `high` probe vetoes `high` without poisoning `low`, while successful probes populate `verified_reasoning_efforts`; `off` is first-class evidence too. Reasoning-token budgets are a separate capability (`reasoning_max_tokens`) and normal routing fails closed until that exact control has positive route evidence. Each successful reasoning probe records reasoning/completion token counts, returned trace size, finish reason, and correctness on a small uniquely-solvable generated permutation puzzle; wire acceptance and behavioral correctness remain separate facts. Requests may set `require_verified_reasoning=true` to reject merely advertised-but-untested reasoning levels. Optional `route_id`, `provider_key`, and full `endpoint_id` selectors make ambiguous gateway routes fail closed instead of guessing. Probe observations persist to the XDG cache directory (falling back to `~/.cache/inference-redox/conformance-v1.json`) and are scoped by execution surface, opaque credential label, exact endpoint, native model, test ID, and test version. Current-version observations—including inconclusive ones—are reused without spending quota until `refresh: true` explicitly requests a new observation; changing a probe definition requires incrementing its test version. Subsequent pool endpoint discovery applies the cached profile before capability filtering. Probes also inherit route workload requirements such as OpenCode Go's agentic/session policy. `INFERENCE_REDOX_CONFORMANCE_CACHE` overrides the cache path. Because stdio requests are multiplexed concurrently, callers that need freshly refreshed probe evidence to affect a later request must wait for the conformance response before sending the dependent request.

Live OpenRouter free-route characterization on 2026-10-25 demonstrates why the empirical layer is necessary. `nvidia/nemotron-3-ultra-550b-a55b:free` accepted `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `max` even though its current public metadata exposed a much narrower effort set. On the terminating conformance puzzle, `off` used zero reasoning tokens and answered incorrectly; all six non-off efforts answered correctly, but their reasoning-token counts were non-monotonic (`minimal=590`, `low=664`, `medium=680`, `high=420`, `xhigh=443`, `max=755`). These are observations, not automatic equivalence classes: one stochastic sample is enough to reject a naive ordinal interpretation but not enough to collapse labels. Separate cap-stress tests showed that total completion `max_tokens` is a hard truncation boundary on the sampled free backends rather than a target they proactively plan around: Nemotron 3.5 Lightning consumed exactly 256, 1024, and 4096 reasoning tokens at those respective total caps and returned no visible answer; Recursive Flint likewise expanded from 344 reasoning tokens at a 512-token completion cap to 1184 at 2048 and 2397 at 4096, with `finish_reason=length` and traces ending mid-thought. OpenRouter also rejects simultaneous effort plus `reasoning.max_tokens`, so inference-redox models effort mode and budget mode independently for that provider. The `reasoning_budget_256` probe against the same Nemotron Ultra route repeatedly reached an NVIDIA 5xx path while an effort-mode control succeeded; the persisted budget observation is therefore inconclusive rather than falsely rejected, and ordinary budget routing remains unavailable until positive evidence exists.

The stdio bridge exposes `models` and `resolve_model` alongside the existing OpenRouter-only `route` / `infer`. Experimental `route_pool` / `infer_pool` operations resolve canonical identity into native OpenRouter/NVIDIA routes and rank them together; keeping these opt-in avoids changing production callers before direct-provider health/quota learning is mature. Interactive requests admit standard + free routes by default. Batch is opt-in and cannot pass through synchronous `infer_pool`.


## Batch execution

Batch is a first-class asynchronous execution class, not a slow synchronous model variant. OpenRouter `:batch` catalog routes participate in cost/capability planning only when the request explicitly admits `batch`. Batch submission is pinned too: `OpenRouter::submit_batch_to_provider` requires an exact endpoint tag, emits `allow_fallbacks=false`, strips the catalog suffix to the underlying model, submits independent custom-ID requests to `/batches`, and exposes status/result fetch plus deletion. The stdio `submit_batch` operation requires `batch_provider`; unpinned gateway batch submission is intentionally unavailable. Inputs are compiled through the same normalized chat/system/schema/reasoning request model, but synchronous inference rejects batch admission rather than pretending an asynchronous job is a completion.

## Quota/resource accounting

Quota is modeled as shared resources, not as `API key × model = faucet`. A `QuotaRouteKey` can bind to any number of credential-, account-, provider-, model-family-, model-, or route-scoped buckets. Multiple credentials bind to the same resource ID when the upstream limit is actually shared, so adding keys cannot multiply account-scoped RPD/TPD by construction.

Execution instances are keyed separately by provider source and an opaque credential/account label. This lets the pool hold multiple accounts for the same provider without putting secrets into routing metadata; selected endpoints retain that label, so execution dispatch and quota binding refer to the same logical account. `register(source, ...)` remains the default-credential shorthand while `register_credential(source, label, ...)` adds explicit instances.

Finite, unknown, and explicitly unlimited quota are separate states. Unknown limits or unknown credit cost produce an `Indeterminate` admission result rather than being treated as infinity. Provider reset timestamps are observation boundaries: once a reported reset time passes, the observation becomes unknown until refreshed instead of assuming every provider refills the same way. `QuotaLedger` reserves finite headroom for concurrent work, releases unaccepted requests, and commits actual token/credit usage when available.

`InferencePool::route_with_quota` is the first routing integration. It applies ordinary model/capability/shape filtering, then quota admission, and only then computes the cheapest-cost band. Known-exhausted routes therefore cannot pin the cost floor; an exhausted $0 route is removed before a healthy paid route is compared. `QuotaUnknownPolicy::Allow` preserves experimental/unmeasured routes while excluding known exhaustion, whereas `Reject` requires known sufficient quota and returns the per-route admission reports when nothing survives.

Token-quota admission distinguishes estimates from safe bounds. Router token estimates remain useful for cost/ranking, but they are not treated as quota truth: input-token demand is known only when `InferenceRequest::prompt_tokens` is supplied, and output-token demand is known only when a hard `max_output_tokens` is present. A live NIM smoke made the reason concrete: the router estimated 6 prompt tokens for `Reply with exactly PONG`, while NVIDIA reported 21 input tokens; expected output was 8, while the model used 60 completion tokens, 56 of them reasoning, under the 256-token hard cap. Without explicit bounds, token buckets are therefore `Indeterminate`, not silently under-reserved.

`infer_with_quota` adds conservative execution accounting without changing ordinary `infer_pool`. Before each ranked candidate it re-evaluates and reserves every known finite bound resource; `QuotaUnknownPolicy::Allow` may proceed through unknown buckets but still remembers those resources for post-call accounting. Success commits normalized actual token usage when the provider reports it. A structured execution failure (`TransportUnknown` or `ResponseReceived`) commits the known reservation as a conservative worst case; if a finite bound resource may have changed by an unquantifiable amount, its remaining observation becomes `Unknown`. Local/non-dispatch errors release the reservation. Failover then re-evaluates quota, so sibling credentials sharing one account bucket cannot manufacture another request after the first ambiguous attempt consumed the shared reservation.

Execution failures carry `ExecutionFailurePhase`: `TransportUnknown` means a request may already have left the client, while `ResponseReceived` means the provider/gateway definitely answered. This is a certainty boundary, not a billing claim—provider-specific observations still decide what a 429/400/500 actually consumed. Local fake-server tests cover both phases, and conformance diagnostics continue to redact response bodies.

## Provider architecture

Implement `InferenceProvider` for another service. The adapter owns endpoint discovery and exact request/response grammar; core routing stays independent of OpenAI-compatible assumptions. OpenRouter, direct NVIDIA NIM, direct GroqCloud, free-only Kilo Gateway, OpenCode Console pay-as-you-go, and OpenCode Go implement the trait. NVIDIA and Groq share a small internal OpenAI-compatible chat/retry/usage normalizer while retaining provider-specific capability and request dialects. NVIDIA currently normalizes plain chat/system prompts, max tokens, temperature and JSON response formats; reasoning/tool dialects remain deliberately unavailable until normalized and conformance-tested. Groq additionally normalizes model-specific reasoning effort and strict structured-output support for the models where Groq documents those features.

## Cache-aware routing

The normalized model already records endpoint cache support and cache pricing. A future cache-affinity layer can add "where is this prefix warm?" observations without changing `RoutingStrategy` or provider request types.

Provider responses may carry normalized quota observations (currently the standard request/token `x-ratelimit-*` family used by services such as Groq). Adapters preserve only typed limit/remaining/reset facts, never raw headers. Observation bucket names do not imply ownership: `QuotaObservationMap` explicitly maps a bucket on a specific execution-surface/credential/provider/model route to a logical quota resource, so organization-wide and key-local limits cannot be conflated by header parsing alone.

Selected routes are never trusted indefinitely. `validate_selected_quota` re-discovers the exact endpoint, reapplies current conformance/capability checks, recomputes request-shape cost, enforces an explicit hard `max_cost_usd` against the maximum output-token bound (while routing still ranks on expected output), and re-evaluates quota immediately before dispatch. `infer_selected_quota` performs the same revalidation and then reserves/settles quota around exact synchronous execution. Batch candidates are rejected by the synchronous selected-route path and remain on the actual batch lifecycle.

Model-less planning is exposed as `select_pool_quota`. It combines canonical-model suitability evidence with a model-less `InferenceRequestTemplate`, then performs bounded live endpoint discovery and quota admission before applying one global `FastestResponseCheap` cost band across models. `ModelSelectionPolicy` makes unknown suitability, unknown quota, the live-discovery cap, and the observation timestamp explicit. Catalog pricing is only a fan-out pre-rank; final ordering always uses discovered endpoint pricing/telemetry. Canonical models without any requested route class are excluded before spending discovery budget. The result retains suitability and quota evidence beside every concrete execution route and reports how many suitable models were deferred by the discovery cap.

The long-lived stdio bridge exposes persistent quota-aware routing as `route_pool_quota` and `infer_pool_quota`. Both require an explicit `quota_unknown_policy` (`allow` or `reject`) instead of silently choosing how to treat unmeasured capacity. A single in-process `QuotaLedger` is shared across these calls, so provider-declared topology and response observations affect later admissions. `quota_state` exposes the current typed graph and observation bindings for inspection. Ordinary `route_pool` / `infer_pool` remain unchanged and do not implicitly opt into quota enforcement.

Quota-aware execution can now consume mapped provider snapshots during failover. `QuotaLedger` owns both the quota graph and its `QuotaObservationMap`; `infer_with_quota` settles local usage, then applies authoritative mapped `remaining` snapshots before considering the next candidate. That means a response from one credential can immediately exhaust a shared account bucket and suppress sibling credentials, while unbound/mismatched observations remain diagnostic-only. A regression also covers concurrent reservations so releasing one request cannot erase another request\x27s held capacity.

Providers can now declare route quota topology through `InferenceProvider::configure_quota_route` immediately before quota admission. Groq uses this to bind its standard request/token headers to shared model-scoped resources under an explicit opaque quota-pool label (organization/project identity, never the API secret). Two Groq credentials assigned the same pool therefore observe and consume the same logical resources. Providers with unknown topology leave routes unbound, preserving `Indeterminate` rather than fabricating ownership.


OpenCode uses one `OPENCODE_API_KEY` to expose two distinct execution surfaces: pay-as-you-go `opencode` under `/inference` and subscription-backed `opencode-go` under `/inference/go`. Each live model list is intersected with its corresponding current `models.dev` catalog (`opencode` or `opencode-go`) and uses models.dev canonical IDs as authoritative reconciliation hints before falling back to aliases. Pay-as-you-go excludes zero-cost client-only promo rows; Go retains subscription rows, including zero-list-cost rows. Pricing, context limits, long-context tiers, capabilities, and per-model API dialect/base come from models.dev; successful responses use OpenCode's reported `usage.cost` / `usageMetadata.cost` as actual cost. Both surfaces dispatch the four supported wire families: OpenAI-compatible Chat Completions, OpenAI Responses, Anthropic Messages, and Gemini. Unknown live rows without sufficient models.dev metadata are omitted rather than assigned guessed pricing or dialects.

`InferenceRequest::agentic` is an explicit semantic claim by the caller rather than something inferred from the prompt. Endpoints carry a provider-independent `WorkloadPolicy` (`any`, `required`, or `forbidden`) plus an optional stable-session requirement, and routing applies those restrictions before cost ranking. OpenCode Go is `ModelRouteClass::Subscription`, requires `agentic=true`, requires a stable opaque `session_id`, and sends both inference-redox's own `User-Agent` and `x-opencode-session`. Subscription routes use the current discounted `opencode-go` token pricing for routing rather than pretending included capacity is free, while retaining that pricing for selected-route hard cost ceilings. Consequently an actual zero-price free route wins over Go when other routing factors are equal, while Go can still beat ordinary paid routes on price. The same generic policy marks current OpenRouter `thinkingmachines/inkling*:free` routes agentic-only because those free endpoints explicitly require agentic harnesses; paid Inkling and unrelated free routes are not restricted. Live Go execution was validated on 2026-10-25 using `glm-5.3-flash`: OpenCode returned `PONG` with 16 input and 105 output tokens, `usage.cost=0`, `billing_source=go`, and `list_cost=allowance=0.0000549` USD. A deliberately tiny 16-token completion cap produced no ordinary text because the model consumed the budget in `reasoning_content`; a 128-token cap completed normally.

Kilo's free adapter deliberately excludes paid catalog rows. Its live `/models` feed is keyless; authenticated execution uses only the current `KILO_API_KEY` spelling, while absent credentials fall back to documented anonymous free access. `kilo-auto/*` entries are marked as virtual routes so dynamic routers cannot be merged into an unrelated canonical model through basename matching. The documented 200-requests/hour free pool is per IP and shared across free models, so Kilo declares one shared provider-scoped request resource for an opaque egress-pool label; because the gateway does not currently return normalized limit headers, its remaining capacity stays `Unknown` rather than being guessed from the published ceiling.

## Canonical model suitability

`ModelSuitabilityRequirements` is a canonical-model prefilter for future standing/background tasks. Its coarse checks evaluate Artificial Analysis summary thresholds, declared input/output modalities, nominal model-level parameters, and reasoning metadata as explicit `suitable` / `unsuitable` / `unknown` results. Missing metadata is unknown rather than failure or success; any known failed requirement makes the model unsuitable. One metadata representative is chosen per canonical identity, preferring OpenRouter standard metadata over free/batch variants and provider-native fallbacks. This layer deliberately does not decide route context, price, health, quota, or provider feature reliability: provider-route conformance and normal routing remain authoritative for execution.

Rich `benchmark_requirements` can additionally name an exact benchmark source, task, metric, unit, optional evaluated-configuration label, and qualifier subset. Aggregation is explicit: `unique` (the safe default) requires exactly one matching observation, while `best` and `worst` intentionally choose an extreme across matching configurations. A stale benchmark snapshot yields `unknown` unless that requirement explicitly sets `allow_stale=true`. This prevents a standing task from accidentally treating the best reasoning configuration as the canonical score for every route.

The stdio bridge exposes canonical suitability as the read-only `suitability` operation. When rich benchmark requirements are present, it fetches only the unique source/task snapshots those requirements need, using the shared benchmark cache, and evaluates them alongside the model catalog. Live merged-catalog smoke on 2026-10-22 with `minimum_coding_index=50` produced 448 canonical reports: 55 suitable, 62 known-unsuitable, and 331 unknown. Of the 55 known-suitable models, 5 currently have a free catalog route, 24 have batch routes, and 3 also map to direct NVIDIA inventory. This is intentionally evidence coverage, not availability: direct inventory and nominal free routes still require health/quota/conformance checks before execution.

Paid OpenRouter execution is first-class rather than a separate adapter. Ordinary OpenRouter model IDs normalize to `ModelRouteClass::Standard`, retain live endpoint pricing, and are admitted by the default interactive policy alongside free and subscription routes. `ModelRouteClass::Batch` remains opt-in. A request can narrow `allowed_route_classes` when it explicitly wants a particular economics class instead of the default interactive mix.
