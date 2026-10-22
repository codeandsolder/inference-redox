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

Provider-native inventories are a separate input. A direct provider route is attached to an existing OpenRouter-backed canonical identity when resolution succeeds; otherwise its native `author/model` ID becomes a provisional canonical identity. Routes carry both `source` and `id`, so `nvidia/nemotron-...` through direct NIM is distinct from the same model string through OpenRouter. NVIDIA NIM is the first native inventory adapter. A provider's `/models` listing is only an advertisement, not proof that the route is callable.

This is intentionally not an availability model. A provider can advertise a model that is currently unhealthy or not entitled for a particular account, and multiple providers can expose the same canonical model under different names. Provider-local availability, health, quota and price observations belong to routing state, not model identity.

Capabilities have three independent evidence layers. `adapter_parameters` says what inference-redox can actually encode on the wire; `supported_parameters` is the provider/catalog's nominal claim; versioned conformance observations can verify or reject claims for one execution-surface/provider/model scope. Effective support requires adapter support and either a provider claim or successful probe, while a failed probe vetoes both. This prevents optimistic provider metadata from advertising features the adapter cannot even send yet. Transient HTTP errors remain inconclusive rather than becoming negative capability evidence.

The stdio bridge exposes `models` and `resolve_model` alongside the existing OpenRouter-only `route` / `infer`. Experimental `route_pool` / `infer_pool` operations resolve canonical identity into native OpenRouter/NVIDIA routes and rank them together; keeping these opt-in avoids changing production callers before direct-provider health/quota learning is mature. Interactive requests admit standard + free routes by default. Batch is opt-in and cannot pass through synchronous `infer_pool`.


## Batch execution

Batch is a first-class asynchronous execution class, not a slow synchronous model variant. OpenRouter `:batch` catalog routes participate in cost/capability planning only when the request explicitly admits `batch`. `OpenRouter::submit_batch` strips the catalog suffix to the underlying model, submits independent custom-ID requests to `/batches`, and exposes status/result fetch plus deletion. The stdio bridge mirrors this as `submit_batch`, `batch_status`, and `delete_batch`. Inputs are compiled through the same normalized chat/system/schema/reasoning request model, but synchronous inference rejects batch admission rather than pretending an asynchronous job is a completion.

## Quota/resource accounting

Quota is modeled as shared resources, not as `API key × model = faucet`. A `QuotaRouteKey` can bind to any number of credential-, account-, provider-, model-family-, model-, or route-scoped buckets. Multiple credentials bind to the same resource ID when the upstream limit is actually shared, so adding keys cannot multiply account-scoped RPD/TPD by construction.

Finite, unknown, and explicitly unlimited quota are separate states. Unknown limits or unknown credit cost produce an `Indeterminate` admission result rather than being treated as infinity. Provider reset timestamps are observation boundaries: once a reported reset time passes, the observation becomes unknown until refreshed instead of assuming every provider refills the same way. `QuotaLedger` reserves finite headroom for concurrent work, releases unaccepted requests, and commits actual token/credit usage when available.

## Provider architecture

Implement `InferenceProvider` for another service. The adapter owns endpoint discovery and exact request/response grammar; core routing stays independent of OpenAI-compatible assumptions. OpenRouter and direct NVIDIA NIM both implement the trait. NVIDIA currently normalizes plain chat/system prompts, max tokens, temperature and JSON response formats; reasoning/tool dialects remain deliberately unavailable until normalized and conformance-tested.

## Cache-aware routing

The normalized model already records endpoint cache support and cache pricing. A future cache-affinity layer can add "where is this prefix warm?" observations without changing `RoutingStrategy` or provider request types.
