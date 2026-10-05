# inference-redox

Provider-agnostic inference routing utilities. Providers keep their exact wire grammar behind adapters; routing consumes normalized endpoint capability, price and performance data.

## OpenRouter

OpenRouter is the first backend. Its endpoint catalog is normalized into:

- prompt/completion/request pricing;
- p50/p75/p90/p95/p99 latency and throughput samples (p95 is interpolated when the source only publishes p90/p99);
- recent uptime;
- context and output limits;
- schema/structured-output support;
- reasoning support and normalized reasoning efforts;
- implicit-cache capability (recorded now; warmed-cache-aware routing is intentionally deferred).

`FastestResponseCheap(x)` first filters incompatible/unhealthy endpoints, finds the cheapest endpoint for the *actual request shape*, admits endpoints costing at most `x × cheapest`, then ranks those by predicted response time:

`latency + expected_output_tokens / throughput`

The p75 TTFT estimate is primary; a p95 TTFT tail estimate is the next tie-breaker, then expected cost. OpenRouter receives the resulting full endpoint tags in explicit order with provider fallback enabled. Transport, HTTP 408/429, and 5xx failures are retried by the client as a second reliability layer.

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

## Provider architecture

Implement `InferenceProvider` for another service. The adapter owns endpoint discovery and exact request/response grammar; core routing stays independent of OpenAI-compatible assumptions.

## Cache-aware routing

The normalized model already records endpoint cache support and cache pricing. A future cache-affinity layer can add "where is this prefix warm?" observations without changing `RoutingStrategy` or provider request types.
