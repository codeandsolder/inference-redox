use inference_redox::{
    Endpoint, EndpointCapabilities, EndpointStats, InferenceRequest, Percentiles, Pricing,
    ReasoningEffort, ReasoningSupport, RequestRequirements, ResponseSchema, RoutingStrategy,
    build_route_plan,
};
use serde_json::json;

fn endpoint(id: &str, prompt: f64, completion: f64, latency: f64, throughput: f64) -> Endpoint {
    Endpoint {
        id: id.to_owned(),
        provider: id.to_owned(),
        model: "x/model".to_owned(),
        display_name: id.to_owned(),
        quantization: None,
        status: 0,
        context_length: Some(100_000),
        max_prompt_tokens: Some(90_000),
        max_completion_tokens: Some(10_000),
        pricing: Pricing {
            prompt_per_token: prompt,
            completion_per_token: completion,
            ..Pricing::default()
        },
        stats: EndpointStats {
            latency_seconds: Percentiles {
                p50: Some(latency),
                p90: Some(latency * 2.0),
                p99: Some(latency * 3.0),
                ..Percentiles::default()
            },
            throughput_tokens_per_second: Percentiles {
                p50: Some(throughput),
                p90: Some(throughput / 2.0),
                p99: Some(throughput / 3.0),
                ..Percentiles::default()
            },
            uptime_30m: Some(99.9),
            ..EndpointStats::default()
        },
        capabilities: EndpointCapabilities {
            supported_parameters: ["response_format", "structured_outputs", "reasoning"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            reasoning: ReasoningSupport::GatewayNormalized,
            supports_implicit_caching: true,
        },
    }
}

#[test]
fn fastest_within_cost_band_wins() -> Result<(), inference_redox::ProviderError> {
    let mut request = InferenceRequest::new("x/model", "hello", 100);
    request.prompt_tokens = Some(1000);
    let plan = build_route_plan(
        RoutingStrategy::FastestResponseCheap(1.5),
        &request,
        vec![
            endpoint("cheap-slow", 1e-6, 2e-6, 1.0, 20.0),
            endpoint("fast-in-band", 1.1e-6, 2.2e-6, 0.1, 200.0),
            endpoint("too-expensive", 3e-6, 6e-6, 0.01, 1000.0),
        ],
    )?;
    assert_eq!(plan.candidates[0].endpoint.id, "fast-in-band");
    assert!(
        plan.candidates
            .iter()
            .all(|candidate| candidate.endpoint.id != "too-expensive")
    );
    Ok(())
}

#[test]
fn request_shape_changes_cost_band() -> Result<(), inference_redox::ProviderError> {
    let prompt_cheap = endpoint("prompt-cheap", 0.1e-6, 5.0e-6, 0.2, 100.0);
    let output_cheap = endpoint("output-cheap", 1.0e-6, 0.1e-6, 0.2, 100.0);

    let mut prompt_heavy = InferenceRequest::new("x/model", "x", 10);
    prompt_heavy.prompt_tokens = Some(10_000);
    let prompt_plan = build_route_plan(
        RoutingStrategy::FastestResponseCheap(1.01),
        &prompt_heavy,
        vec![prompt_cheap.clone(), output_cheap.clone()],
    )?;
    assert_eq!(prompt_plan.candidates[0].endpoint.id, "prompt-cheap");

    let mut output_heavy = InferenceRequest::new("x/model", "x", 10_000);
    output_heavy.prompt_tokens = Some(10);
    let output_plan = build_route_plan(
        RoutingStrategy::FastestResponseCheap(1.01),
        &output_heavy,
        vec![prompt_cheap, output_cheap],
    )?;
    assert_eq!(output_plan.candidates[0].endpoint.id, "output-cheap");
    Ok(())
}

#[test]
fn strict_schema_and_reasoning_are_filtered_before_price()
-> Result<(), inference_redox::ProviderError> {
    let mut capable = endpoint("capable", 2e-6, 2e-6, 0.2, 100.0);
    capable.capabilities.reasoning =
        ReasoningSupport::Exact([ReasoningEffort::High].into_iter().collect());
    let mut cheap_but_missing_schema = endpoint("cheap", 0.1e-6, 0.1e-6, 0.1, 100.0);
    cheap_but_missing_schema
        .capabilities
        .supported_parameters
        .remove("structured_outputs");
    let mut cheap_but_wrong_reasoning = endpoint("wrong-reasoning", 0.1e-6, 0.1e-6, 0.1, 100.0);
    cheap_but_wrong_reasoning.capabilities.reasoning =
        ReasoningSupport::Exact([ReasoningEffort::Low].into_iter().collect());

    let mut request = InferenceRequest::new("x/model", "x", 100);
    request.prompt_tokens = Some(100);
    request.requirements = RequestRequirements {
        response_schema: Some(ResponseSchema::JsonSchema {
            name: "x".into(),
            schema: json!({"type":"object"}),
            strict: true,
        }),
        reasoning: ReasoningEffort::High,
        ..RequestRequirements::default()
    };
    let plan = build_route_plan(
        RoutingStrategy::FastestResponseCheap(1.5),
        &request,
        vec![capable, cheap_but_missing_schema, cheap_but_wrong_reasoning],
    )?;
    assert_eq!(plan.candidates.len(), 1);
    assert_eq!(plan.candidates[0].endpoint.id, "capable");
    Ok(())
}

#[test]
fn p95_is_interpolated_from_p90_and_p99() {
    let p = Percentiles {
        p90: Some(10.0),
        p99: Some(19.0),
        ..Percentiles::default()
    };
    assert_eq!(p.p95_or_interpolate(), Some(15.0));
}

#[test]
fn p75_is_interpolated_from_p50_and_p90() {
    let p = Percentiles {
        p50: Some(10.0),
        p90: Some(18.0),
        ..Percentiles::default()
    };
    assert_eq!(p.p75_or_interpolate(), Some(15.0));
}
