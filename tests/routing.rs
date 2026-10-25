use std::collections::BTreeSet;

use inference_redox::{
    Endpoint, EndpointCapabilities, EndpointStats, InferenceRequest, Percentiles, Pricing,
    ReasoningEffort, ReasoningSupport, RequestRequirements, ResponseSchema, RoutingStrategy,
    build_route_plan,
};
use serde_json::json;

fn endpoint(id: &str, prompt: f64, completion: f64, latency: f64, throughput: f64) -> Endpoint {
    Endpoint {
        source: "test".to_owned(),
        credential: None,
        provider_key: id.to_owned(),
        id: id.to_owned(),
        provider: id.to_owned(),
        model: "x/model".to_owned(),
        route_class: inference_redox::ModelRouteClass::Standard,
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
        workload_policy: inference_redox::WorkloadPolicy::default(),
        capabilities: EndpointCapabilities {
            adapter_parameters: [
                "max_tokens",
                "max_completion_tokens",
                "temperature",
                "response_format",
                "structured_outputs",
                "reasoning",
                "reasoning_effort",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            supported_parameters: [
                "max_tokens",
                "max_completion_tokens",
                "temperature",
                "response_format",
                "structured_outputs",
                "reasoning",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            verified_parameters: BTreeSet::new(),
            rejected_parameters: BTreeSet::new(),
            verified_reasoning_efforts: BTreeSet::new(),
            rejected_reasoning_efforts: BTreeSet::new(),
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
fn reasoning_budget_requires_verified_route_support() -> Result<(), inference_redox::ProviderError>
{
    let mut route = endpoint("budget", 1e-6, 1e-6, 0.1, 100.0);
    route
        .capabilities
        .adapter_parameters
        .insert("reasoning_max_tokens".to_owned());

    let mut request = InferenceRequest::new("x/model", "x", 10);
    request.prompt_tokens = Some(10);
    request.requirements.reasoning_max_tokens = Some(256);

    assert!(matches!(
        build_route_plan(
            RoutingStrategy::FastestResponseCheap(1.5),
            &request,
            vec![route.clone()],
        ),
        Err(inference_redox::ProviderError::NoEligibleEndpoint(_))
    ));

    route
        .capabilities
        .verified_parameters
        .insert("reasoning_max_tokens".to_owned());
    let plan = build_route_plan(
        RoutingStrategy::FastestResponseCheap(1.5),
        &request,
        vec![route],
    )?;
    assert_eq!(plan.candidates.len(), 1);
    Ok(())
}

#[test]
fn verified_reasoning_off_can_be_required_explicitly() -> Result<(), inference_redox::ProviderError>
{
    let mut endpoint = endpoint("verified-off", 1e-6, 1e-6, 0.1, 100.0);
    endpoint
        .capabilities
        .verified_reasoning_efforts
        .insert(ReasoningEffort::None);
    let mut request = InferenceRequest::new("x/model", "x", 10);
    request.prompt_tokens = Some(10);
    request
        .requirements
        .required_parameters
        .insert("reasoning".to_owned());
    request.requirements.require_verified_reasoning = true;

    let plan = build_route_plan(
        RoutingStrategy::FastestResponseCheap(1.5),
        &request,
        vec![endpoint],
    )?;
    assert_eq!(plan.candidates.len(), 1);
    Ok(())
}

#[test]
fn verified_reasoning_requirement_rejects_nominal_only_support()
-> Result<(), inference_redox::ProviderError> {
    let mut route = endpoint("reasoning", 1e-6, 1e-6, 0.1, 100.0);
    route.capabilities.reasoning = ReasoningSupport::GatewayNormalized;

    let mut request = InferenceRequest::new("x/model", "x", 100);
    request.prompt_tokens = Some(100);
    request.requirements.reasoning = ReasoningEffort::High;
    request.requirements.require_verified_reasoning = true;
    assert!(matches!(
        build_route_plan(
            RoutingStrategy::FastestResponseCheap(1.5),
            &request,
            vec![route.clone()],
        ),
        Err(inference_redox::ProviderError::NoEligibleEndpoint(_))
    ));

    route
        .capabilities
        .verified_reasoning_efforts
        .insert(ReasoningEffort::High);
    let plan = build_route_plan(
        RoutingStrategy::FastestResponseCheap(1.5),
        &request,
        vec![route],
    )?;
    assert_eq!(plan.candidates.len(), 1);
    Ok(())
}

#[test]
fn rejected_reasoning_effort_does_not_poison_other_levels()
-> Result<(), inference_redox::ProviderError> {
    let mut route = endpoint("reasoning", 1e-6, 1e-6, 0.1, 100.0);
    route.capabilities.reasoning = ReasoningSupport::GatewayNormalized;
    route
        .capabilities
        .rejected_reasoning_efforts
        .insert(ReasoningEffort::High);

    let mut low = InferenceRequest::new("x/model", "x", 100);
    low.prompt_tokens = Some(100);
    low.requirements.reasoning = ReasoningEffort::Low;
    assert_eq!(
        build_route_plan(
            RoutingStrategy::FastestResponseCheap(1.5),
            &low,
            vec![route.clone()],
        )?
        .candidates
        .len(),
        1
    );

    low.requirements.reasoning = ReasoningEffort::High;
    assert!(matches!(
        build_route_plan(
            RoutingStrategy::FastestResponseCheap(1.5),
            &low,
            vec![route],
        ),
        Err(inference_redox::ProviderError::NoEligibleEndpoint(_))
    ));
    Ok(())
}

#[test]
fn provider_claim_cannot_exceed_adapter_wire_support() {
    let mut optimistic = endpoint("optimistic-provider", 1e-6, 1e-6, 0.1, 100.0);
    optimistic
        .capabilities
        .supported_parameters
        .insert("tools".to_owned());

    let mut request = InferenceRequest::new("x/model", "x", 100);
    request.prompt_tokens = Some(100);
    request
        .requirements
        .required_parameters
        .insert("tools".to_owned());

    assert!(matches!(
        build_route_plan(
            RoutingStrategy::FastestResponseCheap(1.5),
            &request,
            vec![optimistic],
        ),
        Err(inference_redox::ProviderError::NoEligibleEndpoint(_))
    ));
}

#[test]
fn rejected_parameter_overrides_nominal_provider_claim() {
    let mut lying = endpoint("optimistic-provider", 1e-6, 1e-6, 0.1, 100.0);
    lying
        .capabilities
        .rejected_parameters
        .insert("structured_outputs".to_owned());

    let mut request = InferenceRequest::new("x/model", "x", 100);
    request.prompt_tokens = Some(100);
    request.requirements = RequestRequirements {
        response_schema: Some(ResponseSchema::JsonSchema {
            name: "x".into(),
            schema: json!({"type":"object"}),
            strict: true,
        }),
        ..RequestRequirements::default()
    };

    assert!(matches!(
        build_route_plan(
            RoutingStrategy::FastestResponseCheap(1.5),
            &request,
            vec![lying],
        ),
        Err(inference_redox::ProviderError::NoEligibleEndpoint(_))
    ));
}

#[test]
fn rejected_reasoning_claim_overrides_nominal_metadata() {
    let mut lying = endpoint("optimistic-reasoning", 1e-6, 1e-6, 0.1, 100.0);
    lying
        .capabilities
        .rejected_parameters
        .insert("reasoning".to_owned());
    lying.capabilities.reasoning = ReasoningSupport::GatewayNormalized;

    let mut request = InferenceRequest::new("x/model", "x", 100);
    request.prompt_tokens = Some(100);
    request.requirements.reasoning = ReasoningEffort::High;

    assert!(matches!(
        build_route_plan(
            RoutingStrategy::FastestResponseCheap(1.5),
            &request,
            vec![lying],
        ),
        Err(inference_redox::ProviderError::NoEligibleEndpoint(_))
    ));
}

#[test]
fn verified_parameter_can_fill_missing_provider_claim() -> Result<(), inference_redox::ProviderError>
{
    let mut understated = endpoint("understated-provider", 1e-6, 1e-6, 0.1, 100.0);
    understated
        .capabilities
        .supported_parameters
        .remove("structured_outputs");
    understated
        .capabilities
        .verified_parameters
        .insert("structured_outputs".to_owned());

    let mut request = InferenceRequest::new("x/model", "x", 100);
    request.prompt_tokens = Some(100);
    request.requirements = RequestRequirements {
        response_schema: Some(ResponseSchema::JsonSchema {
            name: "x".into(),
            schema: json!({"type":"object"}),
            strict: true,
        }),
        ..RequestRequirements::default()
    };

    let plan = build_route_plan(
        RoutingStrategy::FastestResponseCheap(1.5),
        &request,
        vec![understated],
    )?;
    assert_eq!(plan.candidates.len(), 1);
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

#[test]
fn pricing_override_changes_shape_cost() {
    let pricing = Pricing {
        prompt_per_token: 1.0e-6,
        completion_per_token: 2.0e-6,
        overrides: vec![inference_redox::PricingOverride {
            min_prompt_tokens: 1_000,
            prompt_per_token: Some(3.0e-6),
            completion_per_token: Some(4.0e-6),
            ..inference_redox::PricingOverride::default()
        }],
        ..Pricing::default()
    };
    assert_eq!(pricing.effective(999).prompt_per_token, 1.0e-6);
    assert_eq!(pricing.effective(1_000).prompt_per_token, 3.0e-6);
    assert!(pricing.expected_cost(1_000, 100) > pricing.expected_cost(999, 100));
}

#[test]
fn expected_output_is_not_a_hard_limit() -> Result<(), inference_redox::ProviderError> {
    let mut request = InferenceRequest::new("x/model", "hello", 100);
    request.prompt_tokens = Some(100);
    request.max_output_tokens = Some(200);
    let plan = build_route_plan(
        RoutingStrategy::FastestResponseCheap(1.5),
        &request,
        vec![endpoint("ok", 1e-6, 1e-6, 0.1, 100.0)],
    )?;
    assert_eq!(plan.expected_output_tokens, 100);

    request.max_output_tokens = Some(99);
    assert!(matches!(
        build_route_plan(
            RoutingStrategy::FastestResponseCheap(1.5),
            &request,
            vec![endpoint("bad", 1e-6, 1e-6, 0.1, 100.0)],
        ),
        Err(inference_redox::ProviderError::InvalidRequest(_))
    ));
    Ok(())
}

#[test]
fn missing_route_class_policy_deserializes_to_interactive_default() -> Result<(), serde_json::Error>
{
    let request: InferenceRequest = serde_json::from_value(json!({
        "model": "x/model",
        "prompt": "x",
        "expected_output_tokens": 8,
        "requirements": {
            "response_schema": null,
            "reasoning": "none",
            "required_parameters": [],
            "minimum_uptime_30m": null
        }
    }))?;
    assert_eq!(
        request.requirements.allowed_route_classes,
        BTreeSet::from([
            inference_redox::ModelRouteClass::Standard,
            inference_redox::ModelRouteClass::Free,
            inference_redox::ModelRouteClass::Subscription,
        ])
    );
    Ok(())
}

#[test]
fn agentic_route_policy_is_enforced_before_ranking() -> Result<(), inference_redox::ProviderError> {
    let mut agent_only = endpoint("agent-only", 1e-6, 1e-6, 0.1, 100.0);
    agent_only.workload_policy = inference_redox::WorkloadPolicy {
        agentic: inference_redox::AgenticPolicy::Required,
        requires_stable_session: true,
    };
    let mut request = InferenceRequest::new("x/model", "work", 8);
    request.prompt_tokens = Some(8);
    assert!(
        build_route_plan(
            RoutingStrategy::FastestResponseCheap(1.5),
            &request,
            vec![agent_only.clone()],
        )
        .is_err()
    );

    request.agentic = true;
    assert!(
        build_route_plan(
            RoutingStrategy::FastestResponseCheap(1.5),
            &request,
            vec![agent_only.clone()],
        )
        .is_err()
    );

    request.session_id = Some("conversation-42".to_owned());
    let plan = build_route_plan(
        RoutingStrategy::FastestResponseCheap(1.5),
        &request,
        vec![agent_only],
    )?;
    assert_eq!(plan.candidates.len(), 1);
    Ok(())
}

#[test]
fn forbidden_agentic_route_rejects_agentic_workload() {
    let mut non_agent = endpoint("non-agent", 1e-6, 1e-6, 0.1, 100.0);
    non_agent.workload_policy.agentic = inference_redox::AgenticPolicy::Forbidden;
    let mut request = InferenceRequest::new("x/model", "work", 8);
    request.prompt_tokens = Some(8);
    request.agentic = true;
    assert!(
        build_route_plan(
            RoutingStrategy::FastestResponseCheap(1.5),
            &request,
            vec![non_agent],
        )
        .is_err()
    );
}

#[test]
fn subscription_route_uses_discounted_price_and_true_free_route_wins()
-> Result<(), inference_redox::ProviderError> {
    let mut subscription = endpoint("subscription", 1e-3, 2e-3, 0.1, 100.0);
    subscription.route_class = inference_redox::ModelRouteClass::Subscription;
    let subscription_cost = subscription.pricing.expected_cost(10, 10);
    assert!(subscription_cost > 0.0);

    let mut free = endpoint("free", 0.0, 0.0, 0.1, 100.0);
    free.route_class = inference_redox::ModelRouteClass::Free;

    let mut request = InferenceRequest::new("x/model", "work", 10);
    request.prompt_tokens = Some(10);
    request.requirements.allowed_route_classes = BTreeSet::from([
        inference_redox::ModelRouteClass::Free,
        inference_redox::ModelRouteClass::Subscription,
    ]);
    let plan = build_route_plan(
        RoutingStrategy::FastestResponseCheap(1.5),
        &request,
        vec![subscription, free],
    )?;
    assert_eq!(plan.candidates.len(), 1);
    assert_eq!(plan.candidates[0].endpoint.id, "free");
    assert_eq!(plan.candidates[0].expected_cost_usd, 0.0);
    Ok(())
}
