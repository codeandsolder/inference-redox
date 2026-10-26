use std::{collections::BTreeSet, env, fs, io::Write, sync::Arc};

use inference_redox::providers::{
    groq::GroqCloud, kilo::KiloGateway, nvidia::NvidiaNim, openrouter::OpenRouter,
};
use inference_redox::{
    AgenticPolicy, Endpoint, InferenceProvider, InferenceRequest, RequestRequirements,
    ResponseSchema, RetryPolicy, RouteCandidate, RoutePlan,
};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Clone, Deserialize)]
struct Target {
    source: String,
    route_id: String,
    endpoint_id: Option<String>,
    provider_key: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Case {
    id: String,
    schema: Value,
    target: Value,
    expected_valid: bool,
    prompt: String,
}

fn key_from_file(path: &str, name: &str) -> Option<String> {
    fs::read_to_string(path).ok()?.lines().find_map(|line| {
        let line = line.trim().strip_prefix("export ").unwrap_or(line.trim());
        let (key, value) = line.split_once('=')?;
        (key.trim() == name).then(|| value.trim().trim_matches(['\'', '"']).to_owned())
    })
}

fn env_or_file(name: &str, path: &str) -> Option<String> {
    env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| key_from_file(path, name))
}

fn provider_for(source: &str, env_file: &str) -> Result<Arc<dyn InferenceProvider>, String> {
    match source {
        "openrouter" => env_or_file("OPENROUTER_API_KEY", env_file)
            .map(OpenRouter::new)
            .map(|provider| Arc::new(provider) as Arc<dyn InferenceProvider>)
            .ok_or_else(|| "OPENROUTER_API_KEY missing".to_owned()),
        "groq" => env_or_file("GROQ_API_KEY", env_file)
            .map(|key| GroqCloud::new(key).with_quota_pool("default"))
            .map(|provider| Arc::new(provider) as Arc<dyn InferenceProvider>)
            .ok_or_else(|| "GROQ_API_KEY missing".to_owned()),
        "nvidia" => env_or_file("NVIDIA_API_KEY", env_file)
            .or_else(|| env_or_file("NVIDIA_API_key", env_file))
            .map(NvidiaNim::new)
            .map(|provider| Arc::new(provider) as Arc<dyn InferenceProvider>)
            .ok_or_else(|| "NVIDIA_API_KEY missing".to_owned()),
        "kilo" => Ok(Arc::new(
            KiloGateway::new(env_or_file("KILO_API_KEY", env_file))
                .with_quota_pool("default-egress"),
        )),
        other => Err(format!("unsupported experimental source: {other}")),
    }
}

fn exact_plan(endpoint: Endpoint, request: &InferenceRequest) -> RoutePlan {
    let prompt_tokens = request.prompt_tokens.unwrap_or(256);
    let expected_cost_usd = endpoint
        .pricing
        .expected_cost(prompt_tokens, request.expected_output_tokens);
    RoutePlan {
        prompt_tokens,
        expected_output_tokens: request.expected_output_tokens,
        cheapest_cost_usd: expected_cost_usd,
        cost_ceiling_usd: expected_cost_usd,
        candidates: vec![RouteCandidate {
            endpoint,
            expected_cost_usd,
            expected_response_p75_seconds: None,
            expected_response_p95_seconds: None,
        }],
    }
}

async fn resolve_endpoint(
    provider: &dyn InferenceProvider,
    target: &Target,
) -> Result<Endpoint, String> {
    let endpoints = provider
        .endpoints(&target.route_id)
        .await
        .map_err(|error| error.to_string())?;
    let matching: Vec<_> = endpoints
        .into_iter()
        .filter(|endpoint| {
            target
                .endpoint_id
                .as_ref()
                .is_none_or(|id| &endpoint.id == id)
                && target
                    .provider_key
                    .as_ref()
                    .is_none_or(|key| &endpoint.provider_key == key)
        })
        .collect();
    match matching.as_slice() {
        [endpoint] => Ok(endpoint.clone()),
        _ => Err(format!("target matched {} endpoints", matching.len())),
    }
}

async fn run_case(
    provider: Arc<dyn InferenceProvider>,
    endpoint: Endpoint,
    case: Case,
    constrained: bool,
) -> Value {
    let mut request = InferenceRequest::new(endpoint.model.clone(), case.prompt.clone(), 128);
    request.max_output_tokens = Some(1024);
    request.temperature = Some(0.0);
    request.requirements = if constrained {
        RequestRequirements {
            response_schema: Some(ResponseSchema::JsonSchema {
                name: "official_json_schema_suite_case".to_owned(),
                schema: case.schema.clone(),
                strict: true,
            }),
            required_parameters: BTreeSet::from(["structured_outputs".to_owned()]),
            minimum_uptime_30m: None,
            ..RequestRequirements::default()
        }
    } else {
        RequestRequirements {
            minimum_uptime_30m: None,
            ..RequestRequirements::default()
        }
    };
    if endpoint.workload_policy.agentic == AgenticPolicy::Required {
        request.agentic = true;
    }
    if endpoint.workload_policy.requires_stable_session {
        request.session_id = Some("json-schema-suite-experiment".to_owned());
    }
    let plan = exact_plan(endpoint.clone(), &request);
    let retry = RetryPolicy {
        max_attempts: 1,
        attempt_timeout: std::time::Duration::from_secs(45),
        ..RetryPolicy::default()
    };
    match provider.execute_plan(&request, &plan, retry).await {
        Ok(response) => json!({
            "id": case.id,
            "expected_valid": case.expected_valid,
            "target": case.target,
            "endpoint_id": endpoint.id,
            "provider_key": endpoint.provider_key,
            "ok": true,
            "content": response.content,
            "finish_reason": response.raw.pointer("/choices/0/finish_reason"),
            "usage": response.usage,
        }),
        Err(error) => json!({
            "id": case.id,
            "expected_valid": case.expected_valid,
            "target": case.target,
            "endpoint_id": endpoint.id,
            "provider_key": endpoint.provider_key,
            "ok": false,
            "error": error.to_string(),
        }),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if !(6..=7).contains(&args.len()) {
        return Err(format!(
            "usage: {} TARGET_JSON CASES_JSON OUT_JSONL ENV_FILE CONCURRENCY [constrained|unconstrained]",
            args[0]
        )
        .into());
    }
    let constrained = args.get(6).is_none_or(|mode| mode == "constrained");
    if args
        .get(6)
        .is_some_and(|mode| mode != "constrained" && mode != "unconstrained")
    {
        return Err("mode must be constrained or unconstrained".into());
    }
    let target: Target = serde_json::from_slice(&fs::read(&args[1])?)?;
    let mut cases: Vec<Case> = serde_json::from_slice(&fs::read(&args[2])?)?;
    let output_path = args[3].clone();
    let mut completed = BTreeSet::new();
    if let Ok(contents) = fs::read_to_string(&output_path) {
        for line in contents.lines() {
            if let Ok(value) = serde_json::from_str::<Value>(line)
                && let Some(id) = value.get("id").and_then(Value::as_str)
            {
                completed.insert(id.to_owned());
            }
        }
    }
    cases.retain(|case| !completed.contains(&case.id));
    let provider = provider_for(&target.source, &args[4]).map_err(std::io::Error::other)?;
    let endpoint = resolve_endpoint(provider.as_ref(), &target)
        .await
        .map_err(std::io::Error::other)?;
    let concurrency: usize = args[5].parse()?;
    eprintln!(
        "target={} route={} endpoint={} pending={} completed={} constrained={}",
        target.source,
        target.route_id,
        endpoint.id,
        cases.len(),
        completed.len(),
        constrained
    );

    let concurrency = concurrency.max(1);
    let mut output = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&output_path)?;
    let mut tasks = tokio::task::JoinSet::new();
    for case in cases {
        while tasks.len() >= concurrency {
            if let Some(result) = tasks.join_next().await {
                let row = result?;
                let mut bytes = serde_json::to_vec(&row)?;
                bytes.push(b'\n');
                output.write_all(&bytes)?;
                output.flush()?;
            }
        }
        let provider = Arc::clone(&provider);
        let endpoint = endpoint.clone();
        tasks.spawn(async move { run_case(provider, endpoint, case, constrained).await });
    }
    while let Some(result) = tasks.join_next().await {
        let row = result?;
        let mut bytes = serde_json::to_vec(&row)?;
        bytes.push(b'\n');
        output.write_all(&bytes)?;
        output.flush()?;
    }
    Ok(())
}
