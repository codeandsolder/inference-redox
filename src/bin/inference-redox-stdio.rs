use std::env;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use inference_redox::providers::openrouter::OpenRouter;
use inference_redox::{InferenceProvider, InferenceRequest, RetryPolicy, RoutingStrategy};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;
use tokio::task::JoinSet;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Operation {
    Infer,
    Route,
}

#[derive(Debug, Deserialize)]
struct WireRetryPolicy {
    max_attempts: Option<u32>,
    initial_backoff_ms: Option<u64>,
    max_backoff_ms: Option<u64>,
}

impl WireRetryPolicy {
    fn normalized(self) -> RetryPolicy {
        let defaults = RetryPolicy::default();
        RetryPolicy {
            max_attempts: self.max_attempts.unwrap_or(defaults.max_attempts),
            initial_backoff: self
                .initial_backoff_ms
                .map_or(defaults.initial_backoff, Duration::from_millis),
            max_backoff: self
                .max_backoff_ms
                .map_or(defaults.max_backoff, Duration::from_millis),
        }
    }
}

#[derive(Debug, Deserialize)]
struct WireRequest {
    id: Value,
    op: Operation,
    request: InferenceRequest,
    fastest_response_cheap: f64,
    retry: Option<WireRetryPolicy>,
}

#[derive(Debug, Serialize)]
struct WireSuccess<T> {
    id: Value,
    ok: bool,
    result: T,
}

#[derive(Debug, Serialize)]
struct WireError {
    id: Value,
    ok: bool,
    error: String,
}

async fn write_json<T: Serialize>(stdout: &Arc<Mutex<tokio::io::Stdout>>, value: &T) {
    let Ok(mut bytes) = serde_json::to_vec(value) else {
        return;
    };
    bytes.push(b'\n');
    let mut output = stdout.lock().await;
    let _ = output.write_all(&bytes).await;
    let _ = output.flush().await;
}

async fn handle(
    provider: Arc<OpenRouter>,
    stdout: Arc<Mutex<tokio::io::Stdout>>,
    request: WireRequest,
) {
    let strategy = RoutingStrategy::FastestResponseCheap(request.fastest_response_cheap);
    let id = request.id;
    match request.op {
        Operation::Route => {
            let result = match provider.endpoints(&request.request.model).await {
                Ok(endpoints) => {
                    inference_redox::build_route_plan(strategy, &request.request, endpoints)
                }
                Err(error) => Err(error),
            };
            match result {
                Ok(route) => {
                    write_json(
                        &stdout,
                        &WireSuccess {
                            id,
                            ok: true,
                            result: route,
                        },
                    )
                    .await;
                }
                Err(error) => {
                    write_json(
                        &stdout,
                        &WireError {
                            id,
                            ok: false,
                            error: error.to_string(),
                        },
                    )
                    .await;
                }
            }
        }
        Operation::Infer => {
            let retry = request
                .retry
                .map_or_else(RetryPolicy::default, WireRetryPolicy::normalized);
            let result = provider.infer(strategy, &request.request, retry).await;
            match result {
                Ok((route, response)) => {
                    write_json(
                        &stdout,
                        &WireSuccess {
                            id,
                            ok: true,
                            result: serde_json::json!({"route": route, "response": response}),
                        },
                    )
                    .await;
                }
                Err(error) => {
                    write_json(
                        &stdout,
                        &WireError {
                            id,
                            ok: false,
                            error: error.to_string(),
                        },
                    )
                    .await;
                }
            }
        }
    }
}

fn key_from_env_file(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(value) = line.strip_prefix("OPENROUTER_API_KEY=") {
            let value = value.trim().trim_matches(['\'', '"']);
            if !value.is_empty() {
                return Ok(value.to_owned());
            }
        }
    }
    Err(format!("OPENROUTER_API_KEY not found in {}", path.display()).into())
}

fn load_api_key() -> Result<String, Box<dyn std::error::Error>> {
    if let Ok(key) = env::var("OPENROUTER_API_KEY")
        && !key.trim().is_empty()
    {
        return Ok(key);
    }
    let mut args = env::args_os().skip(1);
    let Some(arg) = args.next() else {
        return Err(
            "OPENROUTER_API_KEY is unset; pass --env-file <path> or set the environment variable"
                .into(),
        );
    };
    if arg != "--env-file" {
        return Err(format!("unknown argument: {}", arg.to_string_lossy()).into());
    }
    let Some(path) = args.next() else {
        return Err("--env-file requires a path".into());
    };
    if let Some(extra) = args.next() {
        return Err(format!("unexpected argument: {}", extra.to_string_lossy()).into());
    }
    key_from_env_file(Path::new(&path))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = load_api_key()?;
    let provider = Arc::new(OpenRouter::new(api_key));
    let stdout = Arc::new(Mutex::new(tokio::io::stdout()));
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    let mut tasks = JoinSet::new();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<WireRequest>(&line) {
            Ok(request) => {
                tasks.spawn(handle(Arc::clone(&provider), Arc::clone(&stdout), request));
                while tasks.try_join_next().is_some() {}
            }
            Err(error) => {
                write_json(
                    &stdout,
                    &WireError {
                        id: Value::Null,
                        ok: false,
                        error: format!("invalid request: {error}"),
                    },
                )
                .await;
            }
        }
    }
    while tasks.join_next().await.is_some() {}
    Ok(())
}
