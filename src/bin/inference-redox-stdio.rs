use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use inference_redox::providers::{nvidia::NvidiaNim, openrouter::OpenRouter};
use inference_redox::{
    InferenceProvider, InferenceRequest, ModelCatalog, ProviderError, RetryPolicy, RoutingStrategy,
};
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
    Models,
    ResolveModel,
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
    #[serde(default)]
    request: Option<InferenceRequest>,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    fastest_response_cheap: Option<f64>,
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

#[derive(Clone)]
struct Providers {
    openrouter: Arc<OpenRouter>,
    nvidia: Option<Arc<NvidiaNim>>,
}

impl Providers {
    async fn catalog(&self) -> Result<ModelCatalog, ProviderError> {
        let mut catalog = self.openrouter.models().await?;
        if let Some(nvidia) = self.nvidia.as_ref()
            && let Ok(inventory) = nvidia.inventory(&catalog).await
        {
            catalog.ingest_provider_inventory(&inventory);
        }
        Ok(catalog)
    }
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

async fn write_error(stdout: &Arc<Mutex<tokio::io::Stdout>>, id: Value, error: impl Into<String>) {
    write_json(
        stdout,
        &WireError {
            id,
            ok: false,
            error: error.into(),
        },
    )
    .await;
}

async fn handle(providers: Providers, stdout: Arc<Mutex<tokio::io::Stdout>>, request: WireRequest) {
    let WireRequest {
        id,
        op,
        request,
        query,
        fastest_response_cheap,
        retry,
    } = request;

    match op {
        Operation::Models => match providers.catalog().await {
            Ok(catalog) => {
                write_json(
                    &stdout,
                    &WireSuccess {
                        id,
                        ok: true,
                        result: catalog,
                    },
                )
                .await;
            }
            Err(error) => write_error(&stdout, id, error.to_string()).await,
        },
        Operation::ResolveModel => {
            let Some(query) = query else {
                write_error(&stdout, id, "resolve_model requires query").await;
                return;
            };
            match providers.catalog().await {
                Ok(catalog) => {
                    write_json(
                        &stdout,
                        &WireSuccess {
                            id,
                            ok: true,
                            result: catalog.resolve(&query),
                        },
                    )
                    .await;
                }
                Err(error) => write_error(&stdout, id, error.to_string()).await,
            }
        }
        Operation::Route | Operation::Infer => {
            let Some(request) = request else {
                write_error(&stdout, id, "route/infer requires request").await;
                return;
            };
            let Some(multiplier) = fastest_response_cheap else {
                write_error(&stdout, id, "route/infer requires fastest_response_cheap").await;
                return;
            };
            let strategy = RoutingStrategy::FastestResponseCheap(multiplier);
            match op {
                Operation::Route => {
                    let result = match providers.openrouter.endpoints(&request.model).await {
                        Ok(endpoints) => {
                            inference_redox::build_route_plan(strategy, &request, endpoints)
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
                        Err(error) => write_error(&stdout, id, error.to_string()).await,
                    }
                }
                Operation::Infer => {
                    let retry =
                        retry.map_or_else(RetryPolicy::default, WireRetryPolicy::normalized);
                    match providers.openrouter.infer(strategy, &request, retry).await {
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
                        Err(error) => write_error(&stdout, id, error.to_string()).await,
                    }
                }
                Operation::Models | Operation::ResolveModel => {}
            }
        }
    }
}

fn key_from_contents(contents: &str, name: &str) -> Option<String> {
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() != name {
            continue;
        }
        let value = value.trim().trim_matches(['\'', '"']);
        if !value.is_empty() {
            return Some(value.to_owned());
        }
    }
    None
}

fn key_from_env_file(
    path: &Path,
    name: &str,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    Ok(key_from_contents(&fs::read_to_string(path)?, name))
}

fn environment_key(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.trim().is_empty())
}

fn env_file_argument() -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let Some(arg) = args.next() else {
        return Ok(None);
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
    Ok(Some(path.into()))
}

fn load_api_keys() -> Result<(String, Option<String>), Box<dyn std::error::Error>> {
    let env_file = env_file_argument()?;
    let from_file = |name: &str| -> Result<Option<String>, Box<dyn std::error::Error>> {
        env_file
            .as_deref()
            .map_or(Ok(None), |path| key_from_env_file(path, name))
    };
    let openrouter = environment_key("OPENROUTER_API_KEY")
        .or(from_file("OPENROUTER_API_KEY")?)
        .ok_or(
            "OPENROUTER_API_KEY is unset; pass --env-file <path> or set the environment variable",
        )?;
    let nvidia = environment_key("NVIDIA_API_KEY").or(from_file("NVIDIA_API_KEY")?);
    Ok((openrouter, nvidia))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (openrouter_key, nvidia_key) = load_api_keys()?;
    let providers = Providers {
        openrouter: Arc::new(OpenRouter::new(openrouter_key)),
        nvidia: nvidia_key.map(|key| Arc::new(NvidiaNim::new(key))),
    };
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
                tasks.spawn(handle(providers.clone(), Arc::clone(&stdout), request));
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

#[cfg(test)]
mod tests {
    use super::key_from_contents;

    #[test]
    fn env_file_parser_handles_export_quotes_and_crlf() {
        let contents = "export OPENROUTER_API_KEY='or-key'\r\nNVIDIA_API_KEY=\"nv-key\"\r\n";
        assert_eq!(
            key_from_contents(contents, "OPENROUTER_API_KEY").as_deref(),
            Some("or-key")
        );
        assert_eq!(
            key_from_contents(contents, "NVIDIA_API_KEY").as_deref(),
            Some("nv-key")
        );
    }
}
