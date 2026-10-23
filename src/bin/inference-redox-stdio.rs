use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use inference_redox::providers::{groq::Groq, nvidia::NvidiaNim, openrouter::OpenRouter};
use inference_redox::{
    BatchInferenceItem, InferencePool, InferenceProvider, InferenceRequest, ModelCatalog,
    ModelSuitabilityRequirements, PooledInferenceResponse, ProviderError, ProviderInventory,
    QuotaExecutionContext, QuotaGraph, QuotaLedger, QuotaObservationMap, QuotaUnknownPolicy,
    RetryPolicy, RoutingStrategy, SharedQuotaLedger,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;
use tokio::task::JoinSet;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Operation {
    Infer,
    Route,
    InferPool,
    RoutePool,
    Models,
    ProviderInventory,
    ResolveModel,
    Suitability,
    QuotaState,
    SubmitBatch,
    BatchStatus,
    DeleteBatch,
}

#[derive(Debug, Clone, Copy, Deserialize)]
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
    provider: Option<String>,
    #[serde(default)]
    suitability: Option<ModelSuitabilityRequirements>,
    #[serde(default)]
    batch_model: Option<String>,
    #[serde(default)]
    batch_items: Option<Vec<BatchInferenceItem>>,
    #[serde(default)]
    batch_provider: Option<String>,
    #[serde(default)]
    batch_id: Option<String>,
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
    groq: Option<Arc<Groq>>,
    quota: SharedQuotaLedger,
}

impl Providers {
    async fn catalog(&self) -> Result<ModelCatalog, ProviderError> {
        let mut catalog = self.openrouter.models().await?;
        if let Some(nvidia) = self.nvidia.as_ref()
            && let Ok(inventory) = nvidia.inventory(&catalog).await
        {
            catalog.ingest_provider_inventory(&inventory);
        }
        if let Some(groq) = self.groq.as_ref()
            && let Ok(inventory) = groq.inventory(&catalog).await
        {
            catalog.ingest_provider_inventory(&inventory);
        }
        Ok(catalog)
    }

    async fn provider_inventory(&self, provider: &str) -> Result<ProviderInventory, ProviderError> {
        let catalog = self.openrouter.models().await?;
        match provider {
            "nvidia" => {
                self.nvidia
                    .as_ref()
                    .ok_or_else(|| {
                        ProviderError::InvalidRequest(
                            "NVIDIA provider is not configured".to_owned(),
                        )
                    })?
                    .inventory(&catalog)
                    .await
            }
            "groq" => {
                self.groq
                    .as_ref()
                    .ok_or_else(|| {
                        ProviderError::InvalidRequest("Groq provider is not configured".to_owned())
                    })?
                    .inventory(&catalog)
                    .await
            }
            other => Err(ProviderError::InvalidRequest(format!(
                "provider_inventory does not support {other}"
            ))),
        }
    }

    fn quota_ledger(
        groq_configured: bool,
        groq_quota_pool_label: &str,
    ) -> Result<SharedQuotaLedger, ProviderError> {
        let mut graph = QuotaGraph::new();
        let mut observations = QuotaObservationMap::new();
        if groq_configured {
            for model in Groq::FREE_PLAN_CHAT_MODELS {
                Groq::bind_free_plan_rate_limits(
                    &mut graph,
                    &mut observations,
                    groq_quota_pool_label,
                    None,
                    model,
                )?;
            }
        }
        Ok(SharedQuotaLedger::new(QuotaLedger::with_observation_map(
            graph,
            observations,
        )))
    }

    fn pool(&self) -> InferencePool {
        let mut pool = InferencePool::new();
        pool.register("openrouter", Arc::clone(&self.openrouter));
        if let Some(nvidia) = self.nvidia.as_ref() {
            pool.register("nvidia", Arc::clone(nvidia));
        }
        if let Some(groq) = self.groq.as_ref() {
            pool.register("groq", Arc::clone(groq));
        }
        pool
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

async fn handle_catalog_operation(
    providers: &Providers,
    stdout: &Arc<Mutex<tokio::io::Stdout>>,
    wire: WireRequest,
) {
    let WireRequest {
        id,
        op,
        query,
        provider,
        suitability,
        ..
    } = wire;
    match op {
        Operation::Models => match providers.catalog().await {
            Ok(catalog) => {
                write_json(
                    stdout,
                    &WireSuccess {
                        id,
                        ok: true,
                        result: catalog,
                    },
                )
                .await;
            }
            Err(error) => write_error(stdout, id, error.to_string()).await,
        },
        Operation::ProviderInventory => {
            let Some(provider) = provider else {
                write_error(stdout, id, "provider_inventory requires provider").await;
                return;
            };
            match providers.provider_inventory(&provider).await {
                Ok(inventory) => {
                    write_json(
                        stdout,
                        &WireSuccess {
                            id,
                            ok: true,
                            result: inventory,
                        },
                    )
                    .await;
                }
                Err(error) => write_error(stdout, id, error.to_string()).await,
            }
        }
        Operation::QuotaState => {
            write_json(
                stdout,
                &WireSuccess {
                    id,
                    ok: true,
                    result: providers.quota.snapshot().await,
                },
            )
            .await;
        }
        Operation::Suitability => {
            let Some(requirements) = suitability else {
                write_error(stdout, id, "suitability requires suitability").await;
                return;
            };
            match providers.catalog().await {
                Ok(catalog) => {
                    write_json(
                        stdout,
                        &WireSuccess {
                            id,
                            ok: true,
                            result: catalog.suitability_reports(&requirements),
                        },
                    )
                    .await;
                }
                Err(error) => write_error(stdout, id, error.to_string()).await,
            }
        }
        Operation::ResolveModel => {
            let Some(query) = query else {
                write_error(stdout, id, "resolve_model requires query").await;
                return;
            };
            match providers.catalog().await {
                Ok(catalog) => {
                    write_json(
                        stdout,
                        &WireSuccess {
                            id,
                            ok: true,
                            result: catalog.resolve(&query),
                        },
                    )
                    .await;
                }
                Err(error) => write_error(stdout, id, error.to_string()).await,
            }
        }
        _ => {}
    }
}

async fn handle_batch_operation(
    providers: &Providers,
    stdout: &Arc<Mutex<tokio::io::Stdout>>,
    wire: WireRequest,
) {
    let WireRequest {
        id,
        op,
        batch_model,
        batch_items,
        batch_provider,
        batch_id,
        ..
    } = wire;
    match op {
        Operation::SubmitBatch => {
            let Some(batch_model) = batch_model else {
                write_error(stdout, id, "submit_batch requires batch_model").await;
                return;
            };
            let Some(batch_items) = batch_items else {
                write_error(stdout, id, "submit_batch requires batch_items").await;
                return;
            };
            let result = if let Some(provider) = batch_provider.as_deref() {
                providers
                    .openrouter
                    .submit_batch_to_provider(&batch_model, provider, &batch_items)
                    .await
            } else {
                providers
                    .openrouter
                    .submit_batch(&batch_model, &batch_items)
                    .await
            };
            match result {
                Ok(batch) => {
                    write_json(
                        stdout,
                        &WireSuccess {
                            id,
                            ok: true,
                            result: batch,
                        },
                    )
                    .await;
                }
                Err(error) => write_error(stdout, id, error.to_string()).await,
            }
        }
        Operation::BatchStatus => {
            let Some(batch_id) = batch_id else {
                write_error(stdout, id, "batch_status requires batch_id").await;
                return;
            };
            match providers.openrouter.batch(&batch_id).await {
                Ok(batch) => {
                    write_json(
                        stdout,
                        &WireSuccess {
                            id,
                            ok: true,
                            result: batch,
                        },
                    )
                    .await;
                }
                Err(error) => write_error(stdout, id, error.to_string()).await,
            }
        }
        Operation::DeleteBatch => {
            let Some(batch_id) = batch_id else {
                write_error(stdout, id, "delete_batch requires batch_id").await;
                return;
            };
            match providers.openrouter.delete_batch(&batch_id).await {
                Ok(()) => {
                    write_json(
                        stdout,
                        &WireSuccess {
                            id,
                            ok: true,
                            result: serde_json::json!({"deleted": true}),
                        },
                    )
                    .await;
                }
                Err(error) => write_error(stdout, id, error.to_string()).await,
            }
        }
        _ => {}
    }
}

fn unix_time_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

async fn handle_pool_operation(
    providers: &Providers,
    stdout: &Arc<Mutex<tokio::io::Stdout>>,
    wire: WireRequest,
) {
    let WireRequest {
        id,
        op,
        request,
        fastest_response_cheap,
        retry,
        ..
    } = wire;
    let Some(request) = request else {
        write_error(stdout, id, "route_pool/infer_pool requires request").await;
        return;
    };
    let Some(multiplier) = fastest_response_cheap else {
        write_error(
            stdout,
            id,
            "route_pool/infer_pool requires fastest_response_cheap",
        )
        .await;
        return;
    };
    let catalog = match providers.catalog().await {
        Ok(catalog) => catalog,
        Err(error) => {
            write_error(stdout, id, error.to_string()).await;
            return;
        }
    };
    let pool = providers.pool();
    let strategy = RoutingStrategy::FastestResponseCheap(multiplier);
    let quota = QuotaExecutionContext {
        unknown_policy: QuotaUnknownPolicy::Allow,
        now_unix: unix_time_seconds(),
    };
    match op {
        Operation::RoutePool => match pool
            .route_with_shared_quota(
                &catalog,
                strategy,
                &request,
                &providers.quota,
                quota.unknown_policy,
                quota.now_unix,
            )
            .await
        {
            Ok(route) => {
                write_json(
                    stdout,
                    &WireSuccess {
                        id,
                        ok: true,
                        result: route.plan,
                    },
                )
                .await;
            }
            Err(error) => write_error(stdout, id, error.to_string()).await,
        },
        Operation::InferPool => {
            let retry = retry.map_or_else(RetryPolicy::default, WireRetryPolicy::normalized);
            match pool
                .infer_with_shared_quota(
                    &catalog,
                    strategy,
                    &request,
                    retry,
                    &providers.quota,
                    quota,
                )
                .await
            {
                Ok(result) => {
                    let result = PooledInferenceResponse {
                        plan: result.quota_plan.plan,
                        selected: result.selected,
                        response: result.response,
                    };
                    write_json(
                        stdout,
                        &WireSuccess {
                            id,
                            ok: true,
                            result,
                        },
                    )
                    .await;
                }
                Err(error) => write_error(stdout, id, error.to_string()).await,
            }
        }
        _ => {}
    }
}

async fn handle_openrouter_operation(
    providers: &Providers,
    stdout: &Arc<Mutex<tokio::io::Stdout>>,
    wire: WireRequest,
) {
    let WireRequest {
        id,
        op,
        request,
        fastest_response_cheap,
        retry,
        ..
    } = wire;
    let Some(request) = request else {
        write_error(stdout, id, "route/infer requires request").await;
        return;
    };
    let Some(multiplier) = fastest_response_cheap else {
        write_error(stdout, id, "route/infer requires fastest_response_cheap").await;
        return;
    };
    let strategy = RoutingStrategy::FastestResponseCheap(multiplier);
    match op {
        Operation::Route => {
            let result = match providers.openrouter.endpoints(&request.model).await {
                Ok(endpoints) => inference_redox::build_route_plan(strategy, &request, endpoints),
                Err(error) => Err(error),
            };
            match result {
                Ok(route) => {
                    write_json(
                        stdout,
                        &WireSuccess {
                            id,
                            ok: true,
                            result: route,
                        },
                    )
                    .await;
                }
                Err(error) => write_error(stdout, id, error.to_string()).await,
            }
        }
        Operation::Infer => {
            let retry = retry.map_or_else(RetryPolicy::default, WireRetryPolicy::normalized);
            match providers.openrouter.infer(strategy, &request, retry).await {
                Ok((route, response)) => {
                    write_json(
                        stdout,
                        &WireSuccess {
                            id,
                            ok: true,
                            result: serde_json::json!({"route": route, "response": response}),
                        },
                    )
                    .await;
                }
                Err(error) => write_error(stdout, id, error.to_string()).await,
            }
        }
        _ => {}
    }
}

async fn handle(providers: Providers, stdout: Arc<Mutex<tokio::io::Stdout>>, request: WireRequest) {
    match request.op {
        Operation::Models
        | Operation::ProviderInventory
        | Operation::ResolveModel
        | Operation::Suitability
        | Operation::QuotaState => {
            handle_catalog_operation(&providers, &stdout, request).await;
        }
        Operation::SubmitBatch | Operation::BatchStatus | Operation::DeleteBatch => {
            handle_batch_operation(&providers, &stdout, request).await;
        }
        Operation::RoutePool | Operation::InferPool => {
            handle_pool_operation(&providers, &stdout, request).await;
        }
        Operation::Route | Operation::Infer => {
            handle_openrouter_operation(&providers, &stdout, request).await;
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

struct ApiKeys {
    openrouter: String,
    nvidia: Option<String>,
    groq: Option<String>,
    groq_quota_pool_label: String,
}

fn load_api_keys() -> Result<ApiKeys, Box<dyn std::error::Error>> {
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
    let nvidia = environment_key("NVIDIA_API_KEY")
        .or_else(|| environment_key("NVIDIA_API_key"))
        .or(from_file("NVIDIA_API_KEY")?)
        .or(from_file("NVIDIA_API_key")?);
    let groq = environment_key("GROQ_API_KEY").or(from_file("GROQ_API_KEY")?);
    let groq_quota_pool_label = environment_key("GROQ_QUOTA_POOL_LABEL")
        .or(from_file("GROQ_QUOTA_POOL_LABEL")?)
        .unwrap_or_else(|| "default".to_owned());
    Ok(ApiKeys {
        openrouter,
        nvidia,
        groq,
        groq_quota_pool_label,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ApiKeys {
        openrouter,
        nvidia,
        groq,
        groq_quota_pool_label,
    } = load_api_keys()?;
    let groq = groq.map(|key| Arc::new(Groq::new_free(key)));
    let quota = Providers::quota_ledger(groq.is_some(), &groq_quota_pool_label)?;
    let providers = Providers {
        openrouter: Arc::new(OpenRouter::new(openrouter)),
        nvidia: nvidia.map(|key| Arc::new(NvidiaNim::new(key))),
        groq,
        quota,
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
    use super::{Operation, WireRequest, key_from_contents};
    use serde_json::json;

    #[test]
    fn env_file_parser_handles_export_quotes_and_crlf() {
        let contents = concat!(
            "export OPENROUTER_API_KEY='or-key'\r\n",
            "NVIDIA_API_KEY=\"nv-key\"\r\n",
            "GROQ_API_KEY='groq-key'\r\n",
            "GROQ_QUOTA_POOL_LABEL='pool-a'\r\n",
        );
        assert_eq!(
            key_from_contents(contents, "OPENROUTER_API_KEY").as_deref(),
            Some("or-key")
        );
        assert_eq!(
            key_from_contents(contents, "NVIDIA_API_KEY").as_deref(),
            Some("nv-key")
        );
        assert_eq!(
            key_from_contents(contents, "GROQ_API_KEY").as_deref(),
            Some("groq-key")
        );
        assert_eq!(
            key_from_contents(contents, "GROQ_QUOTA_POOL_LABEL").as_deref(),
            Some("pool-a")
        );
        assert_eq!(
            key_from_contents("NVIDIA_API_key=legacy", "NVIDIA_API_key").as_deref(),
            Some("legacy")
        );
    }

    #[test]
    fn provider_inventory_wire_request_deserializes() -> Result<(), serde_json::Error> {
        let wire: WireRequest = serde_json::from_value(json!({
            "id": 1,
            "op": "provider_inventory",
            "provider": "groq"
        }))?;
        assert!(matches!(wire.op, Operation::ProviderInventory));
        assert_eq!(wire.provider.as_deref(), Some("groq"));
        Ok(())
    }

    #[test]
    fn quota_state_wire_request_deserializes() -> Result<(), serde_json::Error> {
        let wire: WireRequest = serde_json::from_value(json!({
            "id": 1,
            "op": "quota_state"
        }))?;
        assert!(matches!(wire.op, Operation::QuotaState));
        Ok(())
    }

    #[test]
    fn suitability_wire_request_deserializes() -> Result<(), serde_json::Error> {
        let wire: WireRequest = serde_json::from_value(json!({
            "id": 1,
            "op": "suitability",
            "suitability": {"minimum_coding_index": 50.0}
        }))?;
        assert!(matches!(wire.op, Operation::Suitability));
        assert_eq!(
            wire.suitability
                .and_then(|value| value.minimum_coding_index),
            Some(50.0)
        );
        Ok(())
    }
}
