use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use inference_redox::providers::{
    groq::GroqCloud, kilo::KiloGateway, nvidia::NvidiaNim, openrouter::OpenRouter,
};
use inference_redox::{
    BatchInferenceItem, BenchmarkCatalog, BenchmarkSource, BenchmarkTaskType, ConformanceProbe,
    ConformanceProfile, InferencePool, InferenceProvider, InferenceRequest, ModelCatalog,
    ModelLookup, ModelSuitabilityReport, ModelSuitabilityRequirements, ProviderError,
    QuotaExecutionContext, QuotaGraph, QuotaGraphSnapshot, QuotaLedger, QuotaObservationBinding,
    QuotaUnknownPolicy, RetryPolicy, RoutingStrategy,
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
    InferPoolQuota,
    RoutePoolQuota,
    QuotaState,
    Models,
    ResolveModel,
    Suitability,
    Benchmarks,
    Conformance,
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
struct WireConformanceRequest {
    model: String,
    source: String,
    #[serde(default)]
    route_id: Option<String>,
    #[serde(default)]
    provider_key: Option<String>,
    #[serde(default)]
    refresh: bool,
    probes: Vec<ConformanceProbe>,
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
    suitability: Option<ModelSuitabilityRequirements>,
    #[serde(default)]
    benchmark_source: Option<BenchmarkSource>,
    #[serde(default)]
    benchmark_task_type: Option<BenchmarkTaskType>,
    #[serde(default)]
    conformance: Option<WireConformanceRequest>,
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
    #[serde(default)]
    quota_unknown_policy: Option<QuotaUnknownPolicy>,
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

#[derive(Debug, Serialize)]
struct WireQuotaState {
    graph: QuotaGraphSnapshot,
    observation_bindings: Vec<QuotaObservationBinding>,
}

#[derive(Clone)]
struct Providers {
    openrouter: Arc<OpenRouter>,
    nvidia: Option<Arc<NvidiaNim>>,
    groq: Option<Arc<GroqCloud>>,
    kilo: Arc<KiloGateway>,
    conformance: Arc<Mutex<ConformanceProfile>>,
    conformance_probe_lock: Arc<Mutex<()>>,
    conformance_cache_path: Arc<PathBuf>,
    quota: Arc<Mutex<QuotaLedger>>,
}

impl Providers {
    async fn catalog(&self) -> Result<ModelCatalog, ProviderError> {
        let base_catalog = self.openrouter.models().await?;
        let mut inventories = Vec::new();
        if let Some(nvidia) = self.nvidia.as_ref()
            && let Ok(inventory) = nvidia.inventory(&base_catalog).await
        {
            inventories.push(inventory);
        }
        if let Some(groq) = self.groq.as_ref()
            && let Ok(inventory) = groq.inventory(&base_catalog).await
        {
            inventories.push(inventory);
        }
        if let Ok(inventory) = self.kilo.inventory(&base_catalog).await {
            inventories.push(inventory);
        }
        let mut catalog = base_catalog;
        for inventory in &inventories {
            catalog.ingest_provider_inventory(inventory);
        }
        Ok(catalog)
    }

    async fn pool(&self) -> InferencePool {
        let profile = self.conformance.lock().await.clone();
        let mut pool = InferencePool::new().with_conformance_profile(profile);
        pool.register("openrouter", Arc::clone(&self.openrouter));
        if let Some(nvidia) = self.nvidia.as_ref() {
            pool.register("nvidia", Arc::clone(nvidia));
        }
        if let Some(groq) = self.groq.as_ref() {
            pool.register("groq", Arc::clone(groq));
        }
        pool.register("kilo", Arc::clone(&self.kilo));
        pool
    }

    fn provider(&self, source: &str) -> Option<Arc<dyn InferenceProvider>> {
        match source {
            "openrouter" => Some(self.openrouter.clone()),
            "nvidia" => self
                .nvidia
                .as_ref()
                .map(|provider| Arc::clone(provider) as Arc<dyn InferenceProvider>),
            "groq" => self
                .groq
                .as_ref()
                .map(|provider| Arc::clone(provider) as Arc<dyn InferenceProvider>),
            "kilo" => Some(self.kilo.clone()),
            _ => None,
        }
    }
}

fn unix_time_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

async fn suitability_reports(
    providers: &Providers,
    requirements: &ModelSuitabilityRequirements,
) -> Result<Vec<ModelSuitabilityReport>, ProviderError> {
    let catalog = providers.catalog().await?;
    let mut benchmarks = BenchmarkCatalog::default();
    for (source, task_type) in requirements.benchmark_queries() {
        benchmarks.ingest(providers.openrouter.benchmarks(source, task_type).await?);
    }
    Ok(catalog.suitability_reports_with_benchmarks(requirements, &benchmarks))
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

async fn run_conformance(
    providers: &Providers,
    spec: WireConformanceRequest,
) -> Result<(Vec<inference_redox::ConformanceObservation>, usize), ProviderError> {
    if spec.probes.is_empty() {
        return Err(ProviderError::InvalidRequest(
            "conformance requires at least one probe".to_owned(),
        ));
    }
    let catalog = providers.catalog().await?;
    let resolved = match catalog.resolve(&spec.model) {
        ModelLookup::Resolved { model } => model,
        ModelLookup::Ambiguous { canonical_ids } => {
            return Err(ProviderError::InvalidRequest(format!(
                "ambiguous model {}: {}",
                spec.model,
                canonical_ids.join(", ")
            )));
        }
        ModelLookup::NotFound => {
            return Err(ProviderError::NoEligibleEndpoint(spec.model));
        }
    };
    let matching_routes = resolved
        .routes
        .iter()
        .filter(|route| route.source == spec.source)
        .filter(|route| spec.route_id.as_ref().is_none_or(|id| &route.id == id))
        .collect::<Vec<_>>();
    let [route] = matching_routes.as_slice() else {
        return Err(ProviderError::InvalidRequest(format!(
            "conformance route selection for {} on {} matched {} routes; specify route_id",
            spec.model,
            spec.source,
            matching_routes.len()
        )));
    };
    let provider = providers.provider(&spec.source).ok_or_else(|| {
        ProviderError::InvalidRequest(format!(
            "provider source is not configured: {}",
            spec.source
        ))
    })?;
    let matching_endpoints = provider
        .endpoints(&route.id)
        .await?
        .into_iter()
        .filter(|endpoint| {
            spec.provider_key
                .as_ref()
                .is_none_or(|provider_key| &endpoint.provider_key == provider_key)
        })
        .collect::<Vec<_>>();
    let [endpoint] = matching_endpoints.as_slice() else {
        return Err(ProviderError::InvalidRequest(format!(
            "conformance endpoint selection matched {} endpoints; specify provider_key",
            matching_endpoints.len()
        )));
    };
    let cached_profile = providers.conformance.lock().await.clone();
    let mut observations = Vec::with_capacity(spec.probes.len());
    let mut executed = 0_usize;
    for probe in spec.probes {
        if !spec.refresh
            && let Some(cached) = cached_profile.cached_probe(endpoint, probe)
        {
            observations.push(cached.clone());
            continue;
        }
        executed = executed.saturating_add(1);
        observations.push(
            probe
                .run_provider_model(
                    provider.as_ref(),
                    endpoint.clone(),
                    Some(resolved.canonical_id.clone()),
                )
                .await,
        );
    }
    Ok((observations, executed))
}

async fn handle_conformance_operation(
    providers: &Providers,
    stdout: &Arc<Mutex<tokio::io::Stdout>>,
    wire: WireRequest,
) {
    let id = wire.id;
    let Some(spec) = wire.conformance else {
        write_error(stdout, id, "conformance requires conformance").await;
        return;
    };
    let _probe_guard = providers.conformance_probe_lock.lock().await;
    match run_conformance(providers, spec).await {
        Ok((observations, executed)) => {
            if executed > 0 {
                let profile = {
                    let mut profile = providers.conformance.lock().await;
                    for observation in observations.iter().cloned() {
                        profile.record(observation);
                    }
                    profile.clone()
                };
                if let Err(error) =
                    save_conformance_profile(providers.conformance_cache_path.as_ref(), &profile)
                {
                    let detail = format!("failed to persist conformance cache: {error}");
                    write_error(stdout, id, detail).await;
                    return;
                }
            }
            let cached = observations.len().saturating_sub(executed);
            write_json(
                stdout,
                &WireSuccess {
                    id,
                    ok: true,
                    result: serde_json::json!({
                        "observations": observations,
                        "executed": executed,
                        "cached": cached,
                    }),
                },
            )
            .await;
        }
        Err(error) => write_error(stdout, id, error.to_string()).await,
    }
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
        suitability,
        benchmark_source,
        benchmark_task_type,
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
        Operation::Suitability => {
            let Some(requirements) = suitability else {
                write_error(stdout, id, "suitability requires suitability").await;
                return;
            };
            match suitability_reports(providers, &requirements).await {
                Ok(result) => {
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
        Operation::Benchmarks => {
            let Some(source) = benchmark_source else {
                write_error(stdout, id, "benchmarks requires benchmark_source").await;
                return;
            };
            let Some(task_type) = benchmark_task_type else {
                write_error(stdout, id, "benchmarks requires benchmark_task_type").await;
                return;
            };
            match providers.openrouter.benchmarks(source, task_type).await {
                Ok(snapshot) => {
                    write_json(
                        stdout,
                        &WireSuccess {
                            id,
                            ok: true,
                            result: snapshot,
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
    let pool = providers.pool().await;
    let strategy = RoutingStrategy::FastestResponseCheap(multiplier);
    match op {
        Operation::RoutePool => match pool.route(&catalog, strategy, &request).await {
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
        },
        Operation::InferPool => {
            let retry = retry.map_or_else(RetryPolicy::default, WireRetryPolicy::normalized);
            match pool.infer(&catalog, strategy, &request, retry).await {
                Ok(result) => {
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

async fn handle_quota_pool_operation(
    providers: &Providers,
    stdout: &Arc<Mutex<tokio::io::Stdout>>,
    wire: WireRequest,
) {
    let WireRequest {
        id,
        op,
        request,
        fastest_response_cheap,
        quota_unknown_policy,
        retry,
        ..
    } = wire;
    let Some(request) = request else {
        write_error(stdout, id, "quota pool operation requires request").await;
        return;
    };
    let Some(multiplier) = fastest_response_cheap else {
        write_error(
            stdout,
            id,
            "quota pool operation requires fastest_response_cheap",
        )
        .await;
        return;
    };
    let Some(unknown_policy) = quota_unknown_policy else {
        write_error(
            stdout,
            id,
            "quota pool operation requires quota_unknown_policy",
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
    let pool = providers.pool().await;
    let strategy = RoutingStrategy::FastestResponseCheap(multiplier);
    let now_unix = unix_time_seconds();
    let mut ledger = providers.quota.lock().await;
    match op {
        Operation::RoutePoolQuota => {
            match pool
                .route_with_quota(
                    &catalog,
                    strategy,
                    &request,
                    &mut ledger,
                    unknown_policy,
                    now_unix,
                )
                .await
            {
                Ok(result) => {
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
        Operation::InferPoolQuota => {
            let retry = retry.map_or_else(RetryPolicy::default, WireRetryPolicy::normalized);
            let quota = QuotaExecutionContext {
                unknown_policy,
                now_unix,
            };
            match pool
                .infer_with_quota(&catalog, strategy, &request, retry, &mut ledger, quota)
                .await
            {
                Ok(result) => {
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

async fn handle_quota_state_operation(
    providers: &Providers,
    stdout: &Arc<Mutex<tokio::io::Stdout>>,
    id: Value,
) {
    let ledger = providers.quota.lock().await;
    let result = WireQuotaState {
        graph: ledger.graph().snapshot(),
        observation_bindings: ledger.observation_map().snapshot(),
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
        | Operation::ResolveModel
        | Operation::Suitability
        | Operation::Benchmarks => {
            handle_catalog_operation(&providers, &stdout, request).await;
        }
        Operation::Conformance => {
            handle_conformance_operation(&providers, &stdout, request).await;
        }
        Operation::SubmitBatch | Operation::BatchStatus | Operation::DeleteBatch => {
            handle_batch_operation(&providers, &stdout, request).await;
        }
        Operation::RoutePool | Operation::InferPool => {
            handle_pool_operation(&providers, &stdout, request).await;
        }
        Operation::RoutePoolQuota | Operation::InferPoolQuota => {
            handle_quota_pool_operation(&providers, &stdout, request).await;
        }
        Operation::QuotaState => {
            handle_quota_state_operation(&providers, &stdout, request.id).await;
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

fn default_conformance_cache_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(path) = env::var_os("INFERENCE_REDOX_CONFORMANCE_CACHE") {
        return Ok(PathBuf::from(path));
    }
    if let Some(cache_home) = env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(cache_home)
            .join("inference-redox")
            .join("conformance-v1.json"));
    }
    let home = env::var_os("HOME").ok_or(
        "HOME is unset; set INFERENCE_REDOX_CONFORMANCE_CACHE for persistent probe caching",
    )?;
    Ok(PathBuf::from(home)
        .join(".cache")
        .join("inference-redox")
        .join("conformance-v1.json"))
}

fn load_conformance_profile(path: &Path) -> Result<ConformanceProfile, Box<dyn std::error::Error>> {
    match fs::read(path) {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(ConformanceProfile::default())
        }
        Err(error) => Err(error.into()),
    }
}

fn save_conformance_profile(
    path: &Path,
    profile: &ConformanceProfile,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let parent = path
        .parent()
        .ok_or("conformance cache path has no parent directory")?;
    fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("conformance cache path has no UTF-8 filename")?;
    let temporary = parent.join(format!(".{file_name}.{}.tmp", process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(profile)?)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

struct ApiKeys {
    openrouter: String,
    nvidia: Option<String>,
    groq: Option<String>,
    kilo: Option<String>,
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
    let kilo = environment_key("KILO_API_KEY").or(from_file("KILO_API_KEY")?);
    Ok(ApiKeys {
        openrouter,
        nvidia,
        groq,
        kilo,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let keys = load_api_keys()?;
    let conformance_cache_path = default_conformance_cache_path()?;
    let conformance = load_conformance_profile(&conformance_cache_path)?;
    let providers = Providers {
        openrouter: Arc::new(OpenRouter::new(keys.openrouter)),
        nvidia: keys.nvidia.map(|key| Arc::new(NvidiaNim::new(key))),
        groq: keys
            .groq
            .map(|key| Arc::new(GroqCloud::new(key).with_quota_pool("default"))),
        kilo: Arc::new(KiloGateway::new(keys.kilo).with_quota_pool("default-egress")),
        conformance: Arc::new(Mutex::new(conformance)),
        conformance_probe_lock: Arc::new(Mutex::new(())),
        conformance_cache_path: Arc::new(conformance_cache_path),
        quota: Arc::new(Mutex::new(QuotaLedger::new(QuotaGraph::new()))),
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
    use super::{
        BenchmarkSource, BenchmarkTaskType, Operation, WireRequest, key_from_contents,
        load_conformance_profile, save_conformance_profile,
    };
    use inference_redox::{
        ConformanceObservation, ConformanceOutcome, ConformanceProfile, QuotaUnknownPolicy,
    };
    use serde_json::json;
    use std::fs;
    use std::process;

    #[test]
    fn env_file_parser_handles_export_quotes_and_crlf() {
        let contents = concat!(
            "export OPENROUTER_API_KEY='or-key'\r\n",
            "NVIDIA_API_KEY=\"nv-key\"\r\n",
            "GROQ_API_KEY=groq-key\r\n",
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
            key_from_contents("NVIDIA_API_key=legacy", "NVIDIA_API_key").as_deref(),
            Some("legacy")
        );
        assert_eq!(
            key_from_contents(contents, "GROQ_API_KEY").as_deref(),
            Some("groq-key")
        );
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

    #[test]
    fn rich_benchmark_suitability_wire_request_deserializes() -> Result<(), serde_json::Error> {
        let wire: WireRequest = serde_json::from_value(json!({
            "id": 3,
            "op": "suitability",
            "suitability": {
                "benchmark_requirements": [{
                    "source": "design-arena",
                    "task_type": "coding",
                    "metric": "design_arena_elo",
                    "unit": "elo",
                    "minimum": 1300.0,
                    "configuration": null,
                    "qualifiers": {"arena": "models", "category": "codecategories"},
                    "aggregation": "best"
                }]
            }
        }))?;
        assert_eq!(
            wire.suitability
                .as_ref()
                .map(|requirements| requirements.benchmark_queries().len()),
            Some(1)
        );
        assert_eq!(
            wire.suitability
                .map(|requirements| requirements.benchmark_requirements.len()),
            Some(1)
        );
        Ok(())
    }

    #[test]
    fn conformance_wire_request_deserializes() -> Result<(), serde_json::Error> {
        let wire: WireRequest = serde_json::from_value(json!({
            "id": 4,
            "op": "conformance",
            "conformance": {
                "model": "openai/gpt-oss-120b",
                "source": "groq",
                "probes": ["basic_chat", "strict_json_schema", "reasoning_low"]
            }
        }))?;
        assert!(matches!(wire.op, Operation::Conformance));
        assert_eq!(
            wire.conformance.as_ref().map(|spec| spec.probes.len()),
            Some(3)
        );
        assert_eq!(
            wire.conformance.as_ref().map(|spec| spec.refresh),
            Some(false)
        );
        Ok(())
    }

    #[test]
    fn conformance_wire_refresh_is_explicit() -> Result<(), serde_json::Error> {
        let wire: WireRequest = serde_json::from_value(json!({
            "id": 5,
            "op": "conformance",
            "conformance": {
                "model": "openai/gpt-oss-120b",
                "source": "groq",
                "refresh": true,
                "probes": ["basic_chat"]
            }
        }))?;
        assert_eq!(
            wire.conformance.as_ref().map(|spec| spec.refresh),
            Some(true)
        );
        Ok(())
    }

    #[test]
    fn conformance_profile_round_trips_disk_cache() -> Result<(), String> {
        let path = std::env::temp_dir().join(format!(
            "inference-redox-conformance-test-{}.json",
            process::id()
        ));
        let profile = ConformanceProfile {
            observations: vec![ConformanceObservation {
                source: "groq".to_owned(),
                credential: Some("account-a".to_owned()),
                provider_key: "groq".to_owned(),
                endpoint_id: Some("groq".to_owned()),
                native_model_id: "openai/gpt-oss-120b".to_owned(),
                canonical_id: Some("openai/gpt-oss-120b".to_owned()),
                parameter: "chat_completions".to_owned(),
                test_id: "basic-chat".to_owned(),
                test_version: 1,
                outcome: ConformanceOutcome::Passed,
                observed_unix_ms: 123,
                detail: None,
            }],
        };
        save_conformance_profile(&path, &profile).map_err(|error| error.to_string())?;
        let loaded = load_conformance_profile(&path).map_err(|error| error.to_string())?;
        let _ = fs::remove_file(&path);
        assert_eq!(loaded, profile);
        Ok(())
    }

    #[test]
    fn quota_pool_wire_request_requires_explicit_policy_shape() -> Result<(), serde_json::Error> {
        let wire: WireRequest = serde_json::from_value(json!({
            "id": 6,
            "op": "route_pool_quota",
            "fastest_response_cheap": 1.5,
            "quota_unknown_policy": "reject",
            "request": {
                "model": "openai/gpt-oss-120b",
                "prompt": "hi",
                "expected_output_tokens": 8,
                "requirements": {
                    "allowed_route_classes": ["free"],
                    "response_schema": null,
                    "reasoning": "none",
                    "required_parameters": [],
                    "minimum_uptime_30m": null
                }
            }
        }))?;
        assert!(matches!(wire.op, Operation::RoutePoolQuota));
        assert_eq!(wire.quota_unknown_policy, Some(QuotaUnknownPolicy::Reject));
        Ok(())
    }

    #[test]
    fn benchmarks_wire_request_deserializes() -> Result<(), serde_json::Error> {
        let wire: WireRequest = serde_json::from_value(json!({
            "id": 2,
            "op": "benchmarks",
            "benchmark_source": "artificial-analysis",
            "benchmark_task_type": "coding"
        }))?;
        assert!(matches!(wire.op, Operation::Benchmarks));
        assert_eq!(
            wire.benchmark_source,
            Some(BenchmarkSource::ArtificialAnalysis)
        );
        assert_eq!(wire.benchmark_task_type, Some(BenchmarkTaskType::Coding));
        Ok(())
    }
}
