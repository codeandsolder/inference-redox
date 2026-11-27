use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, de::DeserializeOwned};
use serde_json::Value;
use tokio::sync::Mutex;
use wreq::Client;

use super::openai_compatible::execute_chat_with_retry;
use crate::{
    InferenceRequest, InferenceResponse, NativeModelRecord, ProviderError, RetryPolicy,
    RouteCandidate, RoutePlan,
};

/// Connection configuration and refresh state shared by native model catalogs.
/// A clone shares cached records, exactly as the individual adapters did.
#[derive(Clone)]
pub(super) struct NativeCatalog<K> {
    pub(super) client: Client,
    pub(super) api_key: K,
    pub(super) base_url: String,
    ttl: Duration,
    cached: Arc<Mutex<Option<CachedModels>>>,
}

impl<K> std::fmt::Debug for NativeCatalog<K> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeCatalog")
            .field("base_url", &self.base_url)
            .field("ttl", &self.ttl)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
struct CachedModels {
    fetched_at: Instant,
    models: Vec<NativeModelRecord>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ModelEnvelope<M> {
    pub(super) data: Vec<M>,
}

pub(super) fn select_route<'a>(
    request: &InferenceRequest,
    plan: &'a RoutePlan,
    source: &str,
    mismatch: impl FnOnce() -> String,
) -> Result<&'a RouteCandidate, ProviderError> {
    let Some(candidate) = plan.candidates.first() else {
        return Err(ProviderError::NoEligibleEndpoint(request.model.clone()));
    };
    if candidate.endpoint.source != source || candidate.endpoint.provider_key != source {
        return Err(ProviderError::InvalidRequest(mismatch()));
    }
    Ok(candidate)
}

impl<K: Send + Sync> NativeCatalog<K> {
    pub(super) fn new(api_key: K, base_url: &str) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: base_url.to_owned(),
            ttl: Duration::from_secs(60),
            cached: Arc::new(Mutex::new(None)),
        }
    }

    pub(super) fn set_base_url(&mut self, base_url: impl Into<String>) {
        base_url
            .into()
            .trim_end_matches('/')
            .clone_into(&mut self.base_url);
    }

    pub(super) const fn set_ttl(&mut self, ttl: Duration) {
        self.ttl = ttl;
    }

    pub(super) async fn fetch_models<M: DeserializeOwned + Send>(
        &self,
        api_key: Option<&str>,
        include_error_body: bool,
    ) -> Result<Vec<M>, ProviderError> {
        let request = self
            .client
            .get(format!("{}/models", self.base_url))
            .redirect(wreq::redirect::Policy::limited(10));
        let request = if let Some(key) = api_key {
            request.bearer_auth(key)
        } else {
            request
        };
        let response = request
            .send()
            .await
            .map_err(|error| ProviderError::Catalog(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let detail = if include_error_body {
                format!(
                    "HTTP {status}: {}",
                    response.text().await.unwrap_or_default()
                )
            } else {
                format!("HTTP {status}")
            };
            return Err(ProviderError::Catalog(detail));
        }
        let envelope: ModelEnvelope<M> = response
            .json()
            .await
            .map_err(|error| ProviderError::Catalog(error.to_string()))?;
        Ok(envelope.data)
    }

    /// Free text-only adapters all use the same route check, request and response transport.
    pub(super) async fn execute_free_chat(
        &self,
        api_key: Option<&str>,
        request: &InferenceRequest,
        plan: &RoutePlan,
        retry: RetryPolicy,
        provider: (&str, &str),
        compile_body: impl FnOnce(&InferenceRequest) -> Result<Value, ProviderError> + Send,
    ) -> Result<InferenceResponse, ProviderError> {
        select_route(request, plan, provider.0, || provider.1.to_owned())?;
        let body = compile_body(request)?;
        execute_chat_with_retry(
            &self.client,
            api_key,
            &format!("{}/chat/completions", self.base_url),
            &body,
            provider.0,
            Some(0.0),
            retry,
        )
        .await
    }

    /// Return fresh cached data, refresh without holding the lock across I/O,
    /// and fall back to the previous successful snapshot on refresh failure.
    pub(super) async fn models(
        &self,
        fetch: impl std::future::Future<Output = Result<Vec<NativeModelRecord>, ProviderError>> + Send,
    ) -> Result<Vec<NativeModelRecord>, ProviderError> {
        let cache = self.cached.lock().await;
        if let Some(cached) = cache.as_ref()
            && cached.fetched_at.elapsed() <= self.ttl
        {
            return Ok(cached.models.clone());
        }
        let stale = cache.as_ref().map(|cached| cached.models.clone());
        drop(cache);

        match fetch.await {
            Ok(models) => {
                *self.cached.lock().await = Some(CachedModels {
                    fetched_at: Instant::now(),
                    models: models.clone(),
                });
                Ok(models)
            }
            Err(error) => stale.ok_or(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::future::ready;

    use super::*;

    fn records(id: &str) -> Vec<NativeModelRecord> {
        vec![NativeModelRecord {
            id: id.to_owned(),
            ..NativeModelRecord::default()
        }]
    }

    #[tokio::test]
    async fn fresh_cache_is_shared_and_does_not_poll_fetch() -> Result<(), ProviderError> {
        let catalog = NativeCatalog::new((), "https://catalog.invalid");
        assert_eq!(
            catalog.models(ready(Ok(records("first")))).await?,
            records("first")
        );
        let clone = catalog.clone();
        let mut polled = false;
        let result = clone
            .models(async {
                polled = true;
                Ok(records("unexpected"))
            })
            .await?;
        assert!(!polled);
        assert_eq!(result, records("first"));
        Ok(())
    }

    #[tokio::test]
    async fn refresh_preserves_stale_data_and_replaces_successful_snapshot()
    -> Result<(), Box<dyn std::error::Error>> {
        let catalog = NativeCatalog::new((), "https://catalog.invalid");
        let error = || ProviderError::Catalog("unavailable".to_owned());
        assert!(matches!(
            catalog.models(ready(Err(error()))).await,
            Err(ProviderError::Catalog(_))
        ));
        assert_eq!(
            catalog.models(ready(Ok(records("first")))).await?,
            records("first")
        );
        {
            let mut cache = catalog.cached.lock().await;
            cache.as_mut().ok_or("missing populated cache")?.fetched_at = Instant::now()
                .checked_sub(Duration::from_secs(120))
                .ok_or("cannot represent expired timestamp")?;
        }
        assert_eq!(catalog.models(ready(Err(error()))).await?, records("first"));
        let refreshed = catalog
            .models(async {
                assert!(catalog.cached.try_lock().is_ok());
                Ok(records("second"))
            })
            .await?;
        assert_eq!(refreshed, records("second"));
        assert_eq!(
            catalog.models(ready(Err(error()))).await?,
            records("second")
        );
        Ok(())
    }
}
