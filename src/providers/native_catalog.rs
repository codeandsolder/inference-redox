use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::Client;
use tokio::sync::Mutex;

use crate::{NativeModelRecord, ProviderError};

/// Connection configuration and refresh state shared by native model catalogs.
/// A clone shares cached records, exactly as the individual adapters did.
#[derive(Debug, Clone)]
pub(super) struct NativeCatalog<K> {
    pub(super) client: Client,
    pub(super) api_key: K,
    pub(super) base_url: String,
    ttl: Duration,
    cached: Arc<Mutex<Option<CachedModels>>>,
}

#[derive(Debug, Clone)]
struct CachedModels {
    fetched_at: Instant,
    models: Vec<NativeModelRecord>,
}

impl<K> NativeCatalog<K> {
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

    /// Return fresh cached data, refresh without holding the lock across I/O,
    /// and fall back to the previous successful snapshot on refresh failure.
    pub(super) async fn models(
        &self,
        fetch: impl std::future::Future<Output = Result<Vec<NativeModelRecord>, ProviderError>>,
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
    use std::cell::Cell;
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
        let polled = Cell::new(false);
        let result = clone
            .models(async {
                polled.set(true);
                Ok(records("unexpected"))
            })
            .await?;
        assert!(!polled.get());
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
