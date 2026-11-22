use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufReader, Seek, Write},
    marker::PhantomData,
    path::{Path, PathBuf},
};

use serde::{Serialize, de::DeserializeOwned};
use tempfile::NamedTempFile;
use thiserror::Error;

use crate::{StandingQueueError, StandingTaskLease, StandingTaskQueue};

/// A durable queue in a dedicated directory.
///
/// Independent instances synchronize through a permanent lock file. Each mutation
/// reloads the current validated snapshot, applies the lease-fenced queue operation,
/// and atomically publishes a synced replacement before acknowledging success.
/// The directory and its lock file must not be renamed, removed or modified by
/// external writers while a store is in use.
#[derive(Debug)]
pub struct StandingQueueStore<T> {
    directory: PathBuf,
    task: PhantomData<fn() -> T>,
}

#[derive(Debug, Error)]
pub enum StandingStoreError {
    #[error(transparent)]
    Queue(#[from] StandingQueueError),
    #[error("standing queue I/O: {0}")]
    Io(#[from] io::Error),
    #[error("standing queue snapshot: {0}")]
    Snapshot(#[from] serde_json::Error),
    #[error("snapshot was replaced but directory sync failed; commit is uncertain: {0}")]
    CommitUncertain(io::Error),
}

impl<T: Serialize + DeserializeOwned> StandingQueueStore<T> {
    /// Create a new store in a directory that must not already exist.
    ///
    /// # Errors
    /// Returns an error on existing directories or failed initialization.
    /// A failed initialization may leave the new directory for inspection;
    /// existing state is never silently reset.
    pub fn create(directory: impl AsRef<Path>) -> Result<Self, StandingStoreError> {
        let directory = directory.as_ref();
        let parent = directory
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let parent_handle = File::open(parent)?;
        fs::create_dir(directory)?;
        let store = Self::at_directory(directory)?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(store.lock_path())?;
        lock.lock()?;
        store.persist(&StandingTaskQueue::new())?;
        parent_handle
            .sync_all()
            .map_err(StandingStoreError::CommitUncertain)?;
        drop(lock);
        Ok(store)
    }

    /// Open and validate an existing store.
    ///
    /// # Errors
    /// Missing files, malformed snapshots and invalid queue invariants are errors.
    /// Opening never creates a replacement for missing or corrupt state.
    pub fn open(directory: impl AsRef<Path>) -> Result<Self, StandingStoreError> {
        let store = Self::at_directory(directory.as_ref())?;
        store.snapshot()?;
        Ok(store)
    }

    fn at_directory(directory: &Path) -> Result<Self, StandingStoreError> {
        Ok(Self {
            directory: directory.canonicalize()?,
            task: PhantomData,
        })
    }

    fn lock_path(&self) -> PathBuf {
        self.directory.join("queue.lock")
    }

    fn snapshot_path(&self) -> PathBuf {
        self.directory.join("queue.json")
    }

    fn open_lock(&self) -> Result<File, StandingStoreError> {
        Ok(OpenOptions::new()
            .read(true)
            .write(true)
            .open(self.lock_path())?)
    }

    fn read_snapshot(&self) -> Result<StandingTaskQueue<T>, StandingStoreError> {
        let file = File::open(self.snapshot_path())?;
        Ok(serde_json::from_reader(BufReader::new(file))?)
    }

    fn persist(&self, queue: &StandingTaskQueue<T>) -> Result<(), StandingStoreError> {
        // Open the directory before replacement so an unsupported directory
        // handle or inaccessible parent cannot cause a partial commit.
        let directory = File::open(&self.directory)?;
        let mut temporary = NamedTempFile::new_in(&self.directory)?;
        serde_json::to_writer(temporary.as_file_mut(), queue)?;
        temporary.as_file_mut().flush()?;
        // Validate the serialized form before publication as custom serializers
        // can emit data (for example JSON null for NaN) that T cannot deserialize.
        temporary.as_file_mut().rewind()?;
        let _: StandingTaskQueue<T> =
            serde_json::from_reader(BufReader::new(temporary.as_file_mut()))?;
        temporary.as_file().sync_all()?;
        temporary
            .persist(self.snapshot_path())
            .map_err(|error| error.error)?;
        directory
            .sync_all()
            .map_err(StandingStoreError::CommitUncertain)?;
        Ok(())
    }

    /// Read a coherent, validated snapshot under a shared file lock.
    ///
    /// # Errors
    /// Returns errors on failed locking, I/O or invalid persisted state.
    pub fn snapshot(&self) -> Result<StandingTaskQueue<T>, StandingStoreError> {
        let lock = self.open_lock()?;
        lock.lock_shared()?;
        let snapshot = self.read_snapshot()?;
        drop(lock);
        Ok(snapshot)
    }

    fn mutate<R>(
        &self,
        operation: impl FnOnce(&mut StandingTaskQueue<T>) -> Result<R, StandingQueueError>,
    ) -> Result<R, StandingStoreError> {
        let lock = self.open_lock()?;
        lock.lock()?;
        let mut snapshot = self.read_snapshot()?;
        let result = operation(&mut snapshot)?;
        self.persist(&snapshot)?;
        drop(lock);
        Ok(result)
    }

    /// Insert a uniquely identified pending task and persist it.
    ///
    /// # Errors
    /// Returns queue validation or storage errors.
    pub fn insert(
        &self,
        id: impl Into<String>,
        task: T,
        now_unix: u64,
    ) -> Result<(), StandingStoreError> {
        self.mutate(|queue| queue.insert(id, task, now_unix))
    }

    /// Lease a pending task or replace its expired lease.
    ///
    /// # Errors
    /// Returns queue validation or storage errors. A lease is returned only
    /// after the new snapshot is durable.
    pub fn lease(
        &self,
        id: &str,
        now_unix: u64,
        lease_duration_seconds: u64,
    ) -> Result<StandingTaskLease, StandingStoreError> {
        self.mutate(|queue| queue.lease(id, now_unix, lease_duration_seconds))
    }

    /// Release a currently active lease for retry.
    ///
    /// # Errors
    /// Returns queue validation or storage errors.
    pub fn release(
        &self,
        lease: &StandingTaskLease,
        now_unix: u64,
        last_error: Option<String>,
    ) -> Result<(), StandingStoreError> {
        self.mutate(|queue| queue.release(lease, now_unix, last_error))
    }

    /// Complete a currently active lease.
    ///
    /// # Errors
    /// Returns queue validation or storage errors.
    pub fn complete(
        &self,
        lease: &StandingTaskLease,
        now_unix: u64,
    ) -> Result<(), StandingStoreError> {
        self.mutate(|queue| queue.complete(lease, now_unix))
    }

    /// Cancel a non-terminal task.
    ///
    /// # Errors
    /// Returns queue validation or storage errors.
    pub fn cancel(&self, id: &str, now_unix: u64) -> Result<(), StandingStoreError> {
        self.mutate(|queue| queue.cancel(id, now_unix))
    }

    /// Return expired leases to pending.
    ///
    /// # Errors
    /// Returns storage errors if the resulting snapshot cannot be committed.
    pub fn reap_expired(&self, now_unix: u64) -> Result<usize, StandingStoreError> {
        self.mutate(|queue| Ok(queue.reap_expired(now_unix)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::*;
    use crate::StandingTaskState;

    type TestError = Box<dyn std::error::Error + Send + Sync>;

    #[test]
    fn persisted_lease_and_terminal_state_survive_reopening() -> Result<(), TestError> {
        let root = tempfile::tempdir()?;
        let directory = root.path().join("standing");
        let store = StandingQueueStore::create(&directory)?;
        store.insert("task", String::from("work"), 1)?;
        let old = store.lease("task", 2, 5)?;
        drop(store);

        let reopened = StandingQueueStore::<String>::open(&directory)?;
        assert!(matches!(
            reopened.lease("task", 6, 5),
            Err(StandingStoreError::Queue(
                StandingQueueError::AlreadyLeased { .. }
            ))
        ));
        let new = reopened.lease("task", 7, 5)?;
        assert_ne!(old.lease_id, new.lease_id);
        assert!(matches!(
            reopened.complete(&old, 8),
            Err(StandingStoreError::Queue(StandingQueueError::WrongLease(_)))
        ));
        reopened.complete(&new, 8)?;
        drop(reopened);
        let last = StandingQueueStore::<String>::open(&directory)?.snapshot()?;
        let record = last.get("task").ok_or("task disappeared")?;
        assert_eq!(
            record.state,
            StandingTaskState::Completed {
                completed_at_unix: 8
            }
        );
        assert_eq!(record.attempts, 2);
        Ok(())
    }

    #[test]
    fn independent_handles_cannot_claim_the_same_task() -> Result<(), TestError> {
        let root = tempfile::tempdir()?;
        let directory = root.path().join("standing");
        let store = StandingQueueStore::create(&directory)?;
        store.insert("task", String::from("work"), 1)?;
        let barrier = Arc::new(Barrier::new(2));
        let other = StandingQueueStore::<String>::open(&directory)?;
        let peer_barrier = Arc::clone(&barrier);
        let peer = std::thread::spawn(move || {
            peer_barrier.wait();
            other.lease("task", 2, 10)
        });
        barrier.wait();
        let local = store.lease("task", 2, 10);
        let remote = peer.join().map_err(|_| "peer panicked")?;
        assert_eq!(usize::from(local.is_ok()) + usize::from(remote.is_ok()), 1);
        let snapshot = store.snapshot()?;
        let record = snapshot.get("task").ok_or("task disappeared")?;
        assert_eq!(record.attempts, 1);
        Ok(())
    }

    #[test]
    fn rejected_operation_does_not_rewrite_snapshot() -> Result<(), TestError> {
        let root = tempfile::tempdir()?;
        let store = StandingQueueStore::create(root.path().join("standing"))?;
        store.insert("task", String::from("work"), 1)?;
        let before = fs::read(store.snapshot_path())?;
        assert!(
            store
                .insert("task", String::from("replacement"), 2)
                .is_err()
        );
        assert_eq!(fs::read(store.snapshot_path())?, before);
        Ok(())
    }

    #[test]
    fn non_roundtrippable_payload_cannot_corrupt_store() -> Result<(), TestError> {
        let root = tempfile::tempdir()?;
        let store = StandingQueueStore::<f64>::create(root.path().join("standing"))?;
        store.insert("valid", 1.0, 1)?;
        let before = fs::read(store.snapshot_path())?;
        assert!(matches!(
            store.insert("invalid", f64::NAN, 2),
            Err(StandingStoreError::Snapshot(_))
        ));
        assert_eq!(fs::read(store.snapshot_path())?, before);
        assert_eq!(store.snapshot()?.records().count(), 1);
        Ok(())
    }

    #[test]
    fn open_never_reinitializes_missing_or_invalid_state() -> Result<(), TestError> {
        let root = tempfile::tempdir()?;
        let directory = root.path().join("standing");
        let store = StandingQueueStore::<String>::create(&directory)?;
        assert!(StandingQueueStore::<String>::create(&directory).is_err());
        fs::write(store.snapshot_path(), b"{\"tasks\":{},\"next_lease_id\":0}")?;
        assert!(matches!(
            StandingQueueStore::<String>::open(&directory),
            Err(StandingStoreError::Snapshot(_))
        ));
        fs::remove_file(store.snapshot_path())?;
        assert!(StandingQueueStore::<String>::open(&directory).is_err());
        assert!(store.snapshot().is_err());
        assert!(!store.snapshot_path().exists());
        Ok(())
    }
}
