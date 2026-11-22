use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::StandingInferenceTask;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum StandingTaskState {
    Pending,
    Leased {
        lease_id: u64,
        leased_at_unix: u64,
        expires_at_unix: u64,
    },
    Completed {
        completed_at_unix: u64,
    },
    Cancelled {
        cancelled_at_unix: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StandingTaskRecord<T> {
    pub id: String,
    pub task: T,
    pub state: StandingTaskState,
    pub attempts: u64,
    pub last_error: Option<String>,
    pub created_at_unix: u64,
    pub updated_at_unix: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StandingTaskLease {
    pub task_id: String,
    pub lease_id: u64,
    pub attempt: u64,
    pub expires_at_unix: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StandingTaskQueue<T> {
    tasks: BTreeMap<String, StandingTaskRecord<T>>,
    next_lease_id: u64,
}

pub type StandingInferenceQueue = StandingTaskQueue<StandingInferenceTask>;

impl<T> Default for StandingTaskQueue<T> {
    fn default() -> Self {
        Self {
            tasks: BTreeMap::new(),
            next_lease_id: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StandingQueueError {
    #[error("standing task id must not be empty")]
    EmptyTaskId,
    #[error("standing task already exists: {0}")]
    DuplicateTask(String),
    #[error("standing task not found: {0}")]
    TaskNotFound(String),
    #[error("standing task is already leased until {expires_at_unix}: {task_id}")]
    AlreadyLeased {
        task_id: String,
        expires_at_unix: u64,
    },
    #[error("standing task is not pending: {0}")]
    NotPending(String),
    #[error("standing task lease does not match active lease: {0}")]
    WrongLease(String),
    #[error("standing task lease duration must be greater than zero")]
    InvalidLeaseDuration,
    #[error("standing task lease id space exhausted")]
    LeaseIdExhausted,
}

impl<T> StandingTaskQueue<T> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        id: impl Into<String>,
        task: T,
        now_unix: u64,
    ) -> Result<(), StandingQueueError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(StandingQueueError::EmptyTaskId);
        }
        if self.tasks.contains_key(&id) {
            return Err(StandingQueueError::DuplicateTask(id));
        }
        self.tasks.insert(
            id.clone(),
            StandingTaskRecord {
                id,
                task,
                state: StandingTaskState::Pending,
                attempts: 0,
                last_error: None,
                created_at_unix: now_unix,
                updated_at_unix: now_unix,
            },
        );
        Ok(())
    }

    #[must_use]
    pub fn get(&self, id: &str) -> Option<&StandingTaskRecord<T>> {
        self.tasks.get(id)
    }

    #[must_use]
    pub fn records(&self) -> impl Iterator<Item = &StandingTaskRecord<T>> {
        self.tasks.values()
    }

    pub fn lease(
        &mut self,
        id: &str,
        now_unix: u64,
        lease_duration_seconds: u64,
    ) -> Result<StandingTaskLease, StandingQueueError> {
        if lease_duration_seconds == 0 {
            return Err(StandingQueueError::InvalidLeaseDuration);
        }
        let record = self
            .tasks
            .get_mut(id)
            .ok_or_else(|| StandingQueueError::TaskNotFound(id.to_owned()))?;
        Self::reap_expired_record(record, now_unix);
        match record.state {
            StandingTaskState::Pending => {}
            StandingTaskState::Leased {
                expires_at_unix, ..
            } => {
                return Err(StandingQueueError::AlreadyLeased {
                    task_id: id.to_owned(),
                    expires_at_unix,
                });
            }
            StandingTaskState::Completed { .. } | StandingTaskState::Cancelled { .. } => {
                return Err(StandingQueueError::NotPending(id.to_owned()));
            }
        }

        let lease_id = self.next_lease_id;
        self.next_lease_id = self
            .next_lease_id
            .checked_add(1)
            .ok_or(StandingQueueError::LeaseIdExhausted)?;
        let expires_at_unix = now_unix.saturating_add(lease_duration_seconds);
        record.attempts = record.attempts.saturating_add(1);
        record.updated_at_unix = now_unix;
        record.state = StandingTaskState::Leased {
            lease_id,
            leased_at_unix: now_unix,
            expires_at_unix,
        };
        Ok(StandingTaskLease {
            task_id: id.to_owned(),
            lease_id,
            attempt: record.attempts,
            expires_at_unix,
        })
    }

    pub fn release(
        &mut self,
        lease: &StandingTaskLease,
        now_unix: u64,
        last_error: Option<String>,
    ) -> Result<(), StandingQueueError> {
        let record = self.active_lease_record(lease)?;
        record.state = StandingTaskState::Pending;
        record.last_error = last_error;
        record.updated_at_unix = now_unix;
        Ok(())
    }

    pub fn complete(
        &mut self,
        lease: &StandingTaskLease,
        now_unix: u64,
    ) -> Result<(), StandingQueueError> {
        let record = self.active_lease_record(lease)?;
        record.state = StandingTaskState::Completed {
            completed_at_unix: now_unix,
        };
        record.last_error = None;
        record.updated_at_unix = now_unix;
        Ok(())
    }

    pub fn cancel(&mut self, id: &str, now_unix: u64) -> Result<(), StandingQueueError> {
        let record = self
            .tasks
            .get_mut(id)
            .ok_or_else(|| StandingQueueError::TaskNotFound(id.to_owned()))?;
        record.state = StandingTaskState::Cancelled {
            cancelled_at_unix: now_unix,
        };
        record.updated_at_unix = now_unix;
        Ok(())
    }

    pub fn reap_expired(&mut self, now_unix: u64) -> usize {
        self.tasks
            .values_mut()
            .map(|record| usize::from(Self::reap_expired_record(record, now_unix)))
            .sum()
    }

    fn reap_expired_record(record: &mut StandingTaskRecord<T>, now_unix: u64) -> bool {
        let expired = matches!(
            record.state,
            StandingTaskState::Leased {
                expires_at_unix,
                ..
            } if expires_at_unix <= now_unix
        );
        if expired {
            record.state = StandingTaskState::Pending;
            record.updated_at_unix = now_unix;
        }
        expired
    }

    fn active_lease_record(
        &mut self,
        lease: &StandingTaskLease,
    ) -> Result<&mut StandingTaskRecord<T>, StandingQueueError> {
        let record = self
            .tasks
            .get_mut(&lease.task_id)
            .ok_or_else(|| StandingQueueError::TaskNotFound(lease.task_id.clone()))?;
        match record.state {
            StandingTaskState::Leased { lease_id, .. } if lease_id == lease.lease_id => Ok(record),
            StandingTaskState::Leased { .. }
            | StandingTaskState::Pending
            | StandingTaskState::Completed { .. }
            | StandingTaskState::Cancelled { .. } => {
                Err(StandingQueueError::WrongLease(lease.task_id.clone()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queue() -> StandingTaskQueue<String> {
        StandingTaskQueue::new()
    }

    #[test]
    fn active_lease_prevents_duplicate_claim() -> Result<(), StandingQueueError> {
        let mut queue = queue();
        queue.insert("task", "payload".to_owned(), 10)?;
        let first = queue.lease("task", 20, 60)?;
        assert_eq!(first.attempt, 1);
        assert!(matches!(
            queue.lease("task", 21, 60),
            Err(StandingQueueError::AlreadyLeased { .. })
        ));
        Ok(())
    }

    #[test]
    fn expired_lease_can_be_reclaimed_with_new_token() -> Result<(), StandingQueueError> {
        let mut queue = queue();
        queue.insert("task", "payload".to_owned(), 10)?;
        let first = queue.lease("task", 20, 5)?;
        let second = queue.lease("task", 25, 5)?;
        assert_ne!(first.lease_id, second.lease_id);
        assert_eq!(second.attempt, 2);
        Ok(())
    }

    #[test]
    fn stale_worker_cannot_complete_reclaimed_task() -> Result<(), StandingQueueError> {
        let mut queue = queue();
        queue.insert("task", "payload".to_owned(), 10)?;
        let first = queue.lease("task", 20, 5)?;
        let second = queue.lease("task", 25, 5)?;
        assert!(matches!(
            queue.complete(&first, 26),
            Err(StandingQueueError::WrongLease(_))
        ));
        queue.complete(&second, 26)?;
        assert!(matches!(
            queue.get("task").map(|record| &record.state),
            Some(StandingTaskState::Completed { .. })
        ));
        Ok(())
    }

    #[test]
    fn failed_attempt_returns_to_pending_with_diagnostic() -> Result<(), StandingQueueError> {
        let mut queue = queue();
        queue.insert("task", "payload".to_owned(), 10)?;
        let lease = queue.lease("task", 20, 60)?;
        queue.release(&lease, 21, Some("provider unavailable".to_owned()))?;
        let record = queue
            .get("task")
            .ok_or_else(|| StandingQueueError::TaskNotFound("task".to_owned()))?;
        assert_eq!(record.attempts, 1);
        assert_eq!(record.last_error.as_deref(), Some("provider unavailable"));
        assert_eq!(record.state, StandingTaskState::Pending);
        Ok(())
    }

    #[test]
    fn serialized_queue_preserves_lease_counter() -> Result<(), Box<dyn std::error::Error>> {
        let mut queue = queue();
        queue.insert("a", "one".to_owned(), 1)?;
        let first = queue.lease("a", 2, 1)?;
        assert_eq!(queue.reap_expired(3), 1);
        queue.insert("b", "two".to_owned(), 3)?;

        let json = serde_json::to_string(&queue)?;
        let mut restored: StandingTaskQueue<String> = serde_json::from_str(&json)?;
        let second = restored.lease("b", 4, 10)?;
        assert!(second.lease_id > first.lease_id);
        Ok(())
    }

    #[test]
    fn completed_and_cancelled_tasks_cannot_be_released_or_released_again()
    -> Result<(), StandingQueueError> {
        let mut queue = queue();
        queue.insert("done", "payload".to_owned(), 1)?;
        let lease = queue.lease("done", 2, 10)?;
        queue.complete(&lease, 3)?;
        assert!(matches!(
            queue.lease("done", 4, 10),
            Err(StandingQueueError::NotPending(_))
        ));

        queue.insert("cancelled", "payload".to_owned(), 1)?;
        queue.cancel("cancelled", 2)?;
        assert!(matches!(
            queue.lease("cancelled", 3, 10),
            Err(StandingQueueError::NotPending(_))
        ));
        Ok(())
    }
}
