use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::InferenceRequest;

/// One independent request inside an asynchronous provider batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchInferenceItem {
    pub custom_id: String,
    pub request: InferenceRequest,
}

/// Normalized `OpenRouter` batch state. `raw` retains fields we do not normalize yet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchJob {
    pub id: String,
    pub status: String,
    pub results: Option<Vec<Value>>,
    pub raw: Value,
}

impl BatchJob {
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status.as_str(),
            "completed" | "failed" | "cancelled" | "expired"
        )
    }

    #[must_use]
    pub fn is_success(&self) -> bool {
        self.status == "completed"
    }
}
