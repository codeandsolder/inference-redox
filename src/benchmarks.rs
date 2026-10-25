use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BenchmarkSource {
    ArtificialAnalysis,
    DesignArena,
    Openrouter,
}

impl BenchmarkSource {
    #[must_use]
    pub const fn as_query_str(self) -> &'static str {
        match self {
            Self::ArtificialAnalysis => "artificial-analysis",
            Self::DesignArena => "design-arena",
            Self::Openrouter => "openrouter",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkTaskType {
    Coding,
    Intelligence,
    Agentic,
}

impl BenchmarkTaskType {
    #[must_use]
    pub const fn as_query_str(self) -> &'static str {
        match self {
            Self::Coding => "coding",
            Self::Intelligence => "intelligence",
            Self::Agentic => "agentic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkUnit {
    Index,
    Elo,
    Percent,
    Fraction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkObservation {
    /// `OpenRouter` permanent model slug; this is the canonical-model join key.
    pub canonical_id: String,
    pub source: BenchmarkSource,
    pub task_type: BenchmarkTaskType,
    /// Stable metric name from the normalizer, e.g. `coding_index` or a concrete `OpenRouter` eval.
    pub metric: String,
    pub score: f64,
    pub unit: BenchmarkUnit,
    /// Human-readable evaluated configuration. This may distinguish reasoning modes that share
    /// one canonical model identity.
    pub configuration: Option<String>,
    #[serde(default)]
    pub qualifiers: BTreeMap<String, String>,
    pub sample_count: Option<u64>,
    pub standard_deviation: Option<f64>,
    pub average_cost_usd: Option<f64>,
    pub average_generation_time_ms: Option<u64>,
    pub observed_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkSnapshotMeta {
    pub source: BenchmarkSource,
    pub task_type: BenchmarkTaskType,
    pub as_of: Option<String>,
    pub citation: Option<String>,
    pub source_url: Option<String>,
    pub version: Option<String>,
    pub model_count: Option<u64>,
    /// True only when a refresh failed after the cache TTL and this snapshot is the retained fallback.
    #[serde(default)]
    pub stale: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkSnapshot {
    pub meta: BenchmarkSnapshotMeta,
    pub observations: Vec<BenchmarkObservation>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkCatalog {
    pub snapshots: Vec<BenchmarkSnapshotMeta>,
    pub observations: Vec<BenchmarkObservation>,
}

impl BenchmarkCatalog {
    /// Replace the prior snapshot for the same source/task pair.
    pub fn ingest(&mut self, snapshot: BenchmarkSnapshot) {
        let source = snapshot.meta.source;
        let task_type = snapshot.meta.task_type;
        self.snapshots
            .retain(|meta| meta.source != source || meta.task_type != task_type);
        self.observations.retain(|observation| {
            observation.source != source || observation.task_type != task_type
        });
        self.snapshots.push(snapshot.meta);
        self.observations.extend(snapshot.observations);
    }

    #[must_use]
    pub fn for_model(&self, canonical_id: &str) -> Vec<&BenchmarkObservation> {
        self.observations
            .iter()
            .filter(|observation| observation.canonical_id == canonical_id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(score: f64) -> BenchmarkSnapshot {
        BenchmarkSnapshot {
            meta: BenchmarkSnapshotMeta {
                source: BenchmarkSource::ArtificialAnalysis,
                task_type: BenchmarkTaskType::Coding,
                as_of: Some("now".to_owned()),
                citation: None,
                source_url: None,
                version: Some("v1".to_owned()),
                model_count: Some(1),
                stale: false,
            },
            observations: vec![BenchmarkObservation {
                canonical_id: "vendor/model-v1".to_owned(),
                source: BenchmarkSource::ArtificialAnalysis,
                task_type: BenchmarkTaskType::Coding,
                metric: "coding_index".to_owned(),
                score,
                unit: BenchmarkUnit::Index,
                configuration: Some("Model".to_owned()),
                qualifiers: BTreeMap::new(),
                sample_count: None,
                standard_deviation: None,
                average_cost_usd: None,
                average_generation_time_ms: None,
                observed_at: Some("now".to_owned()),
            }],
        }
    }

    #[test]
    fn newer_snapshot_replaces_same_source_and_task() {
        let mut catalog = BenchmarkCatalog::default();
        catalog.ingest(snapshot(10.0));
        catalog.ingest(snapshot(20.0));
        assert_eq!(catalog.snapshots.len(), 1);
        assert_eq!(catalog.observations.len(), 1);
        assert_eq!(catalog.observations[0].score, 20.0);
    }
}
