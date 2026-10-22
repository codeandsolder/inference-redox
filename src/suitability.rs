use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{ModelCatalog, ModelCatalogEntry, ModelRouteClass};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuitabilityStatus {
    Suitable,
    Unknown,
    Unsuitable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkKind {
    Intelligence,
    Coding,
    Agentic,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelSuitabilityRequirements {
    pub minimum_intelligence_index: Option<f64>,
    pub minimum_coding_index: Option<f64>,
    pub minimum_agentic_index: Option<f64>,
    #[serde(default)]
    pub required_input_modalities: BTreeSet<String>,
    #[serde(default)]
    pub required_output_modalities: BTreeSet<String>,
    /// Canonical/model-level metadata only. Provider-route conformance remains authoritative.
    #[serde(default)]
    pub required_model_parameters: BTreeSet<String>,
    #[serde(default)]
    pub require_reasoning: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SuitabilityCheck {
    Benchmark {
        benchmark: BenchmarkKind,
        minimum: f64,
        observed: Option<f64>,
        status: SuitabilityStatus,
    },
    InputModality {
        modality: String,
        status: SuitabilityStatus,
    },
    OutputModality {
        modality: String,
        status: SuitabilityStatus,
    },
    ModelParameter {
        parameter: String,
        status: SuitabilityStatus,
    },
    Reasoning {
        status: SuitabilityStatus,
    },
}

impl SuitabilityCheck {
    fn status(&self) -> SuitabilityStatus {
        match self {
            Self::Benchmark { status, .. }
            | Self::InputModality { status, .. }
            | Self::OutputModality { status, .. }
            | Self::ModelParameter { status, .. }
            | Self::Reasoning { status } => *status,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSuitabilityReport {
    pub canonical_id: String,
    pub metadata_source: String,
    pub metadata_route_id: String,
    pub status: SuitabilityStatus,
    pub checks: Vec<SuitabilityCheck>,
}

impl ModelSuitabilityRequirements {
    #[must_use]
    pub fn evaluate(&self, entry: &ModelCatalogEntry) -> ModelSuitabilityReport {
        let mut checks = self.benchmark_checks(entry);
        checks.extend(self.modality_checks(entry));
        checks.extend(self.parameter_checks(entry));
        if self.require_reasoning {
            checks.push(SuitabilityCheck::Reasoning {
                status: reasoning_status(entry),
            });
        }
        ModelSuitabilityReport {
            canonical_id: entry.canonical_id.clone(),
            metadata_source: entry.source.clone(),
            metadata_route_id: entry.id.clone(),
            status: aggregate_status(&checks),
            checks,
        }
    }

    fn benchmark_checks(&self, entry: &ModelCatalogEntry) -> Vec<SuitabilityCheck> {
        let aa = entry.benchmarks.artificial_analysis.as_ref();
        [
            (
                BenchmarkKind::Intelligence,
                self.minimum_intelligence_index,
                aa.and_then(|value| value.intelligence_index),
            ),
            (
                BenchmarkKind::Coding,
                self.minimum_coding_index,
                aa.and_then(|value| value.coding_index),
            ),
            (
                BenchmarkKind::Agentic,
                self.minimum_agentic_index,
                aa.and_then(|value| value.agentic_index),
            ),
        ]
        .into_iter()
        .filter_map(|(benchmark, minimum, observed)| {
            minimum.map(|minimum| SuitabilityCheck::Benchmark {
                benchmark,
                minimum,
                observed,
                status: threshold_status(observed, minimum),
            })
        })
        .collect()
    }

    fn modality_checks(&self, entry: &ModelCatalogEntry) -> Vec<SuitabilityCheck> {
        self.required_input_modalities
            .iter()
            .map(|modality| SuitabilityCheck::InputModality {
                modality: modality.clone(),
                status: set_membership_status(&entry.architecture.input_modalities, modality),
            })
            .chain(self.required_output_modalities.iter().map(|modality| {
                SuitabilityCheck::OutputModality {
                    modality: modality.clone(),
                    status: set_membership_status(&entry.architecture.output_modalities, modality),
                }
            }))
            .collect()
    }

    fn parameter_checks(&self, entry: &ModelCatalogEntry) -> Vec<SuitabilityCheck> {
        self.required_model_parameters
            .iter()
            .map(|parameter| SuitabilityCheck::ModelParameter {
                parameter: parameter.clone(),
                status: set_membership_status(&entry.supported_parameters, parameter),
            })
            .collect()
    }
}

impl ModelCatalog {
    /// Evaluate one metadata representative per canonical model.
    #[must_use]
    pub fn suitability_reports(
        &self,
        requirements: &ModelSuitabilityRequirements,
    ) -> Vec<ModelSuitabilityReport> {
        self.canonical_models()
            .into_iter()
            .filter_map(|model| {
                self.metadata_entry(&model.canonical_id)
                    .map(|entry| requirements.evaluate(entry))
            })
            .collect()
    }

    fn metadata_entry(&self, canonical_id: &str) -> Option<&ModelCatalogEntry> {
        self.models
            .iter()
            .filter(|entry| entry.canonical_id == canonical_id)
            .min_by_key(|entry| {
                let source_rank = u8::from(entry.source != "openrouter");
                let route_rank = match entry.route_class {
                    ModelRouteClass::Standard => 0_u8,
                    ModelRouteClass::Free => 1,
                    ModelRouteClass::Batch => 2,
                };
                (source_rank, route_rank, entry.id.len(), entry.id.as_str())
            })
    }
}

fn threshold_status(observed: Option<f64>, minimum: f64) -> SuitabilityStatus {
    observed.map_or(SuitabilityStatus::Unknown, |value| {
        if value >= minimum {
            SuitabilityStatus::Suitable
        } else {
            SuitabilityStatus::Unsuitable
        }
    })
}

fn set_membership_status(values: &BTreeSet<String>, required: &str) -> SuitabilityStatus {
    if values.is_empty() {
        SuitabilityStatus::Unknown
    } else if values.contains(required) {
        SuitabilityStatus::Suitable
    } else {
        SuitabilityStatus::Unsuitable
    }
}

fn reasoning_status(entry: &ModelCatalogEntry) -> SuitabilityStatus {
    let positive = !entry.reasoning.supported_efforts.is_empty()
        || entry.reasoning.mandatory == Some(true)
        || entry.reasoning.default_enabled == Some(true)
        || entry.supported_parameters.contains("reasoning")
        || entry.supported_parameters.contains("reasoning_effort");
    let has_metadata = entry.reasoning.mandatory.is_some()
        || entry.reasoning.default_enabled.is_some()
        || !entry.reasoning.supported_efforts.is_empty()
        || entry.reasoning.default_effort.is_some()
        || !entry.supported_parameters.is_empty();
    if positive {
        SuitabilityStatus::Suitable
    } else if has_metadata {
        SuitabilityStatus::Unsuitable
    } else {
        SuitabilityStatus::Unknown
    }
}

fn aggregate_status(checks: &[SuitabilityCheck]) -> SuitabilityStatus {
    if checks
        .iter()
        .any(|check| check.status() == SuitabilityStatus::Unsuitable)
    {
        SuitabilityStatus::Unsuitable
    } else if checks
        .iter()
        .any(|check| check.status() == SuitabilityStatus::Unknown)
    {
        SuitabilityStatus::Unknown
    } else {
        SuitabilityStatus::Suitable
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ArtificialAnalysisBenchmarks, ModelArchitecture, ModelBenchmarks, ModelReasoning,
        ModelTopProvider, Pricing,
    };

    fn entry(source: &str, id: &str, class: ModelRouteClass) -> ModelCatalogEntry {
        ModelCatalogEntry {
            id: id.to_owned(),
            route_class: class,
            canonical_id: "vendor/model-v1".to_owned(),
            name: "Model".to_owned(),
            description: None,
            aliases: BTreeSet::from([id.to_owned()]),
            created_unix: None,
            context_length: Some(100_000),
            architecture: ModelArchitecture::default(),
            pricing: Pricing::default(),
            pricing_known: true,
            top_provider: ModelTopProvider::default(),
            supported_parameters: BTreeSet::new(),
            reasoning: ModelReasoning::default(),
            benchmarks: ModelBenchmarks::default(),
            knowledge_cutoff: None,
            expiration_date: None,
            source: source.to_owned(),
        }
    }

    #[test]
    fn benchmark_thresholds_are_three_state() {
        let mut model = entry("openrouter", "vendor/model", ModelRouteClass::Standard);
        model.benchmarks.artificial_analysis = Some(ArtificialAnalysisBenchmarks {
            intelligence_index: None,
            coding_index: Some(60.0),
            agentic_index: Some(10.0),
        });
        let requirements = ModelSuitabilityRequirements {
            minimum_intelligence_index: Some(20.0),
            minimum_coding_index: Some(50.0),
            minimum_agentic_index: Some(20.0),
            ..ModelSuitabilityRequirements::default()
        };
        let report = requirements.evaluate(&model);
        assert_eq!(report.status, SuitabilityStatus::Unsuitable);
        assert!(report.checks.iter().any(|check| matches!(
            check,
            SuitabilityCheck::Benchmark {
                benchmark: BenchmarkKind::Intelligence,
                status: SuitabilityStatus::Unknown,
                ..
            }
        )));
    }

    #[test]
    fn missing_metadata_is_unknown_not_unsupported() {
        let model = entry("nvidia", "vendor/native", ModelRouteClass::Free);
        let requirements = ModelSuitabilityRequirements {
            minimum_coding_index: Some(1.0),
            required_input_modalities: BTreeSet::from(["image".to_owned()]),
            require_reasoning: true,
            ..ModelSuitabilityRequirements::default()
        };
        assert_eq!(
            requirements.evaluate(&model).status,
            SuitabilityStatus::Unknown
        );
    }

    #[test]
    fn explicit_modalities_and_parameters_are_enforced() {
        let mut model = entry("openrouter", "vendor/model", ModelRouteClass::Standard);
        model.architecture.input_modalities =
            BTreeSet::from(["text".to_owned(), "image".to_owned()]);
        model.architecture.output_modalities = BTreeSet::from(["text".to_owned()]);
        model.supported_parameters = BTreeSet::from(["reasoning".to_owned(), "tools".to_owned()]);
        let requirements = ModelSuitabilityRequirements {
            required_input_modalities: BTreeSet::from(["image".to_owned()]),
            required_output_modalities: BTreeSet::from(["text".to_owned()]),
            required_model_parameters: BTreeSet::from(["tools".to_owned()]),
            require_reasoning: true,
            ..ModelSuitabilityRequirements::default()
        };
        assert_eq!(
            requirements.evaluate(&model).status,
            SuitabilityStatus::Suitable
        );
    }

    #[test]
    fn canonical_report_prefers_openrouter_standard_metadata() {
        let native = entry("nvidia", "vendor/native", ModelRouteClass::Free);
        let mut batch = entry("openrouter", "vendor/model:batch", ModelRouteClass::Batch);
        batch.benchmarks.artificial_analysis = Some(ArtificialAnalysisBenchmarks {
            coding_index: Some(70.0),
            ..ArtificialAnalysisBenchmarks::default()
        });
        let mut standard = entry("openrouter", "vendor/model", ModelRouteClass::Standard);
        standard.benchmarks.artificial_analysis = Some(ArtificialAnalysisBenchmarks {
            coding_index: Some(80.0),
            ..ArtificialAnalysisBenchmarks::default()
        });
        let catalog = ModelCatalog {
            models: vec![native, batch, standard],
        };
        let reports = catalog.suitability_reports(&ModelSuitabilityRequirements {
            minimum_coding_index: Some(75.0),
            ..ModelSuitabilityRequirements::default()
        });
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].metadata_source, "openrouter");
        assert_eq!(reports[0].metadata_route_id, "vendor/model");
        assert_eq!(reports[0].status, SuitabilityStatus::Suitable);
    }
}
