use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{ModelCatalog, ModelLookup, ModelMatchKind, ModelRouteClass, Pricing};

/// Provider-native model metadata observed from a provider's own inventory API.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NativeModelRecord {
    pub id: String,
    pub display_name: Option<String>,
    pub owned_by: Option<String>,
    pub created_unix: Option<u64>,
    pub context_length: Option<u64>,
    pub max_completion_tokens: Option<u64>,
    pub claimed_parameters: BTreeSet<String>,
    pub pricing: Option<Pricing>,
    pub route_class: Option<ModelRouteClass>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum InventoryResolution {
    Resolved {
        canonical_id: String,
        match_kind: ModelMatchKind,
    },
    Ambiguous {
        canonical_ids: Vec<String>,
    },
    Unmatched,
}

/// One provider/model observation after reconciling the provider-native ID against the canonical catalog.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderModelObservation {
    pub provider: String,
    pub native_id: String,
    pub display_name: Option<String>,
    pub owned_by: Option<String>,
    pub created_unix: Option<u64>,
    pub context_length: Option<u64>,
    pub max_completion_tokens: Option<u64>,
    pub claimed_parameters: BTreeSet<String>,
    pub pricing: Option<Pricing>,
    pub route_class: Option<ModelRouteClass>,
    pub resolution: InventoryResolution,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderInventory {
    pub provider: String,
    pub models: Vec<ProviderModelObservation>,
}

impl ProviderInventory {
    #[must_use]
    pub fn reconcile(
        provider: impl Into<String>,
        native_models: Vec<NativeModelRecord>,
        catalog: &ModelCatalog,
    ) -> Self {
        let provider = provider.into();
        let models = native_models
            .into_iter()
            .map(|native| {
                let resolution = match catalog.resolve(&native.id) {
                    ModelLookup::Resolved { model } => InventoryResolution::Resolved {
                        canonical_id: model.canonical_id,
                        match_kind: model.match_kind,
                    },
                    ModelLookup::Ambiguous { canonical_ids } => {
                        InventoryResolution::Ambiguous { canonical_ids }
                    }
                    ModelLookup::NotFound => InventoryResolution::Unmatched,
                };
                ProviderModelObservation {
                    provider: provider.clone(),
                    native_id: native.id,
                    display_name: native.display_name,
                    owned_by: native.owned_by,
                    created_unix: native.created_unix,
                    context_length: native.context_length,
                    max_completion_tokens: native.max_completion_tokens,
                    claimed_parameters: native.claimed_parameters,
                    pricing: native.pricing,
                    route_class: native.route_class,
                    resolution,
                }
            })
            .collect();
        Self { provider, models }
    }

    #[must_use]
    pub fn resolved_count(&self) -> usize {
        self.models
            .iter()
            .filter(|model| matches!(model.resolution, InventoryResolution::Resolved { .. }))
            .count()
    }

    #[must_use]
    pub fn unmatched_count(&self) -> usize {
        self.models
            .iter()
            .filter(|model| matches!(model.resolution, InventoryResolution::Unmatched))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ModelArchitecture, ModelBenchmarks, ModelCatalogEntry, ModelReasoning, ModelTopProvider,
        Pricing,
    };

    fn catalog_entry(id: &str, canonical_id: &str, aliases: &[&str]) -> ModelCatalogEntry {
        ModelCatalogEntry {
            id: id.to_owned(),
            route_class: ModelRouteClass::Standard,
            canonical_id: canonical_id.to_owned(),
            name: id.to_owned(),
            description: None,
            aliases: aliases.iter().map(|alias| (*alias).to_owned()).collect(),
            created_unix: None,
            context_length: None,
            architecture: ModelArchitecture::default(),
            pricing: Pricing::default(),
            pricing_known: true,
            top_provider: ModelTopProvider::default(),
            supported_parameters: BTreeSet::new(),
            reasoning: ModelReasoning::default(),
            benchmarks: ModelBenchmarks::default(),
            knowledge_cutoff: None,
            expiration_date: None,
            source: "test".to_owned(),
        }
    }

    #[test]
    fn reconciles_native_aliases_without_rewriting_provider_ids() {
        let catalog = ModelCatalog {
            models: vec![catalog_entry(
                "stepfun/step-5-preview",
                "stepfun/step-5-preview-20261008",
                &["stepfun/step-5-preview", "stepfun-ai/Step-5-Preview"],
            )],
        };
        let inventory = ProviderInventory::reconcile(
            "nvidia",
            vec![NativeModelRecord {
                id: "stepfun-ai/step-5-preview".to_owned(),
                route_class: Some(ModelRouteClass::Free),
                ..NativeModelRecord::default()
            }],
            &catalog,
        );
        assert_eq!(inventory.resolved_count(), 1);
        assert_eq!(inventory.models[0].native_id, "stepfun-ai/step-5-preview");
        assert_eq!(inventory.models[0].route_class, Some(ModelRouteClass::Free));
        assert!(matches!(
            &inventory.models[0].resolution,
            InventoryResolution::Resolved { canonical_id, .. }
                if canonical_id == "stepfun/step-5-preview-20261008"
        ));
    }

    #[test]
    fn preserves_unmatched_native_models() {
        let inventory = ProviderInventory::reconcile(
            "nvidia",
            vec![NativeModelRecord {
                id: "vendor/mystery-model".to_owned(),
                ..NativeModelRecord::default()
            }],
            &ModelCatalog::default(),
        );
        assert_eq!(inventory.unmatched_count(), 1);
        assert!(matches!(
            inventory.models[0].resolution,
            InventoryResolution::Unmatched
        ));
    }
}
