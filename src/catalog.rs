use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::Pricing;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArchitecture {
    pub modality: Option<String>,
    pub input_modalities: BTreeSet<String>,
    pub output_modalities: BTreeSet<String>,
    pub tokenizer: Option<String>,
    pub instruct_type: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelTopProvider {
    pub context_length: Option<u64>,
    pub max_completion_tokens: Option<u64>,
    pub is_moderated: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelReasoning {
    pub mandatory: Option<bool>,
    pub default_enabled: Option<bool>,
    pub supported_efforts: BTreeSet<String>,
    pub default_effort: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ArtificialAnalysisBenchmarks {
    pub intelligence_index: Option<f64>,
    pub coding_index: Option<f64>,
    pub agentic_index: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DesignArenaBenchmark {
    pub arena: Option<String>,
    pub category: Option<String>,
    pub elo: Option<f64>,
    pub win_rate: Option<f64>,
    pub rank: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelBenchmarks {
    pub artificial_analysis: Option<ArtificialAnalysisBenchmarks>,
    pub design_arena: Vec<DesignArenaBenchmark>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCatalogEntry {
    /// Routable identifier in the source catalog. Catalog adapters may retain only route
    /// variants that matter cross-provider (currently base and `:free` for `OpenRouter`).
    pub id: String,
    /// Identity of the underlying model/version independent of a source-specific route variant.
    pub canonical_id: String,
    pub name: String,
    pub description: Option<String>,
    pub aliases: BTreeSet<String>,
    pub created_unix: Option<u64>,
    pub context_length: Option<u64>,
    pub architecture: ModelArchitecture,
    pub pricing: Pricing,
    pub top_provider: ModelTopProvider,
    /// Model-level nominal parameter claims from the metadata authority.
    pub supported_parameters: BTreeSet<String>,
    pub reasoning: ModelReasoning,
    pub benchmarks: ModelBenchmarks,
    pub knowledge_cutoff: Option<String>,
    pub expiration_date: Option<String>,
    /// Metadata authority that produced this entry, e.g. `openrouter`.
    pub source: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelCatalog {
    pub models: Vec<ModelCatalogEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelMatchKind {
    ExactAlias,
    NormalizedAlias,
    UniqueSlug,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedModel {
    pub canonical_id: String,
    pub preferred_id: String,
    pub variants: Vec<String>,
    pub matched_alias: String,
    pub match_kind: ModelMatchKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ModelLookup {
    Resolved { model: ResolvedModel },
    Ambiguous { canonical_ids: Vec<String> },
    NotFound,
}

impl ModelCatalog {
    #[must_use]
    pub fn resolve(&self, query: &str) -> ModelLookup {
        let query = query.trim();
        if query.is_empty() {
            return ModelLookup::NotFound;
        }

        let exact = self.matching_canonical_ids(|alias| alias.eq_ignore_ascii_case(query));
        if let Some(result) = self.finish_lookup(query, exact, ModelMatchKind::ExactAlias) {
            return result;
        }

        let normalized_query = normalize_model_name(query);
        let normalized =
            self.matching_canonical_ids(|alias| normalize_model_name(alias) == normalized_query);
        if let Some(result) = self.finish_lookup(query, normalized, ModelMatchKind::NormalizedAlias)
        {
            return result;
        }

        let slug = normalize_model_slug(query);
        let slug_matches = self.matching_canonical_ids(|alias| normalize_model_slug(alias) == slug);
        self.finish_lookup(query, slug_matches, ModelMatchKind::UniqueSlug)
            .unwrap_or(ModelLookup::NotFound)
    }

    #[must_use]
    pub fn canonical_models(&self) -> Vec<ResolvedModel> {
        let mut canonical_ids = BTreeSet::new();
        for model in &self.models {
            canonical_ids.insert(model.canonical_id.clone());
        }
        canonical_ids
            .into_iter()
            .filter_map(|canonical_id| {
                self.resolved(&canonical_id, &canonical_id, ModelMatchKind::ExactAlias)
            })
            .collect()
    }

    fn matching_canonical_ids(&self, predicate: impl Fn(&str) -> bool) -> BTreeMap<String, String> {
        let mut matches = BTreeMap::new();
        for model in &self.models {
            for alias in &model.aliases {
                if predicate(alias) {
                    matches
                        .entry(model.canonical_id.clone())
                        .or_insert_with(|| alias.clone());
                }
            }
        }
        matches
    }

    fn finish_lookup(
        &self,
        query: &str,
        matches: BTreeMap<String, String>,
        match_kind: ModelMatchKind,
    ) -> Option<ModelLookup> {
        if matches.is_empty() {
            return None;
        }
        if matches.len() > 1 {
            return Some(ModelLookup::Ambiguous {
                canonical_ids: matches.into_keys().collect(),
            });
        }
        let (canonical_id, matched_alias) = matches.into_iter().next()?;
        self.resolved(&canonical_id, &matched_alias, match_kind)
            .map(|model| ModelLookup::Resolved { model })
            .or_else(|| {
                Some(ModelLookup::Ambiguous {
                    canonical_ids: vec![query.to_owned()],
                })
            })
    }

    fn resolved(
        &self,
        canonical_id: &str,
        matched_alias: &str,
        match_kind: ModelMatchKind,
    ) -> Option<ResolvedModel> {
        let mut variants: Vec<String> = self
            .models
            .iter()
            .filter(|model| model.canonical_id == canonical_id)
            .map(|model| model.id.clone())
            .collect();
        variants.sort();
        variants.dedup();
        let preferred_id = variants
            .iter()
            .min_by_key(|id| (id.contains(':'), id.len(), id.as_str()))?
            .clone();
        Some(ResolvedModel {
            canonical_id: canonical_id.to_owned(),
            preferred_id,
            variants,
            matched_alias: matched_alias.to_owned(),
            match_kind,
        })
    }
}

#[must_use]
pub fn normalize_model_name(value: &str) -> String {
    value
        .chars()
        .flat_map(char::to_lowercase)
        .filter(char::is_ascii_alphanumeric)
        .collect()
}

#[must_use]
pub fn normalize_model_slug(value: &str) -> String {
    let tail = value.rsplit('/').next().unwrap_or(value);
    let base = tail.split(':').next().unwrap_or(tail);
    normalize_model_name(base)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, canonical_id: &str, name: &str) -> ModelCatalogEntry {
        ModelCatalogEntry {
            id: id.to_owned(),
            canonical_id: canonical_id.to_owned(),
            name: name.to_owned(),
            description: None,
            aliases: [id, canonical_id, name]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            created_unix: None,
            context_length: None,
            architecture: ModelArchitecture::default(),
            pricing: Pricing::default(),
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
    fn canonical_slug_groups_base_and_free_variants() -> Result<(), &'static str> {
        let catalog = ModelCatalog {
            models: vec![
                entry(
                    "openai/gpt-6-luna",
                    "openai/gpt-6-luna-20260922",
                    "OpenAI: GPT-6 Luna",
                ),
                entry(
                    "openai/gpt-6-luna:free",
                    "openai/gpt-6-luna-20260922",
                    "OpenAI: GPT-6 Luna (free)",
                ),
            ],
        };
        let ModelLookup::Resolved { model } = catalog.resolve("openai/gpt-6-luna:free") else {
            return Err("expected route variant to resolve");
        };
        assert_eq!(model.canonical_id, "openai/gpt-6-luna-20260922");
        assert_eq!(model.preferred_id, "openai/gpt-6-luna");
        assert_eq!(model.variants.len(), 2);
        Ok(())
    }

    #[test]
    fn unique_slug_bridges_provider_owner_names() -> Result<(), &'static str> {
        let catalog = ModelCatalog {
            models: vec![entry(
                "stepfun/step-5-preview",
                "stepfun/step-5-preview-20260901",
                "Step 5 Preview",
            )],
        };
        let ModelLookup::Resolved { model } = catalog.resolve("stepfun-ai/step-5-preview") else {
            return Err("expected provider-native owner alias to resolve");
        };
        assert_eq!(model.match_kind, ModelMatchKind::UniqueSlug);
        assert_eq!(model.preferred_id, "stepfun/step-5-preview");
        Ok(())
    }

    #[test]
    fn ambiguous_slug_is_not_guessed() {
        let catalog = ModelCatalog {
            models: vec![
                entry("vendor-a/foo", "vendor-a/foo-1", "Foo A"),
                entry("vendor-b/foo", "vendor-b/foo-1", "Foo B"),
            ],
        };
        assert!(matches!(
            catalog.resolve("other/foo"),
            ModelLookup::Ambiguous { .. }
        ));
    }
}
