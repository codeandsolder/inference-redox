use crate::{
    ModelCatalog, ModelRouteClass, Pricing, ProviderError, ProviderInventory, ReasoningSupport,
};
use std::collections::BTreeSet;

use super::openai_compatible::{
    OpenAiCompatibleProfile, OpenAiCompatibleProvider, ReasoningDialect, TemperatureZeroPolicy,
    TokenLimitField,
};

/// Mistral/Codestral inventory discovery without assuming an account billing mode.
#[derive(Debug, Clone)]
pub struct MistralInventory {
    general: OpenAiCompatibleProvider,
    legacy_codestral: OpenAiCompatibleProvider,
}

impl MistralInventory {
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        Self {
            general: OpenAiCompatibleProvider::new(
                api_key.clone(),
                profile("mistral", "https://api.mistral.ai/v1"),
            ),
            legacy_codestral: OpenAiCompatibleProvider::new(
                api_key,
                profile("codestral", "https://codestral.mistral.ai/v1"),
            ),
        }
    }

    /// Load the account-visible model inventory. Prefer the current Mistral API and fall back to
    /// the historical dedicated Codestral endpoint for old personal Codestral keys.
    ///
    /// Returned rows are observation-only (`route_class = None`) until the caller explicitly
    /// classifies the account's economic mode.
    ///
    /// # Errors
    /// Returns the dedicated-endpoint error when neither current nor legacy inventory succeeds.
    pub async fn inventory(
        &self,
        catalog: &ModelCatalog,
    ) -> Result<ProviderInventory, ProviderError> {
        let mut inventory = match self.general.inventory(catalog).await {
            Ok(inventory) => inventory,
            Err(_) => self.legacy_codestral.inventory(catalog).await?,
        };
        for model in &mut inventory.models {
            model.route_class = None;
            model.pricing = None;
        }
        Ok(inventory)
    }
}

fn profile(source: &str, base_url: &str) -> OpenAiCompatibleProfile {
    OpenAiCompatibleProfile {
        source: source.to_owned(),
        provider_key: source.to_owned(),
        provider_name: "Mistral AI".to_owned(),
        base_url: base_url.to_owned(),
        route_class: ModelRouteClass::Standard,
        token_limit_field: TokenLimitField::MaxTokens,
        reasoning_dialect: ReasoningDialect::Unsupported,
        temperature_zero: TemperatureZeroPolicy::PassThrough,
        adapter_parameters: BTreeSet::default(),
        nominal_parameters: BTreeSet::default(),
        reasoning_support: ReasoningSupport::None,
        supports_implicit_caching: false,
        pricing: Pricing::default(),
        usage_cost_usd: None,
    }
}
