//! Short-lived, endpoint-local evidence of reusable prompt-prefix caches.
//!
//! Affinity is an economic/performance observation, not an endpoint capability.
//! Callers supply an opaque fingerprint of the *cacheable prefix*, never the text.
use serde::{Deserialize, Serialize};

use crate::{Endpoint, InferenceRequest, InferenceResponse, QuotaRouteKey, RouteCandidate};

/// Default lifetime for an observed implicit prompt-cache hit.
pub const DEFAULT_CACHE_AFFINITY_TTL_SECONDS: u64 = 300;
const MAX_OBSERVATIONS: usize = 4096;

/// One observed cache hit, scoped to an exact execution route and prefix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheAffinityObservation {
    pub route: QuotaRouteKey,
    /// Exact provider endpoint, not just the provider/model quota identity.
    pub endpoint_id: String,
    pub prefix_fingerprint: String,
    pub session_id: Option<String>,
    pub cached_prompt_tokens: u64,
    pub observed_at_unix: u64,
    pub ttl_seconds: u64,
    /// Confidence in reusing this observed cache hit, between zero and one.
    pub confidence: f64,
    /// Optional measured latency improvement for this prefix/route, in seconds.
    pub latency_savings_seconds: Option<f64>,
}

/// Bounded, expiring routing evidence; independent from provider dialect/conformance.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CacheAffinityProfile {
    #[serde(default)]
    pub observations: Vec<CacheAffinityObservation>,
}

impl CacheAffinityProfile {
    /// Record exact-route cache evidence. Invalid or already-expired observations are ignored.
    pub fn observe(&mut self, observation: CacheAffinityObservation, now_unix: u64) {
        self.prune(now_unix);
        if observation.prefix_fingerprint.is_empty()
            || observation.endpoint_id.is_empty()
            || observation.prefix_fingerprint.len() > 256
            || observation.cached_prompt_tokens == 0
            || observation.ttl_seconds == 0
            || observation.observed_at_unix > now_unix
            || now_unix.saturating_sub(observation.observed_at_unix) >= observation.ttl_seconds
            || !observation.confidence.is_finite()
            || !(0.0..=1.0).contains(&observation.confidence)
            || observation
                .latency_savings_seconds
                .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return;
        }
        self.remove_exact(
            &observation.route,
            &observation.endpoint_id,
            &observation.prefix_fingerprint,
            observation.session_id.as_deref(),
        );
        self.observations.push(observation);
        if self.observations.len() > MAX_OBSERVATIONS {
            let oldest = self
                .observations
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.observed_at_unix)
                .map(|(index, _)| index);
            if let Some(index) = oldest {
                self.observations.swap_remove(index);
            }
        }
    }

    /// Learn from explicit provider-reported cached token usage, not from catalog claims.
    ///
    /// An explicit zero hit evicts previous evidence for this exact route and prefix.
    pub fn observe_response(
        &mut self,
        request: &InferenceRequest,
        endpoint: &Endpoint,
        response: &InferenceResponse,
        now_unix: u64,
    ) {
        let Some(prefix) = request.cache_prefix_fingerprint.as_deref() else {
            return;
        };
        let Some(tokens) = observed_cached_prompt_tokens(&response.raw) else {
            return;
        };
        let route = QuotaRouteKey::from_endpoint(endpoint);
        if tokens == 0 {
            self.remove_exact(&route, &endpoint.id, prefix, request.session_id.as_deref());
            return;
        }
        self.observe(
            CacheAffinityObservation {
                route,
                endpoint_id: endpoint.id.clone(),
                prefix_fingerprint: prefix.to_owned(),
                session_id: request.session_id.clone(),
                cached_prompt_tokens: tokens,
                observed_at_unix: now_unix,
                ttl_seconds: DEFAULT_CACHE_AFFINITY_TTL_SECONDS,
                confidence: 0.85,
                latency_savings_seconds: None,
            },
            now_unix,
        );
    }

    /// Remove expired, future-dated or otherwise unusable entries.
    pub fn prune(&mut self, now_unix: u64) {
        self.observations.retain(|entry| {
            entry.observed_at_unix <= now_unix
                && now_unix - entry.observed_at_unix < entry.ttl_seconds
                && entry.ttl_seconds > 0
                && entry.confidence.is_finite()
                && (0.0..=1.0).contains(&entry.confidence)
                && !entry.prefix_fingerprint.is_empty()
                && !entry.endpoint_id.is_empty()
                && entry.cached_prompt_tokens > 0
                && entry
                    .latency_savings_seconds
                    .is_none_or(|value| value.is_finite() && value >= 0.0)
        });
    }

    fn remove_exact(
        &mut self,
        route: &QuotaRouteKey,
        endpoint_id: &str,
        prefix: &str,
        session: Option<&str>,
    ) {
        self.observations.retain(|entry| {
            &entry.route != route
                || entry.endpoint_id != endpoint_id
                || entry.prefix_fingerprint != prefix
                || entry.session_id.as_deref() != session
        });
    }

    /// Adjust cost and latency estimates from observed, time-decayed cache reuse.
    ///
    /// The caller's uncached admission and hard-cost checks remain authoritative.
    pub fn adjust_candidate(
        &self,
        request: &InferenceRequest,
        prompt_tokens: u64,
        candidate: &mut RouteCandidate,
        now_unix: u64,
    ) {
        let Some(prefix) = request.cache_prefix_fingerprint.as_deref() else {
            return;
        };
        let route = QuotaRouteKey::from_endpoint(&candidate.endpoint);
        let Some(observation) = self.observations.iter().rev().find(|entry| {
            entry.route == route
                && entry.endpoint_id == candidate.endpoint.id
                && entry.prefix_fingerprint == prefix
                && entry.session_id == request.session_id
                && entry.observed_at_unix <= now_unix
                && now_unix - entry.observed_at_unix < entry.ttl_seconds
                && entry.ttl_seconds > 0
                && entry.confidence.is_finite()
                && (0.0..=1.0).contains(&entry.confidence)
                && entry.cached_prompt_tokens > 0
                && entry
                    .latency_savings_seconds
                    .is_none_or(|value| value.is_finite() && value >= 0.0)
        }) else {
            return;
        };

        #[expect(
            clippy::cast_precision_loss,
            reason = "scoring projections, not exact token accounting"
        )]
        let survival = (1.0
            - (now_unix - observation.observed_at_unix) as f64 / observation.ttl_seconds as f64)
            * observation.confidence;
        if survival <= 0.0 {
            return;
        }
        let rates = candidate.endpoint.pricing.effective(prompt_tokens);
        if let Some(cache_read_rate) = rates.cache_read_per_token
            && cache_read_rate.is_finite()
            && cache_read_rate >= 0.0
            && rates.prompt_per_token.is_finite()
            && rates.prompt_per_token > cache_read_rate
        {
            #[expect(
                clippy::cast_precision_loss,
                reason = "expected USD saving is approximate"
            )]
            let expected_reused =
                observation.cached_prompt_tokens.min(prompt_tokens) as f64 * survival;
            let savings = expected_reused
                * (rates.prompt_per_token - cache_read_rate)
                * (1.0 - candidate.endpoint.pricing.discount.clamp(0.0, 1.0));
            if savings.is_finite() && savings >= 0.0 {
                candidate.expected_cost_usd = (candidate.expected_cost_usd - savings).max(0.0);
            }
        }
        if let Some(saved) = observation.latency_savings_seconds {
            if let Some(seconds) = &mut candidate.expected_response_p75_seconds {
                *seconds = (*seconds - saved * survival).max(0.0);
            }
            if let Some(seconds) = &mut candidate.expected_response_p95_seconds {
                *seconds = (*seconds - saved * survival).max(0.0);
            }
        }
    }
}

fn observed_cached_prompt_tokens(raw: &serde_json::Value) -> Option<u64> {
    [
        "/usage/prompt_tokens_details/cached_tokens",
        "/usage/input_tokens_details/cached_tokens",
        "/usage/cache_read_input_tokens",
        "/usageMetadata/cachedContentTokenCount",
        "/usage_metadata/cache_read_input_tokens",
    ]
    .into_iter()
    .find_map(|path| raw.pointer(path).and_then(serde_json::Value::as_u64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn recognizes_native_cache_telemetry_including_zero() {
        assert_eq!(
            observed_cached_prompt_tokens(
                &json!({"usage": {"prompt_tokens_details": {"cached_tokens": 240}}})
            ),
            Some(240)
        );
        assert_eq!(
            observed_cached_prompt_tokens(&json!({"usage": {"cache_read_input_tokens": 0}})),
            Some(0)
        );
        assert_eq!(
            observed_cached_prompt_tokens(
                &json!({"usageMetadata": {"cachedContentTokenCount": 120}})
            ),
            Some(120)
        );
        assert_eq!(observed_cached_prompt_tokens(&json!({})), None);
    }
}

#[cfg(test)]
mod routing_tests {
    use super::*;
    use crate::{
        EndpointCapabilities, EndpointStats, InferenceRequest, ModelRouteClass, Percentiles,
        Pricing, RoutingStrategy, WorkloadPolicy,
    };

    fn endpoint(name: &str, prompt_rate: f64, cache_rate: Option<f64>, latency: f64) -> Endpoint {
        Endpoint {
            source: "native".to_owned(),
            credential: Some("credential-a".to_owned()),
            provider_key: name.to_owned(),
            id: name.to_owned(),
            provider: name.to_owned(),
            model: "test/model".to_owned(),
            route_class: ModelRouteClass::Standard,
            display_name: name.to_owned(),
            quantization: None,
            status: 0,
            context_length: Some(8192),
            max_prompt_tokens: Some(8192),
            max_completion_tokens: Some(8192),
            pricing: Pricing {
                prompt_per_token: prompt_rate,
                cache_read_per_token: cache_rate,
                ..Pricing::default()
            },
            stats: EndpointStats {
                latency_seconds: Percentiles {
                    p50: Some(latency),
                    p75: Some(latency),
                    p95: Some(latency),
                    ..Percentiles::default()
                },
                throughput_tokens_per_second: Percentiles {
                    p50: Some(100.0),
                    ..Percentiles::default()
                },
                ..EndpointStats::default()
            },
            workload_policy: WorkloadPolicy::default(),
            capabilities: EndpointCapabilities::default(),
        }
    }

    fn request() -> InferenceRequest {
        let mut request = InferenceRequest::new("test/model", "prefix + suffix", 100);
        request.prompt_tokens = Some(1000);
        request.cache_prefix_fingerprint = Some("opaque-blake3-of-prefix".to_owned());
        request.session_id = Some("one-session".to_owned());
        request.requirements.minimum_uptime_30m = None;
        request
    }

    fn observation(endpoint: &Endpoint) -> CacheAffinityObservation {
        CacheAffinityObservation {
            route: QuotaRouteKey::from_endpoint(endpoint),
            endpoint_id: endpoint.id.clone(),
            prefix_fingerprint: "opaque-blake3-of-prefix".to_owned(),
            session_id: Some("one-session".to_owned()),
            cached_prompt_tokens: 900,
            observed_at_unix: 100,
            ttl_seconds: 200,
            confidence: 1.0,
            latency_savings_seconds: None,
        }
    }

    fn route(
        request: &InferenceRequest,
        profile: Option<&CacheAffinityProfile>,
        now_unix: u64,
    ) -> Result<crate::RoutePlan, crate::ProviderError> {
        crate::router::build_route_plan_with_affinity(
            RoutingStrategy::FastestResponseCheap(1.1),
            request,
            vec![
                endpoint("cold", 0.000_002, None, 0.4),
                endpoint("warm", 0.000_003, Some(0.000_000_2), 0.1),
            ],
            profile,
            now_unix,
        )
    }

    #[test]
    fn observed_cache_hit_can_enter_cost_band_and_win() -> Result<(), crate::ProviderError> {
        let request = request();
        let cold = route(&request, None, 100)?;
        assert_eq!(cold.candidates.len(), 1);
        assert_eq!(cold.candidates[0].endpoint.id, "cold");

        let mut profile = CacheAffinityProfile::default();
        let warm = endpoint("warm", 0.000_003, Some(0.000_000_2), 0.1);
        profile.observe(observation(&warm), 100);
        let plan = route(&request, Some(&profile), 100)?;
        assert_eq!(plan.candidates[0].endpoint.id, "warm");
        assert!(plan.candidates[0].expected_cost_usd < cold.cheapest_cost_usd);
        assert_eq!(plan.cheapest_cost_usd, plan.candidates[0].expected_cost_usd);
        Ok(())
    }

    #[test]
    fn affinity_expires_decays_and_remains_scoped() -> Result<(), crate::ProviderError> {
        let request = request();
        let warm = endpoint("warm", 0.000_003, Some(0.000_000_2), 0.1);
        let mut profile = CacheAffinityProfile::default();
        profile.observe(observation(&warm), 100);
        let early = route(&request, Some(&profile), 100)?;
        let decayed = route(&request, Some(&profile), 200)?;
        let expired = route(&request, Some(&profile), 300)?;
        assert!(decayed.cheapest_cost_usd > early.cheapest_cost_usd);
        assert_eq!(expired.candidates[0].endpoint.id, "cold");

        let mut other_session = request.clone();
        other_session.session_id = Some("another-session".to_owned());
        assert_eq!(
            route(&other_session, Some(&profile), 100)?.candidates[0]
                .endpoint
                .id,
            "cold"
        );
        let mut other_prefix = request.clone();
        other_prefix.cache_prefix_fingerprint = Some("different-prefix".to_owned());
        assert_eq!(
            route(&other_prefix, Some(&profile), 100)?.candidates[0]
                .endpoint
                .id,
            "cold"
        );
        Ok(())
    }

    #[test]
    fn cache_hit_telemetry_updates_and_zero_eviction() {
        let request = request();
        let warm = endpoint("warm", 0.000_003, Some(0.000_000_2), 0.1);
        let mut profile = CacheAffinityProfile::default();
        let response = |cached_tokens| crate::InferenceResponse {
            content: "ok".to_owned(),
            model: None,
            provider: None,
            usage_cost_usd: None,
            usage: crate::TokenUsage::default(),
            tool_calls: vec![],
            output_repairs: vec![],
            quota_observations: vec![],
            raw: serde_json::json!({"usage": {"prompt_tokens_details": {"cached_tokens": cached_tokens}}}),
        };
        profile.observe_response(&request, &warm, &response(800), 100);
        assert_eq!(profile.observations.len(), 1);
        assert_eq!(profile.observations[0].cached_prompt_tokens, 800);
        profile.observe_response(&request, &warm, &response(0), 101);
        assert_eq!(profile.observations.len(), 0);

        profile.observe(observation(&warm), 100);
        profile.observe(observation(&warm), 101);
        assert_eq!(profile.observations.len(), 1);
    }

    #[test]
    fn observations_round_trip_and_untrusted_values_do_not_affect_scores()
    -> Result<(), Box<dyn std::error::Error>> {
        let warm = endpoint("warm", 0.000_003, Some(0.000_000_2), 0.1);
        let mut profile = CacheAffinityProfile::default();
        let mut invalid = observation(&warm);
        invalid.confidence = f64::NAN;
        profile.observe(invalid, 100);
        invalid = observation(&warm);
        invalid.observed_at_unix = 101;
        profile.observe(invalid, 100);
        assert_eq!(profile.observations.len(), 0);

        profile.observe(observation(&warm), 100);
        let round_trip: CacheAffinityProfile =
            serde_json::from_str(&serde_json::to_string(&profile)?)?;
        assert_eq!(round_trip, profile);
        Ok(())
    }

    #[test]
    fn optional_latency_savings_changes_ranking_without_cache_pricing() {
        let mut profile = CacheAffinityProfile::default();
        let mut fast = endpoint("warm", 0.000_002, None, 0.1);
        fast.stats.latency_seconds.p75 = Some(0.9);
        let mut obs = observation(&fast);
        obs.latency_savings_seconds = Some(0.9);
        profile.observe(obs, 100);
        let mut candidate = RouteCandidate {
            endpoint: fast,
            expected_cost_usd: 0.002,
            expected_response_p75_seconds: Some(1.0),
            expected_response_p95_seconds: Some(2.0),
        };
        profile.adjust_candidate(&request(), 1000, &mut candidate, 100);
        assert_eq!(candidate.expected_cost_usd, 0.002);
        assert!(
            candidate
                .expected_response_p75_seconds
                .is_some_and(|value| (value - 0.1).abs() < 1e-12)
        );
    }
}
