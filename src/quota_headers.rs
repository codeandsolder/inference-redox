use reqwest::header::HeaderMap;

use crate::{QuotaMetric, QuotaObservation};

pub(crate) fn standard_rate_limit_observations(headers: &HeaderMap) -> Vec<QuotaObservation> {
    let requests = observation(
        headers,
        "rate_limit.requests",
        QuotaMetric::Requests,
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
    );
    let tokens = observation(
        headers,
        "rate_limit.tokens",
        QuotaMetric::TotalTokens,
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
    );
    [requests, tokens].into_iter().flatten().collect()
}

fn observation(
    headers: &HeaderMap,
    bucket: &str,
    metric: QuotaMetric,
    limit_header: &str,
    remaining_header: &str,
    reset_header: &str,
) -> Option<QuotaObservation> {
    let limit = integer_header(headers, limit_header);
    let remaining = integer_header(headers, remaining_header);
    let reset_after_millis = text_header(headers, reset_header).and_then(parse_duration_millis);
    if limit.is_none() && remaining.is_none() && reset_after_millis.is_none() {
        return None;
    }
    Some(QuotaObservation {
        bucket: bucket.to_owned(),
        metric: Some(metric),
        limit,
        remaining,
        reset_after_millis,
    })
}

fn text_header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name)?.to_str().ok().map(str::trim)
}

fn integer_header(headers: &HeaderMap, name: &str) -> Option<u64> {
    text_header(headers, name)?.parse().ok()
}

fn parse_duration_millis(raw: &str) -> Option<u64> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let bytes = raw.as_bytes();
    let mut index = 0_usize;
    let mut total = 0_u64;
    while index < bytes.len() {
        let start = index;
        while index < bytes.len() && (bytes[index].is_ascii_digit() || bytes[index] == b'.') {
            index += 1;
        }
        if start == index {
            return None;
        }
        let number = &raw[start..index];
        let unit_start = index;
        while index < bytes.len() && bytes[index].is_ascii_alphabetic() {
            index += 1;
        }
        let unit = &raw[unit_start..index];
        let component = match unit {
            "h" => decimal_component_millis(number, 3_600_000)?,
            "m" => decimal_component_millis(number, 60_000)?,
            "s" => decimal_component_millis(number, 1_000)?,
            "ms" => decimal_component_millis(number, 1)?,
            _ => return None,
        };
        total = total.checked_add(component)?;
    }
    Some(total)
}

fn decimal_component_millis(raw: &str, unit_millis: u64) -> Option<u64> {
    let (whole, fraction) = raw.split_once('.').unwrap_or((raw, ""));
    let whole: u64 = if whole.is_empty() {
        0
    } else {
        whole.parse().ok()?
    };
    let whole_millis = whole.checked_mul(unit_millis)?;
    if fraction.is_empty() {
        return Some(whole_millis);
    }
    if !fraction.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let digits = fraction.len().min(6);
    let numerator: u64 = fraction[..digits].parse().ok()?;
    let denominator = 10_u64.checked_pow(u32::try_from(digits).ok()?)?;
    let fraction_millis = numerator
        .checked_mul(unit_millis)?
        .checked_div(denominator)?;
    whole_millis.checked_add(fraction_millis)
}

#[cfg(test)]
mod tests {
    use reqwest::header::{HeaderMap, HeaderValue};

    use super::*;

    #[test]
    fn parses_groq_style_request_and_token_buckets() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-ratelimit-limit-requests",
            HeaderValue::from_static("14400"),
        );
        headers.insert(
            "x-ratelimit-remaining-requests",
            HeaderValue::from_static("14370"),
        );
        headers.insert(
            "x-ratelimit-reset-requests",
            HeaderValue::from_static("2m59.56s"),
        );
        headers.insert(
            "x-ratelimit-limit-tokens",
            HeaderValue::from_static("18000"),
        );
        headers.insert(
            "x-ratelimit-remaining-tokens",
            HeaderValue::from_static("17997"),
        );
        headers.insert(
            "x-ratelimit-reset-tokens",
            HeaderValue::from_static("7.66s"),
        );
        let observations = standard_rate_limit_observations(&headers);
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0].bucket, "rate_limit.requests");
        assert_eq!(observations[0].metric, Some(QuotaMetric::Requests));
        assert_eq!(observations[0].limit, Some(14_400));
        assert_eq!(observations[0].remaining, Some(14_370));
        assert_eq!(observations[0].reset_after_millis, Some(179_560));
        assert_eq!(observations[1].metric, Some(QuotaMetric::TotalTokens));
        assert_eq!(observations[1].reset_after_millis, Some(7_660));
    }

    #[test]
    fn ignores_unrecognized_or_malformed_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-ratelimit-limit-requests",
            HeaderValue::from_static("nope"),
        );
        headers.insert(
            "x-ratelimit-reset-requests",
            HeaderValue::from_static("tomorrow"),
        );
        headers.insert("x-secret-quota-thing", HeaderValue::from_static("123"));
        assert_eq!(standard_rate_limit_observations(&headers), Vec::new());
    }
}
