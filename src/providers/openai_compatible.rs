use std::time::Duration;

use reqwest::{
    Client, StatusCode,
    header::{HeaderMap, RETRY_AFTER},
};
use serde_json::{Map, Value, json};
use tokio::time::{sleep, timeout};

use crate::{
    ExecutionFailurePhase, InferenceRequest, InferenceResponse, ProviderError, ResponseSchema,
    RetryPolicy, TokenUsage,
};

pub(crate) fn base_chat_body(
    request: &InferenceRequest,
    max_tokens_field: &str,
) -> Map<String, Value> {
    let mut messages = Vec::with_capacity(2);
    if let Some(system_prompt) = request.system_prompt.as_ref() {
        messages.push(json!({"role": "system", "content": system_prompt}));
    }
    messages.push(json!({"role": "user", "content": request.prompt}));

    let mut body = Map::new();
    body.insert("model".to_owned(), json!(request.model));
    body.insert("messages".to_owned(), Value::Array(messages));
    if let Some(maximum) = request.max_output_tokens {
        body.insert(max_tokens_field.to_owned(), json!(maximum));
    }
    if let Some(temperature) = request.temperature {
        body.insert("temperature".to_owned(), json!(temperature));
    }
    body
}

pub(crate) fn response_format(schema: &ResponseSchema) -> Value {
    match schema {
        ResponseSchema::JsonObject => json!({"type": "json_object"}),
        ResponseSchema::JsonSchema {
            name,
            schema,
            strict,
        } => {
            let lowered = crate::lower_json_schema(schema);
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": lowered,
                    "strict": strict,
                }
            })
        }
    }
}

#[derive(Debug)]
struct AttemptFailure {
    phase: ExecutionFailurePhase,
    detail: String,
    retryable: bool,
    retry_after: Option<Duration>,
    quota_observations: Vec<crate::QuotaObservation>,
}

impl AttemptFailure {
    fn into_provider_error(self) -> ProviderError {
        ProviderError::Execution {
            phase: self.phase,
            detail: self.detail,
            quota_observations: self.quota_observations,
        }
    }
}

fn retryable_status(status: StatusCode) -> bool {
    status == StatusCode::REQUEST_TIMEOUT
        || status == StatusCode::TOO_MANY_REQUESTS
        || status.is_server_error()
}

fn retry_after(response: &reqwest::Response) -> Option<Duration> {
    response
        .headers()
        .get(RETRY_AFTER)?
        .to_str()
        .ok()?
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

fn http_error_detail(status: StatusCode, body: &str) -> String {
    let parsed = serde_json::from_str::<Value>(body).ok();
    let error_type = parsed
        .as_ref()
        .and_then(|value| value.pointer("/error/type"))
        .and_then(Value::as_str);
    let error_code = parsed
        .as_ref()
        .and_then(|value| value.pointer("/error/code"))
        .and_then(Value::as_str);
    match (error_type, error_code) {
        (Some(error_type), Some(error_code)) => {
            format!("HTTP {status} ({error_type}/{error_code})")
        }
        (Some(error_type), None) => format!("HTTP {status} ({error_type})"),
        (None, Some(error_code)) => format!("HTTP {status} ({error_code})"),
        (None, None) => format!("HTTP {status}"),
    }
}

fn response_shape_detail(provider: &str, raw: &Value) -> String {
    let message_keys = raw
        .pointer("/choices/0/message")
        .and_then(Value::as_object)
        .map(|message| message.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    let finish_reason = raw
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str)
        .unwrap_or("missing");
    let prompt_tokens = raw.pointer("/usage/prompt_tokens").and_then(Value::as_u64);
    let completion_tokens = raw
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64);
    format!(
        "{provider} response had no non-empty string content; message_keys={message_keys:?}; finish_reason={finish_reason}; prompt_tokens={prompt_tokens:?}; completion_tokens={completion_tokens:?}"
    )
}

#[derive(Debug)]
pub(crate) struct JsonResponse {
    pub raw: Value,
    pub quota_observations: Vec<crate::QuotaObservation>,
}

async fn execute_once(
    client: &Client,
    api_key: Option<&str>,
    url: &str,
    body: &Value,
    headers: &HeaderMap,
) -> Result<JsonResponse, AttemptFailure> {
    let mut request = client.post(url).headers(headers.clone()).json(body);
    if let Some(api_key) = api_key {
        request = request.bearer_auth(api_key);
    }
    let response = request.send().await.map_err(|error| AttemptFailure {
        phase: ExecutionFailurePhase::TransportUnknown,
        detail: error.to_string(),
        retryable: true,
        retry_after: None,
        quota_observations: Vec::new(),
    })?;
    let status = response.status();
    let retry_after = retry_after(&response);
    let quota_observations =
        crate::quota_headers::standard_rate_limit_observations(response.headers());
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(AttemptFailure {
            phase: ExecutionFailurePhase::ResponseReceived,
            detail: http_error_detail(status, &body),
            retryable: retryable_status(status),
            retry_after,
            quota_observations,
        });
    }
    let raw: Value = response.json().await.map_err(|error| AttemptFailure {
        phase: ExecutionFailurePhase::ResponseReceived,
        detail: error.to_string(),
        retryable: false,
        retry_after,
        quota_observations: quota_observations.clone(),
    })?;
    Ok(JsonResponse {
        raw,
        quota_observations,
    })
}

pub(crate) async fn execute_json_with_retry_headers(
    client: &Client,
    api_key: Option<&str>,
    url: &str,
    body: &Value,
    headers: &HeaderMap,
    retry: RetryPolicy,
) -> Result<JsonResponse, ProviderError> {
    if retry.max_attempts == 0 {
        return Err(ProviderError::InvalidRequest(
            "retry.max_attempts must be at least 1".to_owned(),
        ));
    }
    if retry.attempt_timeout.is_zero() {
        return Err(ProviderError::InvalidRequest(
            "retry.attempt_timeout must be greater than zero".to_owned(),
        ));
    }
    let mut backoff = retry.initial_backoff;
    let mut last_error = None;
    let mut last_phase = None;
    let mut last_quota_observations = Vec::new();

    for attempt in 1..=retry.max_attempts {
        let result = timeout(
            retry.attempt_timeout,
            execute_once(client, api_key, url, body, headers),
        )
        .await
        .unwrap_or_else(|_| {
            Err(AttemptFailure {
                phase: ExecutionFailurePhase::TransportUnknown,
                detail: format!(
                    "request attempt timed out after {} ms",
                    retry.attempt_timeout.as_millis()
                ),
                retryable: true,
                retry_after: None,
                quota_observations: Vec::new(),
            })
        });
        match result {
            Ok(response) => return Ok(response),
            Err(failure) if !failure.retryable => return Err(failure.into_provider_error()),
            Err(failure) => {
                if last_phase.is_none() || failure.phase == ExecutionFailurePhase::ResponseReceived
                {
                    last_phase = Some(failure.phase);
                }
                if !failure.quota_observations.is_empty() {
                    last_quota_observations = failure.quota_observations;
                }
                last_error = Some(failure.detail);
                if attempt < retry.max_attempts {
                    let delay = failure
                        .retry_after
                        .unwrap_or(backoff)
                        .min(retry.max_backoff);
                    sleep(delay).await;
                    backoff = backoff.saturating_mul(2).min(retry.max_backoff);
                }
            }
        }
    }
    Err(ProviderError::Execution {
        phase: last_phase.unwrap_or(ExecutionFailurePhase::TransportUnknown),
        detail: last_error.unwrap_or_else(|| "request failed without an error body".to_owned()),
        quota_observations: last_quota_observations,
    })
}

pub(crate) async fn execute_json_with_retry(
    client: &Client,
    api_key: Option<&str>,
    url: &str,
    body: &Value,
    retry: RetryPolicy,
) -> Result<JsonResponse, ProviderError> {
    execute_json_with_retry_headers(client, api_key, url, body, &HeaderMap::new(), retry).await
}

pub(crate) async fn execute_chat_with_retry(
    client: &Client,
    api_key: Option<&str>,
    url: &str,
    body: &Value,
    provider: &str,
    usage_cost_usd: Option<f64>,
    retry: RetryPolicy,
) -> Result<InferenceResponse, ProviderError> {
    let JsonResponse {
        raw,
        quota_observations,
    } = execute_json_with_retry(client, api_key, url, body, retry).await?;
    let content = raw
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .filter(|content| !content.is_empty())
        .ok_or_else(|| ProviderError::Execution {
            phase: ExecutionFailurePhase::ResponseReceived,
            detail: response_shape_detail(provider, &raw),
            quota_observations: quota_observations.clone(),
        })?;
    Ok(InferenceResponse {
        content: content.to_owned(),
        model: raw.get("model").and_then(Value::as_str).map(str::to_owned),
        provider: Some(provider.to_owned()),
        usage_cost_usd,
        usage: TokenUsage::from_openai_compatible(&raw),
        tool_calls: Vec::new(),
        output_repairs: Vec::new(),
        quota_observations,
        raw,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_common_chat_fields() {
        let mut request = InferenceRequest::new("vendor/model", "hello", 16);
        request.system_prompt = Some("system".to_owned());
        request.max_output_tokens = Some(32);
        request.temperature = Some(0.25);
        let body = base_chat_body(&request, "max_completion_tokens");
        assert_eq!(
            body.get("max_completion_tokens").and_then(Value::as_u64),
            Some(32)
        );
        assert_eq!(body.get("temperature").and_then(Value::as_f64), Some(0.25));
        assert_eq!(
            body.get("messages").and_then(Value::as_array).map(Vec::len),
            Some(2)
        );
    }

    #[test]
    fn http_error_detail_does_not_echo_provider_message() {
        let body = r#"{"error":{"type":"invalid_request_error","code":"bad_schema","message":"sensitive echoed prompt"}}"#;
        let detail = http_error_detail(StatusCode::BAD_REQUEST, body);
        assert_eq!(
            detail,
            "HTTP 400 Bad Request (invalid_request_error/bad_schema)"
        );
        assert!(!detail.contains("sensitive"));
    }
}
