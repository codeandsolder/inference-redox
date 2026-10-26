use std::collections::BTreeMap;

use jsonschema::JsonType;
use jsonschema::error::{TypeKind, ValidationErrorKind};
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};
use thiserror::Error;

use crate::SchemaFeature;

/// A deliberately narrow, schema-directed repair applied to otherwise-invalid structured output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputRepairKind {
    MarkdownFence,
    StringToInteger,
    StringToNumber,
    StringToBoolean,
    StringToNull,
    StringToArray,
    StringToObject,
    IntegerToBoolean,
    BooleanToInteger,
}

/// One local repair. The path is a JSON Pointer into the output instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputRepair {
    pub instance_path: String,
    pub kind: OutputRepairKind,
}

/// One local JSON-Schema validation failure, without echoing the response value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaViolation {
    pub keyword: String,
    pub instance_path: String,
    pub schema_path: String,
}

/// Local structured-output validation failed even after the safe repair whitelist was exhausted.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("structured output failed JSON-Schema validation: {summary}")]
pub struct StructuredOutputFailure {
    pub summary: String,
    pub violations: Vec<SchemaViolation>,
}

#[derive(Debug, Clone, PartialEq)]
struct RepairCandidate {
    instance_path: String,
    value: Value,
    kind: OutputRepairKind,
}

/// Best-effort semantic lowering using `jsonschema`'s canonical IR.
///
/// Canonicalization is deliberately non-fatal: unsupported constructs remain in their original
/// form rather than becoming a locally invented approximation. The original schema remains the
/// authority for response validation.
#[must_use]
pub fn lower_json_schema(schema: &Value) -> Value {
    if contains_model_guidance(schema) || !has_lowering_opportunity(schema) {
        return schema.clone();
    }
    let original_features = crate::schema_feature_requirements(schema);
    let Ok(canonical) = jsonschema::canonical::options()
        .offline()
        .canonicalize(schema)
    else {
        return schema.clone();
    };
    let mut candidate = canonical.to_json_schema();
    // The provider already selected a structured-output dialect. Repeating the meta-schema URI on
    // the wire is semantically unnecessary and is itself rejected by some provider subsets.
    if let Some(object) = candidate.as_object_mut() {
        object.remove("$schema");
    }
    let candidate_features = crate::schema_feature_requirements(&candidate);
    let introduced: std::collections::BTreeSet<SchemaFeature> = candidate_features
        .difference(&original_features)
        .copied()
        .collect();
    let enum_simplification = introduced == std::collections::BTreeSet::from([SchemaFeature::Enum])
        && (original_features.contains(&SchemaFeature::OneOf)
            || original_features.contains(&SchemaFeature::AnyOf))
        && !candidate_features.contains(&SchemaFeature::OneOf)
        && !candidate_features.contains(&SchemaFeature::AnyOf);
    if introduced.is_empty() || enum_simplification {
        candidate
    } else {
        schema.clone()
    }
}

fn has_lowering_opportunity(value: &Value) -> bool {
    match value {
        Value::Object(map) => {
            map.keys()
                .any(|key| matches!(key.as_str(), "oneOf" | "anyOf" | "allOf"))
                || map.values().any(has_lowering_opportunity)
        }
        Value::Array(values) => values.iter().any(has_lowering_opportunity),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => false,
    }
}

fn contains_model_guidance(value: &Value) -> bool {
    match value {
        Value::Object(map) => {
            map.keys().any(|key| {
                matches!(
                    key.as_str(),
                    "title"
                        | "description"
                        | "default"
                        | "examples"
                        | "deprecated"
                        | "readOnly"
                        | "writeOnly"
                )
            }) || map.values().any(contains_model_guidance)
        }
        Value::Array(values) => values.iter().any(contains_model_guidance),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => false,
    }
}

/// Verify that a strict caller schema can be checked locally without network access.
pub(crate) fn validate_schema_offline(schema: &Value) -> Result<(), StructuredOutputFailure> {
    jsonschema::options()
        .offline()
        .build(schema)
        .map(|_| ())
        .map_err(|error| StructuredOutputFailure {
            summary: format!("strict JSON Schema is not locally resolvable offline: {error}"),
            violations: Vec::new(),
        })
}

/// Parse, narrowly repair, and validate one JSON response against the original schema.
///
/// Repairs are accepted only when the final value validates completely. No fields are invented,
/// removed, clamped, truncated, or branch-selected on the model's behalf.
///
/// # Errors
/// Returns `StructuredOutputFailure` when the text is not parseable JSON, the schema cannot be
/// resolved locally in offline mode, or the value remains invalid after the narrow repair pass.
pub fn validate_and_repair_json_text(
    schema: &Value,
    text: &str,
) -> Result<(String, Vec<OutputRepair>), StructuredOutputFailure> {
    let (value, mut repairs) = parse_json_text(text)?;
    let (value, mut value_repairs) = validate_and_repair_value(schema, value)?;
    repairs.append(&mut value_repairs);
    let content = serde_json::to_string(&value).map_err(|error| StructuredOutputFailure {
        summary: format!("failed to serialize validated JSON: {error}"),
        violations: Vec::new(),
    })?;
    Ok((content, repairs))
}

/// Narrowly repair and validate an already-parsed JSON value.
///
/// # Errors
/// Returns `StructuredOutputFailure` when the schema cannot be resolved locally in offline mode or
/// the value remains invalid after the narrow repair pass.
pub fn validate_and_repair_value(
    schema: &Value,
    value: Value,
) -> Result<(Value, Vec<OutputRepair>), StructuredOutputFailure> {
    let validator = jsonschema::options()
        .offline()
        .build(schema)
        .map_err(|error| StructuredOutputFailure {
            summary: format!("schema cannot be validated locally offline: {error}"),
            violations: Vec::new(),
        })?;
    if validator.is_valid(&value) {
        return Ok((value, Vec::new()));
    }

    let mut candidate_value = value;
    let mut repairs = Vec::new();
    for _ in 0..4 {
        let candidates = repair_candidates(&validator, &candidate_value);
        if candidates.is_empty() {
            break;
        }
        let mut changed = false;
        for candidate in candidates {
            if replace_at_pointer(
                &mut candidate_value,
                &candidate.instance_path,
                candidate.value,
            ) {
                repairs.push(OutputRepair {
                    instance_path: candidate.instance_path,
                    kind: candidate.kind,
                });
                changed = true;
            }
        }
        if validator.is_valid(&candidate_value) {
            return Ok((candidate_value, repairs));
        }
        if !changed {
            break;
        }
    }

    Err(validation_failure(&validator, &candidate_value))
}

fn parse_json_text(text: &str) -> Result<(Value, Vec<OutputRepair>), StructuredOutputFailure> {
    if let Ok(value) = serde_json::from_str(text) {
        return Ok((value, Vec::new()));
    }
    if let Some(inner) = exact_markdown_fence_body(text)
        && let Ok(value) = serde_json::from_str(inner)
    {
        return Ok((
            value,
            vec![OutputRepair {
                instance_path: String::new(),
                kind: OutputRepairKind::MarkdownFence,
            }],
        ));
    }
    Err(StructuredOutputFailure {
        summary: "response was not JSON and had no exact JSON Markdown fence".to_owned(),
        violations: vec![SchemaViolation {
            keyword: "json_parse".to_owned(),
            instance_path: String::new(),
            schema_path: String::new(),
        }],
    })
}

fn exact_markdown_fence_body(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    let rest = trimmed.strip_prefix("```")?;
    let (first_line, body) = rest.split_once('\n')?;
    if !first_line.trim().is_empty() && !first_line.trim().eq_ignore_ascii_case("json") {
        return None;
    }
    body.strip_suffix("```").map(str::trim)
}

fn validation_failure(validator: &jsonschema::Validator, value: &Value) -> StructuredOutputFailure {
    let violations: Vec<SchemaViolation> = validator
        .iter_errors(value)
        .take(16)
        .map(|error| SchemaViolation {
            keyword: error.kind().keyword().to_owned(),
            instance_path: error.instance_path().to_string(),
            schema_path: error.schema_path().to_string(),
        })
        .collect();
    let summary = violations.first().map_or_else(
        || "output is invalid for the requested schema".to_owned(),
        |violation| {
            format!(
                "keyword {} failed at instance {} (schema {})",
                violation.keyword,
                display_path(&violation.instance_path),
                display_path(&violation.schema_path)
            )
        },
    );
    StructuredOutputFailure {
        summary,
        violations,
    }
}

fn display_path(path: &str) -> &str {
    if path.is_empty() { "/" } else { path }
}

fn repair_candidates(validator: &jsonschema::Validator, value: &Value) -> Vec<RepairCandidate> {
    let mut by_path: BTreeMap<String, Vec<RepairCandidate>> = BTreeMap::new();
    for error in validator.iter_errors(value) {
        let path = error.instance_path().to_string();
        let Some(current) = value_at_pointer(value, &path) else {
            continue;
        };
        let candidates = match error.kind() {
            ValidationErrorKind::Type { kind } => type_repair_candidates(&path, current, kind),
            ValidationErrorKind::Constant { expected_value } => {
                exact_value_repair_candidate(&path, current, expected_value.as_ref())
                    .into_iter()
                    .collect()
            }
            ValidationErrorKind::Enum { options } => options
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|expected| exact_value_repair_candidate(&path, current, expected))
                .collect(),
            _ => Vec::new(),
        };
        by_path.entry(path).or_default().extend(candidates);
    }

    by_path
        .into_values()
        .filter_map(unique_repair_candidate)
        .collect()
}

fn unique_repair_candidate(candidates: Vec<RepairCandidate>) -> Option<RepairCandidate> {
    let mut unique: Vec<RepairCandidate> = Vec::new();
    for candidate in candidates {
        if unique
            .iter()
            .all(|existing| existing.value != candidate.value)
        {
            unique.push(candidate);
        }
    }
    if unique.len() == 1 {
        unique.into_iter().next()
    } else {
        None
    }
}

fn type_repair_candidates(path: &str, current: &Value, kind: &TypeKind) -> Vec<RepairCandidate> {
    let types: Vec<JsonType> = match kind {
        TypeKind::Single(json_type) => vec![*json_type],
        TypeKind::Multiple(types) => types.iter().collect(),
    };
    types
        .into_iter()
        .filter_map(|json_type| coerce_to_type(path, current, json_type))
        .collect()
}

fn coerce_to_type(path: &str, current: &Value, expected: JsonType) -> Option<RepairCandidate> {
    match (current, expected) {
        (Value::String(text), JsonType::Integer) => {
            let parsed = parse_json_scalar(text)?;
            parsed
                .as_number()
                .filter(|number| number_is_integer(number))?;
            Some(candidate(path, parsed, OutputRepairKind::StringToInteger))
        }
        (Value::String(text), JsonType::Number) => {
            let parsed = parse_json_scalar(text)?;
            parsed.as_number()?;
            Some(candidate(path, parsed, OutputRepairKind::StringToNumber))
        }
        (Value::String(text), JsonType::Boolean) => {
            let parsed = parse_json_scalar(text)?;
            parsed.as_bool()?;
            Some(candidate(path, parsed, OutputRepairKind::StringToBoolean))
        }
        (Value::String(text), JsonType::Null) => {
            let parsed = parse_json_scalar(text)?;
            if parsed.is_null() {
                Some(candidate(path, parsed, OutputRepairKind::StringToNull))
            } else {
                None
            }
        }
        (Value::String(text), JsonType::Array) => {
            let parsed: Value = serde_json::from_str(text).ok()?;
            parsed.as_array()?;
            Some(candidate(path, parsed, OutputRepairKind::StringToArray))
        }
        (Value::String(text), JsonType::Object) => {
            let parsed: Value = serde_json::from_str(text).ok()?;
            parsed.as_object()?;
            Some(candidate(path, parsed, OutputRepairKind::StringToObject))
        }
        (Value::Number(number), JsonType::Boolean) => number_to_bool(number)
            .map(|value| candidate(path, Value::Bool(value), OutputRepairKind::IntegerToBoolean)),
        (Value::Bool(value), JsonType::Integer | JsonType::Number) => Some(candidate(
            path,
            Value::Number(Number::from(u8::from(*value))),
            OutputRepairKind::BooleanToInteger,
        )),
        _ => None,
    }
}

fn exact_value_repair_candidate(
    path: &str,
    current: &Value,
    expected: &Value,
) -> Option<RepairCandidate> {
    match (current, expected) {
        (Value::String(text), Value::Number(number)) => {
            let parsed = parse_json_scalar(text)?;
            if parsed == *expected {
                let kind = if number_is_integer(number) {
                    OutputRepairKind::StringToInteger
                } else {
                    OutputRepairKind::StringToNumber
                };
                Some(candidate(path, parsed, kind))
            } else {
                None
            }
        }
        (Value::String(text), Value::Bool(_)) => {
            let parsed = parse_json_scalar(text)?;
            (parsed == *expected)
                .then(|| candidate(path, parsed, OutputRepairKind::StringToBoolean))
        }
        (Value::String(text), Value::Null) => {
            let parsed = parse_json_scalar(text)?;
            (parsed == *expected).then(|| candidate(path, parsed, OutputRepairKind::StringToNull))
        }
        (Value::String(text), Value::Array(_)) => {
            let parsed: Value = serde_json::from_str(text).ok()?;
            (parsed == *expected).then(|| candidate(path, parsed, OutputRepairKind::StringToArray))
        }
        (Value::String(text), Value::Object(_)) => {
            let parsed: Value = serde_json::from_str(text).ok()?;
            (parsed == *expected).then(|| candidate(path, parsed, OutputRepairKind::StringToObject))
        }
        (Value::Number(number), Value::Bool(expected_bool)) => number_to_bool(number)
            .filter(|value| value == expected_bool)
            .map(|value| candidate(path, Value::Bool(value), OutputRepairKind::IntegerToBoolean)),
        (Value::Bool(value), Value::Number(number)) => {
            bool_to_matching_number(*value, number).map(|number| {
                candidate(
                    path,
                    Value::Number(number),
                    OutputRepairKind::BooleanToInteger,
                )
            })
        }
        _ => None,
    }
}

fn parse_json_scalar(text: &str) -> Option<Value> {
    let value: Value = serde_json::from_str(text).ok()?;
    (!value.is_array() && !value.is_object()).then_some(value)
}

fn number_to_bool(number: &Number) -> Option<bool> {
    match number.as_i64() {
        Some(0) => Some(false),
        Some(1) => Some(true),
        _ => match number.as_u64() {
            Some(0) => Some(false),
            Some(1) => Some(true),
            _ => None,
        },
    }
}

fn bool_to_matching_number(value: bool, expected: &Number) -> Option<Number> {
    let desired = u64::from(value);
    let signed = i64::try_from(desired).ok();
    (expected.as_u64() == Some(desired) || expected.as_i64() == signed).then(|| expected.clone())
}

fn number_is_integer(number: &Number) -> bool {
    number.is_i64()
        || number.is_u64()
        || number
            .as_f64()
            .is_some_and(|value| value.is_finite() && value.fract() == 0.0)
}

fn candidate(path: &str, value: Value, kind: OutputRepairKind) -> RepairCandidate {
    RepairCandidate {
        instance_path: path.to_owned(),
        value,
        kind,
    }
}

fn value_at_pointer<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        Some(value)
    } else {
        value.pointer(path)
    }
}

fn replace_at_pointer(value: &mut Value, path: &str, replacement: Value) -> bool {
    if path.is_empty() {
        *value = replacement;
        return true;
    }
    let Some(target) = value.pointer_mut(path) else {
        return false;
    };
    *target = replacement;
    true
}

/// Enforce the application-visible response contract locally after provider execution.
pub(crate) fn validate_inference_response(
    request: &crate::InferenceRequest,
    mut response: crate::InferenceResponse,
) -> Result<crate::InferenceResponse, StructuredOutputFailure> {
    if let Some(schema) = match request.requirements.response_schema.as_ref() {
        Some(crate::ResponseSchema::JsonObject) => Some(serde_json::json!({"type": "object"})),
        Some(crate::ResponseSchema::JsonSchema {
            schema,
            strict: true,
            ..
        }) => Some(schema.clone()),
        Some(crate::ResponseSchema::JsonSchema { strict: false, .. }) | None => None,
    } {
        let (content, repairs) = validate_and_repair_json_text(&schema, &response.content)?;
        response.content = content;
        response.output_repairs.extend(repairs);
    }
    validate_tool_calls(request, &mut response)?;
    Ok(response)
}

fn validate_tool_calls(
    request: &crate::InferenceRequest,
    response: &mut crate::InferenceResponse,
) -> Result<(), StructuredOutputFailure> {
    use crate::ToolChoice;

    if request.tools.is_empty() {
        return Ok(());
    }
    let call_count = response.tool_calls.len();
    if request.parallel_tool_calls == Some(false) && call_count > 1 {
        return Err(tool_failure(
            "parallel_tool_calls",
            "/tool_calls",
            "multiple tool calls were returned while parallel_tool_calls=false",
        ));
    }
    match request.tool_choice.as_ref() {
        Some(ToolChoice::Required | ToolChoice::Function { .. }) if call_count == 0 => {
            return Err(tool_failure(
                "tool_choice",
                "/tool_calls",
                "the request required a tool call but none was returned",
            ));
        }
        Some(ToolChoice::None) if call_count != 0 => {
            return Err(tool_failure(
                "tool_choice",
                "/tool_calls",
                "tool calls were returned while tool_choice=none",
            ));
        }
        _ => {}
    }

    for (index, call) in response.tool_calls.iter_mut().enumerate() {
        let base = format!("/tool_calls/{index}");
        let Some(tool) = request.tools.iter().find(|tool| tool.name == call.name) else {
            return Err(tool_failure(
                "tool_name",
                &format!("{base}/name"),
                "the provider returned a function that was not offered",
            ));
        };
        if let Some(ToolChoice::Function { name }) = request.tool_choice.as_ref()
            && call.name != *name
        {
            return Err(tool_failure(
                "tool_choice",
                &format!("{base}/name"),
                "the provider returned a function other than the exact requested function",
            ));
        }
        if !tool.strict {
            continue;
        }
        match validate_and_repair_value(&tool.parameters, call.arguments.clone()) {
            Ok((arguments, repairs)) => {
                call.arguments = arguments;
                response
                    .output_repairs
                    .extend(repairs.into_iter().map(|repair| OutputRepair {
                        instance_path: prefix_pointer(
                            &format!("{base}/arguments"),
                            &repair.instance_path,
                        ),
                        kind: repair.kind,
                    }));
            }
            Err(failure) => {
                return Err(prefix_failure(failure, &format!("{base}/arguments")));
            }
        }
    }
    Ok(())
}

fn tool_failure(keyword: &str, instance_path: &str, summary: &str) -> StructuredOutputFailure {
    StructuredOutputFailure {
        summary: summary.to_owned(),
        violations: vec![SchemaViolation {
            keyword: keyword.to_owned(),
            instance_path: instance_path.to_owned(),
            schema_path: String::new(),
        }],
    }
}

fn prefix_failure(
    mut failure: StructuredOutputFailure,
    instance_prefix: &str,
) -> StructuredOutputFailure {
    for violation in &mut failure.violations {
        violation.instance_path = prefix_pointer(instance_prefix, &violation.instance_path);
    }
    failure
}

fn prefix_pointer(prefix: &str, path: &str) -> String {
    if path.is_empty() {
        prefix.to_owned()
    } else {
        format!("{prefix}{path}")
    }
}

/// Map a validator keyword failure onto the existing route-level schema capability model.
#[must_use]
pub(crate) fn schema_feature_for_validation_keyword(keyword: &str) -> Option<SchemaFeature> {
    match keyword {
        "oneOf" => Some(SchemaFeature::OneOf),
        "anyOf" => Some(SchemaFeature::AnyOf),
        "allOf" => Some(SchemaFeature::AllOf),
        "enum" => Some(SchemaFeature::Enum),
        "$ref" => Some(SchemaFeature::DefsRef),
        "contains" => Some(SchemaFeature::Contains),
        "uniqueItems" => Some(SchemaFeature::UniqueItems),
        "not" => Some(SchemaFeature::Not),
        "dependentRequired" => Some(SchemaFeature::DependentRequired),
        "propertyNames" => Some(SchemaFeature::PropertyNames),
        "minProperties" | "maxProperties" => Some(SchemaFeature::ObjectBounds),
        "patternProperties" => Some(SchemaFeature::PatternProperties),
        "dependentSchemas" => Some(SchemaFeature::DependentSchemas),
        "unevaluatedProperties" => Some(SchemaFeature::UnevaluatedProperties),
        "unevaluatedItems" => Some(SchemaFeature::UnevaluatedItems),
        "format" => Some(SchemaFeature::Format),
        "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" => {
            Some(SchemaFeature::NumericBounds)
        }
        "multipleOf" => Some(SchemaFeature::MultipleOf),
        "pattern" => Some(SchemaFeature::StringPattern),
        "minLength" | "maxLength" => Some(SchemaFeature::StringBounds),
        "minItems" | "maxItems" => Some(SchemaFeature::ArrayBounds),
        "additionalItems" => Some(SchemaFeature::LegacyTupleItems),
        "tool_choice" | "tool_name" => Some(SchemaFeature::ToolTurnEnvelope),
        "parallel_tool_calls" => Some(SchemaFeature::ToolTurnMultiple),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn coerces_string_to_integer_only_when_schema_requires_it()
    -> Result<(), StructuredOutputFailure> {
        let schema = json!({"type":"object","properties":{"n":{"type":"integer"}},"required":["n"],"additionalProperties":false});
        let (content, repairs) = validate_and_repair_json_text(&schema, r#"{"n":"1"}"#)?;
        assert_eq!(
            serde_json::from_str::<Value>(&content).ok(),
            Some(json!({"n":1}))
        );
        assert_eq!(repairs.len(), 1);
        assert_eq!(repairs[0].kind, OutputRepairKind::StringToInteger);
        Ok(())
    }

    #[test]
    fn repairs_boolean_integer_confusion_for_exact_enum() -> Result<(), StructuredOutputFailure> {
        let schema = json!({"enum":[true]});
        let (value, repairs) = validate_and_repair_value(&schema, json!(1))?;
        assert_eq!(value, json!(true));
        assert_eq!(repairs.len(), 1);
        assert_eq!(repairs[0].kind, OutputRepairKind::IntegerToBoolean);
        Ok(())
    }

    #[test]
    fn refuses_semantic_invention() {
        let schema = json!({"type":"object","properties":{"n":{"type":"integer"}},"required":["n"],"additionalProperties":false});
        assert!(validate_and_repair_json_text(&schema, r#"{"n":"banana"}"#).is_err());
    }

    #[test]
    fn strips_only_exact_json_markdown_fence() -> Result<(), StructuredOutputFailure> {
        let schema = json!({"type":"object","properties":{"ok":{"const":true}},"required":["ok"],"additionalProperties":false});
        let (content, repairs) =
            validate_and_repair_json_text(&schema, "```json\n{\"ok\":true}\n```")?;
        assert_eq!(
            serde_json::from_str::<Value>(&content).ok(),
            Some(json!({"ok":true}))
        );
        assert_eq!(repairs.len(), 1);
        assert_eq!(repairs[0].kind, OutputRepairKind::MarkdownFence);
        Ok(())
    }

    #[test]
    fn repairs_strict_tool_arguments_at_the_tool_path() -> Result<(), StructuredOutputFailure> {
        let mut request = crate::InferenceRequest::new("x/model", "lookup", 8);
        request.tools.push(crate::FunctionTool {
            name: "lookup".to_owned(),
            description: None,
            parameters: json!({
                "type":"object",
                "properties":{"n":{"type":"integer"}},
                "required":["n"],
                "additionalProperties":false
            }),
            strict: true,
        });
        request.tool_choice = Some(crate::ToolChoice::Required);
        let response = crate::InferenceResponse {
            content: String::new(),
            model: None,
            provider: None,
            usage_cost_usd: None,
            usage: crate::TokenUsage::default(),
            tool_calls: vec![crate::ToolCall {
                id: None,
                name: "lookup".to_owned(),
                arguments: json!({"n":"1"}),
            }],
            output_repairs: Vec::new(),
            quota_observations: Vec::new(),
            raw: Value::Null,
        };
        let validated = validate_inference_response(&request, response)?;
        assert_eq!(validated.tool_calls[0].arguments, json!({"n":1}));
        assert_eq!(validated.output_repairs.len(), 1);
        assert_eq!(
            validated.output_repairs[0].instance_path,
            "/tool_calls/0/arguments/n"
        );
        Ok(())
    }

    #[test]
    fn refuses_wrong_exact_tool_choice() {
        let mut request = crate::InferenceRequest::new("x/model", "lookup", 8);
        for name in ["lookup", "search"] {
            request.tools.push(crate::FunctionTool {
                name: name.to_owned(),
                description: None,
                parameters: json!({"type":"object"}),
                strict: false,
            });
        }
        request.tool_choice = Some(crate::ToolChoice::Function {
            name: "lookup".to_owned(),
        });
        let response = crate::InferenceResponse {
            content: String::new(),
            model: None,
            provider: None,
            usage_cost_usd: None,
            usage: crate::TokenUsage::default(),
            tool_calls: vec![crate::ToolCall {
                id: None,
                name: "search".to_owned(),
                arguments: json!({}),
            }],
            output_repairs: Vec::new(),
            quota_observations: Vec::new(),
            raw: Value::Null,
        };
        assert!(validate_inference_response(&request, response).is_err());
    }

    #[test]
    fn canonical_lowering_preserves_simple_semantics() {
        let schema = json!({"oneOf":[{"const":"a"},{"const":"b"}]});
        let lowered = lower_json_schema(&schema);
        assert_eq!(lowered, json!({"enum":["a","b"]}));
        assert!(jsonschema::options().offline().build(&lowered).is_ok());
        for value in [json!("a"), json!("c")] {
            let original_valid = jsonschema::options()
                .offline()
                .build(&schema)
                .is_ok_and(|validator| validator.is_valid(&value));
            let lowered_valid = jsonschema::options()
                .offline()
                .build(&lowered)
                .is_ok_and(|validator| validator.is_valid(&value));
            assert_eq!(original_valid, lowered_valid);
        }
    }

    #[test]
    fn strict_remote_refs_are_not_fetched_implicitly() {
        let schema = json!({"$ref":"https://example.invalid/schema.json"});
        assert!(validate_schema_offline(&schema).is_err());
    }

    #[test]
    fn canonical_lowering_preserves_model_guidance_verbatim() {
        let schema = json!({
            "oneOf":[
                {"const":"a","description":"Choose A for the fast path"},
                {"const":"b"}
            ]
        });
        assert_eq!(lower_json_schema(&schema), schema);
    }

    #[test]
    fn canonical_lowering_does_not_introduce_new_feature_classes() {
        let schema = json!({
            "type":"object",
            "properties":{"n":{"type":"integer"}},
            "required":["n"],
            "additionalProperties":false
        });
        assert_eq!(lower_json_schema(&schema), schema);
    }
}
