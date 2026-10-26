use std::collections::{BTreeMap, BTreeSet};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::model::STRUCTURED_TOOL_TRANSPORT_INSTRUCTION;
use crate::{
    AgenticPolicy, ConstraintEnforcement, Endpoint, EndpointCapabilities, FunctionTool,
    InferenceProvider, InferenceRequest, InferenceResponse, ProviderError, ReasoningEffort,
    RequestRequirements, ResponseSchema, RetryPolicy, RouteCandidate, RoutePlan, SchemaFeature,
    ToolChoice, approximate_prompt_tokens,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConformanceOutcome {
    Passed,
    Failed,
    Inconclusive,
}

/// Which schema-bearing API surface a dialect feature probe exercises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchemaFeatureTransport {
    StructuredOutputs,
    Tools,
}

impl SchemaFeatureTransport {
    #[must_use]
    pub const fn parameter(self) -> &'static str {
        match self {
            Self::StructuredOutputs => "structured_outputs",
            Self::Tools => "tools",
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StructuredOutputs => "structured_outputs",
            Self::Tools => "tools",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConformanceProbe {
    BasicChat,
    SystemRole,
    JsonObject,
    JsonSchemaLoose,
    JsonSchemaLooseStress,
    StrictJsonSchema,
    StrictJsonSchemaStress,
    StrictJsonSchemaStressPrompted,
    ToolSchemaLoose,
    ToolSchemaLooseStress,
    ToolSchemaStrict,
    ToolSchemaStrictStress,
    ToolSchemaStrictStressPrompted,
    SchemaFeature {
        transport: SchemaFeatureTransport,
        feature: SchemaFeature,
    },
    ReasoningOff,
    ReasoningMinimal,
    ReasoningLow,
    ReasoningMedium,
    ReasoningHigh,
    ReasoningXhigh,
    ReasoningMax,
    ReasoningBudget256,
}

const CONSTRAINT_STRESS_NONCE: &str = "7e57b4a19c2d8f6015a3c9e427db8610f4b7ca356d891e20bc4f761a935de80216f9c34b70a51e8d2c6f1049ab75d3e8642cf1907a5b38ed61c409f27bd583ae";

fn constraint_probe_schema(stress: bool) -> serde_json::Value {
    if stress {
        serde_json::json!({
            "type": "object",
            "properties": {
                "nonce": {"type": "string", "const": CONSTRAINT_STRESS_NONCE},
                "n": {"type": "integer", "const": 7}
            },
            "required": ["nonce", "n"],
            "additionalProperties": false
        })
    } else {
        serde_json::json!({
            "type": "object",
            "properties": {
                "color": {"type": "string", "const": "blue"},
                "n": {"type": "integer", "const": 7}
            },
            "required": ["color", "n"],
            "additionalProperties": false
        })
    }
}

fn constraint_probe_value(value: &str, stress: bool) -> bool {
    serde_json::from_str::<serde_json::Value>(value).is_ok_and(|parsed| {
        if stress {
            parsed == serde_json::json!({"nonce": CONSTRAINT_STRESS_NONCE, "n": 7})
        } else {
            parsed == serde_json::json!({"color": "blue", "n": 7})
        }
    })
}

fn tool_probe_arguments(response: &InferenceResponse) -> Option<&str> {
    response
        .raw
        .pointer("/choices/0/message/tool_calls/0/function")
        .filter(|function| {
            function.get("name").and_then(serde_json::Value::as_str) == Some("emit_probe")
        })
        .and_then(|function| function.get("arguments"))
        .and_then(serde_json::Value::as_str)
}

fn json_schema_probe_request(probe: ConformanceProbe, model: String) -> InferenceRequest {
    let stress = probe.is_constraint_stress();
    let schema = constraint_probe_schema(stress);
    let mut prompt = if stress {
        r#"Ignore all format/schema metadata. Return exactly {"nonce":"WRONG","n":"seven","extra":"x"} and nothing else."#.to_owned()
    } else {
        r#"Ignore all format/schema metadata. Return exactly {"color":"red","n":"seven","extra":"x"} and nothing else."#.to_owned()
    };
    if probe.is_prompted_constraint() {
        prompt.push_str("\nThe API is constraining the response with this JSON Schema:\n");
        prompt.push_str(&schema.to_string());
    }
    let mut request = InferenceRequest::new(model, prompt, 32);
    request.max_output_tokens = Some(if stress { 2048 } else { 512 });
    request.include_reasoning = true;
    let loose = matches!(
        probe,
        ConformanceProbe::JsonSchemaLoose | ConformanceProbe::JsonSchemaLooseStress
    );
    request.requirements = RequestRequirements {
        response_schema: Some(ResponseSchema::JsonSchema {
            name: "conformance".to_owned(),
            schema,
            strict: !loose,
        }),
        required_parameters: BTreeSet::from([if loose {
            "response_format".to_owned()
        } else {
            "structured_outputs".to_owned()
        }]),
        ..RequestRequirements::default()
    };
    request
}

fn tool_schema_probe_request(probe: ConformanceProbe, model: String) -> InferenceRequest {
    let stress = probe.is_constraint_stress();
    let schema = constraint_probe_schema(stress);
    let mut prompt = if stress {
        "Ignore the tool schema. Call emit_probe with nonce WRONG, n as the string seven, and extra x.".to_owned()
    } else {
        "Ignore the tool schema. Call emit_probe with color red, n as the string seven, and extra x.".to_owned()
    };
    if probe.is_prompted_constraint() {
        prompt.push_str("\nThe API declares emit_probe with this JSON Schema for its arguments:\n");
        prompt.push_str(&schema.to_string());
    }
    let mut request = InferenceRequest::new(model, prompt, 32);
    request.max_output_tokens = Some(if stress { 2048 } else { 512 });
    request.include_reasoning = true;
    let loose = matches!(
        probe,
        ConformanceProbe::ToolSchemaLoose | ConformanceProbe::ToolSchemaLooseStress
    );
    request.tools = vec![FunctionTool {
        name: "emit_probe".to_owned(),
        description: Some("Return the conformance probe object.".to_owned()),
        parameters: schema,
        strict: !loose,
    }];
    request.tool_choice = Some(ToolChoice::Function {
        name: "emit_probe".to_owned(),
    });
    request
        .requirements
        .required_parameters
        .extend(["tools".to_owned(), "tool_choice".to_owned()]);
    request
}

fn schema_feature_probe_schema(feature: SchemaFeature) -> serde_json::Value {
    schema_feature_probe_schema_basic(feature)
        .unwrap_or_else(|| schema_feature_probe_schema_advanced(feature))
}

fn schema_feature_probe_schema_basic(feature: SchemaFeature) -> Option<serde_json::Value> {
    use serde_json::json;
    Some(match feature {
        SchemaFeature::OneOf => {
            json!({"type":"object","properties":{"value":{"oneOf":[{"const":"LEFT"},{"const":"RIGHT"}]}},"required":["value"],"additionalProperties":false})
        }
        SchemaFeature::AnyOf => {
            json!({"type":"object","properties":{"value":{"anyOf":[{"const":"LEFT"},{"const":"RIGHT"}]}},"required":["value"],"additionalProperties":false})
        }
        SchemaFeature::AllOf => {
            json!({"type":"object","properties":{"value":{"allOf":[{"type":"string"},{"const":"ALL_OK"}]}},"required":["value"],"additionalProperties":false})
        }
        SchemaFeature::Enum => {
            json!({"type":"object","properties":{"value":{"type":"string","enum":["RED","GREEN"]}},"required":["value"],"additionalProperties":false})
        }
        SchemaFeature::NestedObject => {
            json!({"type":"object","properties":{"outer":{"type":"object","properties":{"token":{"const":"NESTED_OK"}},"required":["token"],"additionalProperties":false}},"required":["outer"],"additionalProperties":false})
        }
        SchemaFeature::ArrayItems => {
            json!({"type":"object","properties":{"values":{"type":"array","items":{"const":7}}},"required":["values"],"additionalProperties":false})
        }
        SchemaFeature::ArrayBounds => {
            json!({"type":"object","properties":{"values":{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":2}},"required":["values"],"additionalProperties":false})
        }
        SchemaFeature::TuplePrefixItems => {
            json!({"type":"object","properties":{"values":{"type":"array","prefixItems":[{"const":"A"},{"const":"B"}],"items":false,"minItems":2,"maxItems":2}},"required":["values"],"additionalProperties":false})
        }
        SchemaFeature::NumericBounds => {
            json!({"type":"object","properties":{"value":{"type":"integer","minimum":10,"maximum":12}},"required":["value"],"additionalProperties":false})
        }
        SchemaFeature::MultipleOf => {
            json!({"type":"object","properties":{"value":{"type":"integer","multipleOf":7,"minimum":7}},"required":["value"],"additionalProperties":false})
        }
        SchemaFeature::StringPattern => {
            json!({"type":"object","properties":{"value":{"type":"string","pattern":"^SCHEMA_[A-Z]{4}$"}},"required":["value"],"additionalProperties":false})
        }
        SchemaFeature::StringBounds => {
            json!({"type":"object","properties":{"value":{"type":"string","minLength":6,"maxLength":6}},"required":["value"],"additionalProperties":false})
        }
        SchemaFeature::NullableType => {
            json!({"type":"object","properties":{"value":{"type":["string","null"]}},"required":["value"],"additionalProperties":false})
        }
        SchemaFeature::DefsRef => {
            json!({"$defs":{"answer":{"type":"string","const":"REF_OK"}},"type":"object","properties":{"value":{"$ref":"#/$defs/answer"}},"required":["value"],"additionalProperties":false})
        }
        SchemaFeature::BooleanType => {
            json!({"type":"object","properties":{"value":{"type":"boolean","const":true}},"required":["value"],"additionalProperties":false})
        }
        SchemaFeature::NumberType => {
            json!({"type":"object","properties":{"value":{"type":"number","const":1.5}},"required":["value"],"additionalProperties":false})
        }
        SchemaFeature::NullType => {
            json!({"type":"object","properties":{"value":{"type":"null"}},"required":["value"],"additionalProperties":false})
        }
        SchemaFeature::ArrayType => {
            json!({"type":"object","properties":{"value":{"type":"array"}},"required":["value"],"additionalProperties":false})
        }
        SchemaFeature::TypeUnion => {
            json!({"type":"object","properties":{"value":{"type":["string","integer"]}},"required":["value"],"additionalProperties":false})
        }
        SchemaFeature::RootArray => {
            json!({"type":"array","items":{"const":7},"minItems":1,"maxItems":1})
        }
        SchemaFeature::RootPrimitive => json!({"type":"string","const":"ROOT_OK"}),
        _ => return None,
    })
}

fn schema_feature_probe_schema_advanced(feature: SchemaFeature) -> serde_json::Value {
    use serde_json::json;
    match feature {
        SchemaFeature::RecursiveRef => {
            json!({"$defs":{"node":{"oneOf":[{"type":"object","properties":{"kind":{"const":"leaf"}},"required":["kind"],"additionalProperties":false},{"type":"object","properties":{"kind":{"const":"branch"},"child":{"$ref":"#/$defs/node"}},"required":["kind","child"],"additionalProperties":false}]}},"type":"object","properties":{"root":{"type":"object","properties":{"kind":{"const":"branch"},"child":{"$ref":"#/$defs/node"}},"required":["kind","child"],"additionalProperties":false}},"required":["root"],"additionalProperties":false})
        }
        SchemaFeature::Contains => {
            json!({"type":"object","properties":{"values":{"type":"array","items":{"type":"integer"},"contains":{"const":7},"minContains":1}},"required":["values"],"additionalProperties":false})
        }
        SchemaFeature::UniqueItems => {
            json!({"type":"object","properties":{"values":{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":2,"uniqueItems":true}},"required":["values"],"additionalProperties":false})
        }
        SchemaFeature::Not => {
            json!({"type":"object","properties":{"value":{"type":"string","not":{"const":"WRONG"}}},"required":["value"],"additionalProperties":false})
        }
        SchemaFeature::Conditional => {
            json!({"type":"object","properties":{"mode":{"const":"A"},"value":{"type":"integer"}},"required":["mode","value"],"additionalProperties":false,"if":{"properties":{"mode":{"const":"A"}}},"then":{"properties":{"value":{"const":1}}},"else":{"properties":{"value":{"const":2}}}})
        }
        SchemaFeature::DependentRequired => {
            json!({"type":"object","properties":{"trigger":{"type":"string"},"dependent":{"type":"string"}},"required":["trigger"],"dependentRequired":{"trigger":["dependent"]},"additionalProperties":false})
        }
        SchemaFeature::PropertyNames => {
            json!({"type":"object","properties":{"map":{"type":"object","propertyNames":{"pattern":"^x_[a-z]+$"},"additionalProperties":{"type":"integer"},"minProperties":1}},"required":["map"],"additionalProperties":false})
        }
        SchemaFeature::ObjectBounds => {
            json!({"type":"object","properties":{"map":{"type":"object","additionalProperties":{"type":"integer"},"minProperties":2,"maxProperties":2}},"required":["map"],"additionalProperties":false})
        }
        SchemaFeature::AdditionalPropertiesSchema => {
            json!({"type":"object","properties":{"map":{"type":"object","additionalProperties":{"const":7},"minProperties":1}},"required":["map"],"additionalProperties":false})
        }
        SchemaFeature::PatternProperties => {
            json!({"type":"object","properties":{"map":{"type":"object","patternProperties":{"^x_[a-z]+$":{"const":7}},"additionalProperties":false,"minProperties":1}},"required":["map"],"additionalProperties":false})
        }
        SchemaFeature::DependentSchemas => {
            json!({"type":"object","properties":{"trigger":{"const":"yes"},"dependent":{"type":"string"}},"required":["trigger"],"additionalProperties":false,"dependentSchemas":{"trigger":{"required":["dependent"],"properties":{"dependent":{"const":"OK"}}}}})
        }
        SchemaFeature::UnevaluatedProperties => {
            json!({"type":"object","properties":{"value":{"const":7}},"required":["value"],"unevaluatedProperties":false})
        }
        SchemaFeature::Format => {
            json!({"type":"object","properties":{"value":{"type":"string","format":"uuid"}},"required":["value"],"additionalProperties":false})
        }
        SchemaFeature::OptionalProperties => {
            json!({"type":"object","properties":{"must":{"const":"REQUIRED"},"optional":{"const":"OPTIONAL"}},"required":["must"],"additionalProperties":false})
        }
        SchemaFeature::OpenObject => {
            json!({"type":"object","properties":{"known":{"const":"KNOWN"}},"required":["known"],"additionalProperties":true})
        }
        SchemaFeature::RootComposition => {
            json!({"oneOf":[
                {"type":"object","properties":{"kind":{"const":"left"}},"required":["kind"],"additionalProperties":false},
                {"type":"object","properties":{"kind":{"const":"right"}},"required":["kind"],"additionalProperties":false}
            ]})
        }
        SchemaFeature::RootRef => {
            json!({"$defs":{"answer":{"type":"object","properties":{"value":{"const":"ROOT_REF_OK"}},"required":["value"],"additionalProperties":false}},"$ref":"#/$defs/answer"})
        }
        SchemaFeature::UnevaluatedItems => {
            json!({"type":"object","properties":{"values":{"type":"array","prefixItems":[{"const":1}],"unevaluatedItems":false,"minItems":1,"maxItems":1}},"required":["values"],"additionalProperties":false})
        }
        SchemaFeature::LegacyTupleItems => {
            json!({"type":"object","properties":{"values":{"type":"array","items":[{"const":"A"},{"const":"B"}],"additionalItems":false,"minItems":2,"maxItems":2}},"required":["values"],"additionalProperties":false})
        }
        SchemaFeature::ToolTurnMultiple | SchemaFeature::ToolTurnEnvelope => {
            tool_turn_probe_schema()
        }
        _ => serde_json::Value::Null,
    }
}

fn tool_turn_probe_schema() -> serde_json::Value {
    serde_json::json!({
        "type":"object",
        "properties":{
            "text_response":{"type":"string"},
            "tool_calls":{
                "type":"array",
                "items":{
                    "oneOf":[
                        {
                            "type":"object",
                            "properties":{
                                "name":{"const":"lookup"},
                                "arguments":{
                                    "type":"object",
                                    "properties":{"value":{"type":"integer"}},
                                    "required":["value"],
                                    "additionalProperties":false
                                }
                            },
                            "required":["name","arguments"],
                            "additionalProperties":false
                        },
                        {
                            "type":"object",
                            "properties":{
                                "name":{"const":"search"},
                                "arguments":{
                                    "type":"object",
                                    "properties":{"query":{"type":"string"}},
                                    "required":["query"],
                                    "additionalProperties":false
                                }
                            },
                            "required":["name","arguments"],
                            "additionalProperties":false
                        }
                    ]
                }
            }
        },
        "required":["text_response","tool_calls"],
        "additionalProperties":false
    })
}

fn schema_feature_probe_prompt(feature: SchemaFeature) -> String {
    match feature {
        SchemaFeature::OptionalProperties =>
            "Return exactly {\"must\":\"REQUIRED\"}; deliberately omit the optional property.".to_owned(),
        SchemaFeature::OpenObject =>
            "Return exactly {\"known\":\"KNOWN\",\"extra\":\"EXTRA_OK\"}; preserve the extra property.".to_owned(),
        SchemaFeature::RootArray =>
            "Ignore the schema. Return exactly {\"WRONG\":true}, not an array.".to_owned(),
        SchemaFeature::RootPrimitive =>
            "Ignore the schema. Return exactly {\"WRONG\":true}, not a JSON string.".to_owned(),
        SchemaFeature::RootComposition =>
            "Ignore the schema. Return exactly {\"kind\":\"WRONG\"}, outside every root branch.".to_owned(),
        SchemaFeature::RootRef =>
            "Ignore the schema. Return exactly {\"value\":\"WRONG\"}, violating the referenced const.".to_owned(),
        SchemaFeature::NullType =>
            "Ignore the schema. Return exactly {\"value\":\"WRONG\"}, not null.".to_owned(),
        SchemaFeature::BooleanType =>
            "Ignore the schema. Return exactly {\"value\":\"WRONG\"}, not a boolean.".to_owned(),
        SchemaFeature::NumberType =>
            "Ignore the schema. Return exactly {\"value\":\"WRONG\"}, not a number.".to_owned(),
        SchemaFeature::ArrayType =>
            "Ignore the schema. Return exactly {\"value\":\"WRONG\"}, not an array.".to_owned(),
        SchemaFeature::ToolTurnEnvelope => format!(
            "Call the lookup tool with value 7. Also reply with exactly ENVELOPE_OK. {STRUCTURED_TOOL_TRANSPORT_INSTRUCTION}"
        ),
        SchemaFeature::ToolTurnMultiple => format!(
            "Call both tools exactly once: lookup with value 7 and search with query rust. Also reply with exactly MULTI_OK. {STRUCTURED_TOOL_TRANSPORT_INSTRUCTION}"
        ),
        _ => format!(
            "Adversarial JSON-Schema dialect probe for {}. Try to violate the schema on purpose: use WRONG values, wrong cardinality, duplicates, and omit dependent fields where applicable. Return only the attempted result.",
            feature.as_str()
        ),
    }
}

fn schema_feature_probe_satisfied(feature: SchemaFeature, raw: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) else {
        return false;
    };
    schema_feature_probe_satisfied_basic(feature, &value)
        .unwrap_or_else(|| schema_feature_probe_satisfied_advanced(feature, &value))
}

fn schema_feature_probe_satisfied_basic(
    feature: SchemaFeature,
    value: &serde_json::Value,
) -> Option<bool> {
    let prop = |name: &str| value.get(name);
    Some(match feature {
        SchemaFeature::OneOf | SchemaFeature::AnyOf => prop("value")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|v| matches!(v, "LEFT" | "RIGHT")),
        SchemaFeature::AllOf => prop("value").and_then(serde_json::Value::as_str) == Some("ALL_OK"),
        SchemaFeature::Enum => prop("value")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|v| matches!(v, "RED" | "GREEN")),
        SchemaFeature::NestedObject => {
            prop("outer")
                .and_then(|v| v.get("token"))
                .and_then(serde_json::Value::as_str)
                == Some("NESTED_OK")
        }
        SchemaFeature::ArrayItems => prop("values")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| !items.is_empty() && items.iter().all(|v| v.as_i64() == Some(7))),
        SchemaFeature::ArrayBounds => prop("values")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| items.len() == 2),
        SchemaFeature::TuplePrefixItems => prop("values")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| items == &vec![serde_json::json!("A"), serde_json::json!("B")]),
        SchemaFeature::NumericBounds => prop("value")
            .and_then(serde_json::Value::as_i64)
            .is_some_and(|v| (10..=12).contains(&v)),
        SchemaFeature::MultipleOf => prop("value")
            .and_then(serde_json::Value::as_i64)
            .is_some_and(|v| v >= 7 && v % 7 == 0),
        SchemaFeature::StringPattern => prop("value")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|v| {
                v.len() == 11
                    && v.starts_with("SCHEMA_")
                    && v[7..].chars().all(|c| c.is_ascii_uppercase())
            }),
        SchemaFeature::StringBounds => prop("value")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|v| v.chars().count() == 6),
        SchemaFeature::NullableType => prop("value").is_some_and(|v| v.is_null() || v.is_string()),
        SchemaFeature::DefsRef => {
            prop("value").and_then(serde_json::Value::as_str) == Some("REF_OK")
        }
        SchemaFeature::RecursiveRef => prop("root").is_some_and(|root| {
            root.get("kind").and_then(serde_json::Value::as_str) == Some("branch")
                && root.get("child").is_some_and(valid_recursive_probe_node)
        }),
        SchemaFeature::BooleanType => {
            prop("value").and_then(serde_json::Value::as_bool) == Some(true)
        }
        SchemaFeature::NumberType => prop("value").and_then(serde_json::Value::as_f64) == Some(1.5),
        SchemaFeature::NullType => prop("value").is_some_and(serde_json::Value::is_null),
        SchemaFeature::ArrayType => prop("value").is_some_and(serde_json::Value::is_array),
        SchemaFeature::TypeUnion => {
            prop("value").is_some_and(|value| value.is_string() || value.is_i64())
        }
        SchemaFeature::RootArray => value
            .as_array()
            .is_some_and(|items| items == &vec![serde_json::json!(7)]),
        SchemaFeature::RootPrimitive => value.as_str() == Some("ROOT_OK"),
        _ => return None,
    })
}

fn schema_feature_probe_satisfied_advanced(
    feature: SchemaFeature,
    value: &serde_json::Value,
) -> bool {
    let prop = |name: &str| value.get(name);
    match feature {
        SchemaFeature::Contains => prop("values")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| items.iter().any(|v| v.as_i64() == Some(7))),
        SchemaFeature::UniqueItems => prop("values")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| items.len() == 2 && items[0] != items[1]),
        SchemaFeature::Not => prop("value")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|v| v != "WRONG"),
        SchemaFeature::Conditional => {
            prop("mode").and_then(serde_json::Value::as_str) == Some("A")
                && prop("value").and_then(serde_json::Value::as_i64) == Some(1)
        }
        SchemaFeature::DependentRequired => {
            prop("trigger").is_some() && prop("dependent").is_some()
        }
        SchemaFeature::PropertyNames => prop("map")
            .and_then(serde_json::Value::as_object)
            .is_some_and(|map| {
                !map.is_empty()
                    && map.keys().all(|key| {
                        key.starts_with("x_")
                            && key.len() > 2
                            && key[2..].chars().all(|c| c.is_ascii_lowercase())
                    })
            }),
        SchemaFeature::ObjectBounds => prop("map")
            .and_then(serde_json::Value::as_object)
            .is_some_and(|map| map.len() == 2),
        SchemaFeature::AdditionalPropertiesSchema => prop("map")
            .and_then(serde_json::Value::as_object)
            .is_some_and(|map| !map.is_empty() && map.values().all(|v| v.as_i64() == Some(7))),
        SchemaFeature::PatternProperties => prop("map")
            .and_then(serde_json::Value::as_object)
            .is_some_and(|map| {
                !map.is_empty()
                    && map.iter().all(|(key, v)| {
                        key.starts_with("x_") && key.len() > 2 && v.as_i64() == Some(7)
                    })
            }),
        SchemaFeature::DependentSchemas => {
            prop("trigger").and_then(serde_json::Value::as_str) == Some("yes")
                && prop("dependent").and_then(serde_json::Value::as_str) == Some("OK")
        }
        SchemaFeature::UnevaluatedProperties => value.as_object().is_some_and(|map| {
            map.len() == 1 && map.get("value").and_then(serde_json::Value::as_i64) == Some(7)
        }),
        SchemaFeature::Format => prop("value")
            .and_then(serde_json::Value::as_str)
            .is_some_and(is_uuid_like),
        SchemaFeature::OptionalProperties => value.as_object().is_some_and(|map| {
            map.len() == 1
                && map.get("must").and_then(serde_json::Value::as_str) == Some("REQUIRED")
        }),
        SchemaFeature::OpenObject => value.as_object().is_some_and(|map| {
            map.get("known").and_then(serde_json::Value::as_str) == Some("KNOWN")
                && map.get("extra").and_then(serde_json::Value::as_str) == Some("EXTRA_OK")
        }),
        SchemaFeature::RootComposition => prop("kind")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|kind| matches!(kind, "left" | "right")),
        SchemaFeature::RootRef => {
            prop("value").and_then(serde_json::Value::as_str) == Some("ROOT_REF_OK")
        }
        SchemaFeature::UnevaluatedItems => prop("values")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| items == &vec![serde_json::json!(1)]),
        SchemaFeature::LegacyTupleItems => prop("values")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| items == &vec![serde_json::json!("A"), serde_json::json!("B")]),
        SchemaFeature::ToolTurnEnvelope => tool_turn_probe_satisfied(value, false),
        SchemaFeature::ToolTurnMultiple => tool_turn_probe_satisfied(value, true),
        _ => false,
    }
}

fn tool_turn_probe_satisfied(value: &serde_json::Value, multiple: bool) -> bool {
    let expected_text = if multiple { "MULTI_OK" } else { "ENVELOPE_OK" };
    if value
        .get("text_response")
        .and_then(serde_json::Value::as_str)
        != Some(expected_text)
    {
        return false;
    }
    let Some(calls) = value
        .get("tool_calls")
        .and_then(serde_json::Value::as_array)
    else {
        return false;
    };
    let lookup = calls.iter().any(|call| {
        call.get("name").and_then(serde_json::Value::as_str) == Some("lookup")
            && call
                .pointer("/arguments/value")
                .and_then(serde_json::Value::as_i64)
                == Some(7)
    });
    if !multiple {
        return calls.len() == 1 && lookup;
    }
    let search = calls.iter().any(|call| {
        call.get("name").and_then(serde_json::Value::as_str) == Some("search")
            && call
                .pointer("/arguments/query")
                .and_then(serde_json::Value::as_str)
                == Some("rust")
    });
    calls.len() == 2 && lookup && search
}

fn is_uuid_like(value: &str) -> bool {
    if value.len() != 36 {
        return false;
    }
    value.char_indices().all(|(index, character)| match index {
        8 | 13 | 18 | 23 => character == '-',
        _ => character.is_ascii_hexdigit(),
    })
}

fn valid_recursive_probe_node(node: &serde_json::Value) -> bool {
    match node.get("kind").and_then(serde_json::Value::as_str) {
        Some("leaf") => node.as_object().is_some_and(|map| map.len() == 1),
        Some("branch") => node.get("child").is_some_and(valid_recursive_probe_node),
        _ => false,
    }
}

fn schema_feature_probe_request(
    transport: SchemaFeatureTransport,
    feature: SchemaFeature,
    model: String,
) -> InferenceRequest {
    let schema = schema_feature_probe_schema(feature);
    let mut request = InferenceRequest::new(model, schema_feature_probe_prompt(feature), 64);
    request.max_output_tokens = Some(2048);
    request.include_reasoning = true;
    match transport {
        SchemaFeatureTransport::StructuredOutputs => {
            request.requirements = RequestRequirements {
                response_schema: Some(ResponseSchema::JsonSchema {
                    name: format!("schema_feature_{}", feature.as_str()),
                    schema,
                    strict: true,
                }),
                required_parameters: BTreeSet::from(["structured_outputs".to_owned()]),
                ..RequestRequirements::default()
            };
        }
        SchemaFeatureTransport::Tools => {
            request.tools = vec![FunctionTool {
                name: "emit_probe".to_owned(),
                description: Some("Return the JSON-Schema dialect probe value.".to_owned()),
                parameters: schema,
                strict: true,
            }];
            request.tool_choice = Some(ToolChoice::Function {
                name: "emit_probe".to_owned(),
            });
            request
                .requirements
                .required_parameters
                .extend(["tools".to_owned(), "tool_choice".to_owned()]);
        }
    }
    request
}

fn configure_probe_request(endpoint: &Endpoint, request: &mut InferenceRequest) {
    if !endpoint
        .capabilities
        .adapter_parameters
        .contains("include_reasoning")
    {
        request.include_reasoning = false;
    }
    if endpoint.workload_policy.agentic == AgenticPolicy::Required {
        request.agentic = true;
    }
    if endpoint.workload_policy.requires_stable_session {
        request.session_id = Some("inference-redox-conformance".to_owned());
    }
}

fn probe_route_plan(endpoint: Endpoint, request: &InferenceRequest) -> RoutePlan {
    let prompt_tokens = request.prompt_tokens.unwrap_or_else(|| {
        request
            .system_prompt
            .as_deref()
            .map_or(0, approximate_prompt_tokens)
            .saturating_add(approximate_prompt_tokens(&request.prompt))
    });
    let expected_cost_usd = endpoint
        .pricing
        .expected_cost(prompt_tokens, request.expected_output_tokens);
    RoutePlan {
        prompt_tokens,
        expected_output_tokens: request.expected_output_tokens,
        cheapest_cost_usd: expected_cost_usd,
        cost_ceiling_usd: expected_cost_usd,
        candidates: vec![RouteCandidate {
            endpoint,
            expected_cost_usd,
            expected_response_p75_seconds: None,
            expected_response_p95_seconds: None,
        }],
    }
}

impl ConformanceProbe {
    #[must_use]
    pub fn test_id(self) -> String {
        match self {
            Self::BasicChat => "basic-chat".to_owned(),
            Self::SystemRole => "system-role".to_owned(),
            Self::JsonObject => "json-object".to_owned(),
            Self::JsonSchemaLoose => "json-schema-loose".to_owned(),
            Self::JsonSchemaLooseStress => "json-schema-loose-stress".to_owned(),
            Self::StrictJsonSchema => "strict-json-schema".to_owned(),
            Self::StrictJsonSchemaStress => "strict-json-schema-stress".to_owned(),
            Self::StrictJsonSchemaStressPrompted => "strict-json-schema-stress-prompted".to_owned(),
            Self::ToolSchemaLoose => "tool-schema-loose".to_owned(),
            Self::ToolSchemaLooseStress => "tool-schema-loose-stress".to_owned(),
            Self::ToolSchemaStrict => "tool-schema-strict".to_owned(),
            Self::ToolSchemaStrictStress => "tool-schema-strict-stress".to_owned(),
            Self::ToolSchemaStrictStressPrompted => "tool-schema-strict-stress-prompted".to_owned(),
            Self::SchemaFeature { transport, feature } => {
                format!("schema-feature-{}-{}", transport.as_str(), feature.as_str())
            }
            Self::ReasoningOff => "reasoning-off".to_owned(),
            Self::ReasoningMinimal => "reasoning-minimal".to_owned(),
            Self::ReasoningLow => "reasoning-low".to_owned(),
            Self::ReasoningMedium => "reasoning-medium".to_owned(),
            Self::ReasoningHigh => "reasoning-high".to_owned(),
            Self::ReasoningXhigh => "reasoning-xhigh".to_owned(),
            Self::ReasoningMax => "reasoning-max".to_owned(),
            Self::ReasoningBudget256 => "reasoning-budget-256".to_owned(),
        }
    }

    #[must_use]
    pub const fn test_version(self) -> u32 {
        match self {
            Self::ReasoningOff
            | Self::ReasoningMinimal
            | Self::ReasoningLow
            | Self::ReasoningMedium
            | Self::ReasoningHigh
            | Self::ReasoningXhigh
            | Self::ReasoningMax => 6,
            Self::SchemaFeature {
                feature:
                    SchemaFeature::ToolTurnEnvelope
                    | SchemaFeature::ToolTurnMultiple
                    | SchemaFeature::RootArray
                    | SchemaFeature::RootPrimitive
                    | SchemaFeature::RootComposition
                    | SchemaFeature::RootRef
                    | SchemaFeature::NullType
                    | SchemaFeature::BooleanType
                    | SchemaFeature::NumberType
                    | SchemaFeature::ArrayType,
                ..
            }
            | Self::JsonSchemaLoose
            | Self::StrictJsonSchema
            | Self::ToolSchemaLoose
            | Self::ToolSchemaStrict => 2,
            Self::SchemaFeature { .. }
            | Self::ReasoningBudget256
            | Self::BasicChat
            | Self::SystemRole
            | Self::JsonObject
            | Self::JsonSchemaLooseStress
            | Self::StrictJsonSchemaStress
            | Self::StrictJsonSchemaStressPrompted
            | Self::ToolSchemaLooseStress
            | Self::ToolSchemaStrictStress
            | Self::ToolSchemaStrictStressPrompted => 1,
        }
    }

    #[must_use]
    pub const fn parameter(self) -> &'static str {
        match self {
            Self::BasicChat => "chat_completions",
            Self::SystemRole => "system_role",
            Self::JsonObject | Self::JsonSchemaLoose | Self::JsonSchemaLooseStress => {
                "response_format"
            }
            Self::StrictJsonSchema
            | Self::StrictJsonSchemaStress
            | Self::StrictJsonSchemaStressPrompted => "structured_outputs",
            Self::ToolSchemaLoose
            | Self::ToolSchemaLooseStress
            | Self::ToolSchemaStrict
            | Self::ToolSchemaStrictStress
            | Self::ToolSchemaStrictStressPrompted => "tools",
            Self::SchemaFeature { transport, .. } => transport.parameter(),
            Self::ReasoningOff
            | Self::ReasoningMinimal
            | Self::ReasoningLow
            | Self::ReasoningMedium
            | Self::ReasoningHigh
            | Self::ReasoningXhigh
            | Self::ReasoningMax => "reasoning",
            Self::ReasoningBudget256 => "reasoning_max_tokens",
        }
    }

    #[must_use]
    pub const fn reasoning_effort(self) -> Option<ReasoningEffort> {
        match self {
            Self::ReasoningOff => Some(ReasoningEffort::None),
            Self::ReasoningMinimal => Some(ReasoningEffort::Minimal),
            Self::ReasoningLow => Some(ReasoningEffort::Low),
            Self::ReasoningMedium => Some(ReasoningEffort::Medium),
            Self::ReasoningHigh => Some(ReasoningEffort::High),
            Self::ReasoningXhigh => Some(ReasoningEffort::Xhigh),
            Self::ReasoningMax => Some(ReasoningEffort::Max),
            Self::BasicChat
            | Self::SystemRole
            | Self::JsonObject
            | Self::JsonSchemaLoose
            | Self::JsonSchemaLooseStress
            | Self::StrictJsonSchema
            | Self::StrictJsonSchemaStress
            | Self::StrictJsonSchemaStressPrompted
            | Self::ToolSchemaLoose
            | Self::ToolSchemaLooseStress
            | Self::ToolSchemaStrict
            | Self::ToolSchemaStrictStress
            | Self::ToolSchemaStrictStressPrompted
            | Self::SchemaFeature { .. }
            | Self::ReasoningBudget256 => None,
        }
    }

    #[must_use]
    pub const fn is_reasoning_probe(self) -> bool {
        matches!(
            self,
            Self::ReasoningOff
                | Self::ReasoningMinimal
                | Self::ReasoningLow
                | Self::ReasoningMedium
                | Self::ReasoningHigh
                | Self::ReasoningXhigh
                | Self::ReasoningMax
                | Self::ReasoningBudget256
        )
    }

    #[must_use]
    pub const fn is_constraint_probe(self) -> bool {
        matches!(
            self,
            Self::JsonSchemaLoose
                | Self::JsonSchemaLooseStress
                | Self::StrictJsonSchema
                | Self::StrictJsonSchemaStress
                | Self::StrictJsonSchemaStressPrompted
                | Self::ToolSchemaLoose
                | Self::ToolSchemaLooseStress
                | Self::ToolSchemaStrict
                | Self::ToolSchemaStrictStress
                | Self::ToolSchemaStrictStressPrompted
                | Self::SchemaFeature { .. }
        )
    }

    #[must_use]
    pub const fn is_constraint_stress(self) -> bool {
        matches!(
            self,
            Self::JsonSchemaLooseStress
                | Self::StrictJsonSchemaStress
                | Self::StrictJsonSchemaStressPrompted
                | Self::ToolSchemaLooseStress
                | Self::ToolSchemaStrictStress
                | Self::ToolSchemaStrictStressPrompted
        )
    }

    #[must_use]
    pub const fn is_prompted_constraint(self) -> bool {
        matches!(
            self,
            Self::StrictJsonSchemaStressPrompted | Self::ToolSchemaStrictStressPrompted
        )
    }

    #[must_use]
    pub fn request(self, model: impl Into<String>) -> InferenceRequest {
        let model = model.into();
        match self {
            Self::BasicChat => {
                let mut request = InferenceRequest::new(model, "Reply with PROBE_OK.", 16);
                request.max_output_tokens = Some(256);
                request.temperature = Some(0.0);
                request
            }
            Self::SystemRole => {
                let mut request = InferenceRequest::new(model, "Reply briefly.", 16);
                request.system_prompt = Some("This is a protocol conformance probe.".to_owned());
                request.max_output_tokens = Some(256);
                request.temperature = Some(0.0);
                request
            }
            Self::JsonObject => {
                let mut request = InferenceRequest::new(
                    model,
                    "Return a JSON object whose only key is ok and whose value is true.",
                    16,
                );
                request.max_output_tokens = Some(256);
                request.temperature = Some(0.0);
                request.requirements = RequestRequirements {
                    response_schema: Some(ResponseSchema::JsonObject),
                    ..RequestRequirements::default()
                };
                request
            }
            Self::JsonSchemaLoose
            | Self::JsonSchemaLooseStress
            | Self::StrictJsonSchema
            | Self::StrictJsonSchemaStress
            | Self::StrictJsonSchemaStressPrompted => json_schema_probe_request(self, model),
            Self::ToolSchemaLoose
            | Self::ToolSchemaLooseStress
            | Self::ToolSchemaStrict
            | Self::ToolSchemaStrictStress
            | Self::ToolSchemaStrictStressPrompted => tool_schema_probe_request(self, model),
            Self::SchemaFeature { transport, feature } => {
                schema_feature_probe_request(transport, feature, model)
            }
            Self::ReasoningOff
            | Self::ReasoningMinimal
            | Self::ReasoningLow
            | Self::ReasoningMedium
            | Self::ReasoningHigh
            | Self::ReasoningXhigh
            | Self::ReasoningMax => {
                let mut request = InferenceRequest::new(
                    model,
                    "A,B,C,D,E,F are a permutation of 1..6. A+B=7; B+C=8; C+D=9; D+E=7; E+F=5; |A-F|=2. Find ABCDEF. Reply with the 6 digits only.",
                    16,
                );
                request.max_output_tokens = Some(2048);
                request.include_reasoning = true;
                request.requirements.reasoning =
                    self.reasoning_effort().unwrap_or(ReasoningEffort::None);
                request
                    .requirements
                    .required_parameters
                    .insert("reasoning".to_owned());
                request
            }
            Self::ReasoningBudget256 => {
                let mut request = InferenceRequest::new(
                    model,
                    "A,B,C,D,E,F are a permutation of 1..6. A+B=7; B+C=8; C+D=9; D+E=7; E+F=5; |A-F|=2. Find ABCDEF. Reply with the 6 digits only.",
                    16,
                );
                request.max_output_tokens = Some(2048);
                request.include_reasoning = true;
                request.requirements.reasoning_max_tokens = Some(256);
                request
            }
        }
    }

    /// Run one probe against a single provider/model scope. The supplied endpoint is used only
    /// as the exact route candidate; evidence is provider-scoped unless a future adapter can
    /// independently address a narrower endpoint.
    pub async fn run_provider_model<P: InferenceProvider + ?Sized>(
        self,
        provider: &P,
        endpoint: Endpoint,
        canonical_id: Option<String>,
    ) -> ConformanceObservation {
        let source = endpoint.source.clone();
        let credential = endpoint.credential.clone();
        let provider_key = endpoint.provider_key.clone();
        let endpoint_id = Some(endpoint.id.clone());
        let native_model_id = endpoint.model.clone();
        let mut request = self.request(native_model_id.clone());
        configure_probe_request(&endpoint, &mut request);
        let plan = probe_route_plan(endpoint, &request);
        let retry = RetryPolicy {
            max_attempts: 1,
            ..RetryPolicy::default()
        };
        let observed = match provider.execute_plan(&request, &plan, retry).await {
            Ok(response) => self.observe_response(&response),
            Err(error) => ProbeExecution::from_error(self, &error),
        };
        ConformanceObservation {
            source,
            credential,
            provider_key,
            endpoint_id,
            native_model_id,
            canonical_id,
            parameter: self.parameter().to_owned(),
            reasoning_effort: self.reasoning_effort(),
            reasoning_tokens: observed.reasoning_tokens,
            completion_tokens: observed.completion_tokens,
            reasoning_chars: observed.reasoning_chars,
            reasoning_trace: observed.reasoning_trace,
            response_content: observed.response_content,
            tool_arguments: observed.tool_arguments,
            puzzle_correct: observed.puzzle_correct,
            finish_reason: observed.finish_reason,
            constraint_satisfied: observed.constraint_satisfied,
            test_id: self.test_id(),
            test_version: self.test_version(),
            outcome: observed.outcome,
            observed_unix_ms: unix_time_ms(),
            detail: observed.detail,
        }
    }

    fn observe_response(self, response: &InferenceResponse) -> ProbeExecution {
        let reasoning_trace = response
            .raw
            .pointer("/choices/0/message/reasoning")
            .or_else(|| response.raw.pointer("/choices/0/message/reasoning_content"))
            .or_else(|| response.raw.pointer("/choices/0/message/reasoning_details"))
            .cloned();
        let reasoning_chars = reasoning_trace
            .as_ref()
            .map(serde_json::Value::to_string)
            .and_then(|value| u64::try_from(value.chars().count()).ok());
        let finish_reason = response
            .raw
            .pointer("/choices/0/finish_reason")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        let constraint_satisfied = self.observed_constraint(response);
        let (outcome, detail) = if self.is_constraint_probe()
            && finish_reason.as_deref() == Some("length")
        {
            (
                ConformanceOutcome::Inconclusive,
                Some(
                    "constraint probe hit the completion limit before a terminal result".to_owned(),
                ),
            )
        } else {
            match self.validate_response(response) {
                Ok(()) => (ConformanceOutcome::Passed, None),
                Err(detail) => (ConformanceOutcome::Failed, Some(detail)),
            }
        };
        ProbeExecution {
            reasoning_tokens: response.usage.reasoning_tokens,
            completion_tokens: response.usage.output_tokens,
            reasoning_chars,
            reasoning_trace,
            response_content: Some(response.content.clone()),
            tool_arguments: tool_probe_arguments(response).map(str::to_owned),
            puzzle_correct: self
                .is_reasoning_probe()
                .then(|| response.content.trim() == "253614"),
            finish_reason,
            constraint_satisfied,
            outcome,
            detail,
        }
    }

    fn observed_constraint(self, response: &InferenceResponse) -> Option<bool> {
        match self {
            Self::JsonSchemaLoose
            | Self::JsonSchemaLooseStress
            | Self::StrictJsonSchema
            | Self::StrictJsonSchemaStress
            | Self::StrictJsonSchemaStressPrompted => Some(constraint_probe_value(
                &response.content,
                self.is_constraint_stress(),
            )),
            Self::ToolSchemaLoose
            | Self::ToolSchemaLooseStress
            | Self::ToolSchemaStrict
            | Self::ToolSchemaStrictStress
            | Self::ToolSchemaStrictStressPrompted => {
                Some(tool_probe_arguments(response).is_some_and(|arguments| {
                    constraint_probe_value(arguments, self.is_constraint_stress())
                }))
            }
            Self::SchemaFeature { transport, feature } => Some(match transport {
                SchemaFeatureTransport::StructuredOutputs => {
                    schema_feature_probe_satisfied(feature, &response.content)
                }
                SchemaFeatureTransport::Tools => tool_probe_arguments(response)
                    .is_some_and(|arguments| schema_feature_probe_satisfied(feature, arguments)),
            }),
            _ => None,
        }
    }

    fn validate_response(self, response: &InferenceResponse) -> Result<(), String> {
        match self {
            Self::JsonObject => {
                let value: serde_json::Value = serde_json::from_str(&response.content)
                    .map_err(|error| format!("response was not JSON: {error}"))?;
                if value.get("ok").and_then(serde_json::Value::as_bool) == Some(true) {
                    Ok(())
                } else {
                    Err("response JSON did not contain boolean ok=true".to_owned())
                }
            }
            Self::ToolSchemaLoose
            | Self::ToolSchemaLooseStress
            | Self::ToolSchemaStrict
            | Self::ToolSchemaStrictStress
            | Self::ToolSchemaStrictStressPrompted => {
                if tool_probe_arguments(response).is_some() {
                    Ok(())
                } else {
                    Err("provider accepted tools/tool_choice but did not call the forced emit_probe function".to_owned())
                }
            }
            Self::SchemaFeature { transport, feature } => {
                let raw = match transport {
                    SchemaFeatureTransport::StructuredOutputs => response.content.as_str(),
                    SchemaFeatureTransport::Tools => tool_probe_arguments(response).ok_or_else(|| {
                        "provider accepted tools/tool_choice but did not call the forced emit_probe function".to_owned()
                    })?,
                };
                if schema_feature_probe_satisfied(feature, raw) {
                    Ok(())
                } else {
                    Err(format!(
                        "schema feature {} was accepted but not enforced",
                        feature.as_str()
                    ))
                }
            }
            Self::BasicChat | Self::SystemRole => {
                if response.content.trim().is_empty() {
                    Err("response content was empty".to_owned())
                } else {
                    Ok(())
                }
            }
            Self::JsonSchemaLoose
            | Self::JsonSchemaLooseStress
            | Self::StrictJsonSchema
            | Self::StrictJsonSchemaStress
            | Self::StrictJsonSchemaStressPrompted
            | Self::ReasoningOff
            | Self::ReasoningMinimal
            | Self::ReasoningLow
            | Self::ReasoningMedium
            | Self::ReasoningHigh
            | Self::ReasoningXhigh
            | Self::ReasoningMax
            | Self::ReasoningBudget256 => Ok(()),
        }
    }
}

#[derive(Debug)]
struct ProbeExecution {
    reasoning_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    reasoning_chars: Option<u64>,
    reasoning_trace: Option<serde_json::Value>,
    response_content: Option<String>,
    tool_arguments: Option<String>,
    puzzle_correct: Option<bool>,
    finish_reason: Option<String>,
    constraint_satisfied: Option<bool>,
    outcome: ConformanceOutcome,
    detail: Option<String>,
}

impl ProbeExecution {
    fn from_error(probe: ConformanceProbe, error: &ProviderError) -> Self {
        Self {
            reasoning_tokens: None,
            completion_tokens: None,
            reasoning_chars: None,
            reasoning_trace: None,
            response_content: None,
            tool_arguments: None,
            puzzle_correct: None,
            finish_reason: None,
            constraint_satisfied: None,
            outcome: classify_probe_error(probe, error),
            detail: Some(probe_error_detail(error)),
        }
    }
}

fn classify_probe_error(probe: ConformanceProbe, error: &ProviderError) -> ConformanceOutcome {
    match probe_http_status(error) {
        Some(400 | 422) => ConformanceOutcome::Failed,
        Some(404)
            if matches!(
                probe,
                ConformanceProbe::JsonSchemaLoose
                    | ConformanceProbe::JsonSchemaLooseStress
                    | ConformanceProbe::StrictJsonSchema
                    | ConformanceProbe::StrictJsonSchemaStress
                    | ConformanceProbe::StrictJsonSchemaStressPrompted
                    | ConformanceProbe::ToolSchemaLoose
                    | ConformanceProbe::ToolSchemaLooseStress
                    | ConformanceProbe::ToolSchemaStrict
                    | ConformanceProbe::ToolSchemaStrictStress
                    | ConformanceProbe::ToolSchemaStrictStressPrompted
            ) =>
        {
            ConformanceOutcome::Failed
        }
        _ => ConformanceOutcome::Inconclusive,
    }
}

fn probe_error_detail(error: &ProviderError) -> String {
    probe_http_status(error).map_or_else(
        || match error {
            ProviderError::NoEligibleEndpoint(_) => "no eligible endpoint".to_owned(),
            ProviderError::InvalidRequest(_) => "invalid request".to_owned(),
            ProviderError::InvalidStrategy(_) => "invalid strategy".to_owned(),
            ProviderError::Catalog(_) => "catalog error".to_owned(),
            ProviderError::Request(_) => "request error".to_owned(),
            ProviderError::Execution { .. } => "execution error".to_owned(),
            ProviderError::Response(_) => "response error".to_owned(),
        },
        |status| format!("HTTP {status}"),
    )
}

fn probe_http_status(error: &ProviderError) -> Option<u16> {
    let text = match error {
        ProviderError::Catalog(text)
        | ProviderError::Request(text)
        | ProviderError::Response(text) => text,
        ProviderError::Execution { detail, .. } => detail,
        ProviderError::NoEligibleEndpoint(_)
        | ProviderError::InvalidRequest(_)
        | ProviderError::InvalidStrategy(_) => return None,
    };
    let tail = text.split_once("HTTP ")?.1;
    let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
        .unwrap_or(0)
}

/// Versioned evidence from one provider/model conformance probe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceObservation {
    /// Execution surface, e.g. `openrouter` or direct `nvidia`.
    pub source: String,
    /// Opaque logical credential/account label. Never contains the secret itself.
    pub credential: Option<String>,
    /// Stable provider selector within the execution surface.
    pub provider_key: String,
    /// Optional exact endpoint refinement when that endpoint is independently addressable.
    pub endpoint_id: Option<String>,
    /// Model identifier spoken to this provider route.
    pub native_model_id: String,
    pub canonical_id: Option<String>,
    pub parameter: String,
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    #[serde(default)]
    pub reasoning_chars: Option<u64>,
    #[serde(default)]
    pub reasoning_trace: Option<serde_json::Value>,
    #[serde(default)]
    pub response_content: Option<String>,
    #[serde(default)]
    pub tool_arguments: Option<String>,
    #[serde(default)]
    pub puzzle_correct: Option<bool>,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub constraint_satisfied: Option<bool>,
    pub test_id: String,
    pub test_version: u32,
    pub outcome: ConformanceOutcome,
    pub observed_unix_ms: u64,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceProfile {
    pub observations: Vec<ConformanceObservation>,
}

impl ConformanceProfile {
    /// Return a current-version cached observation for one exact probe scope.
    #[must_use]
    pub fn cached_probe(
        &self,
        endpoint: &Endpoint,
        probe: ConformanceProbe,
    ) -> Option<&ConformanceObservation> {
        self.observations
            .iter()
            .filter(|observation| observation.matches_endpoint(endpoint))
            .filter(|observation| {
                observation.parameter == probe.parameter()
                    && observation.test_id == probe.test_id()
                    && observation.test_version == probe.test_version()
            })
            .max_by_key(|observation| observation.observed_unix_ms)
    }

    /// Record one observation, replacing an older observation for the same exact probe/version.
    pub fn record(&mut self, observation: ConformanceObservation) {
        if let Some(existing) = self.observations.iter_mut().find(|existing| {
            existing.same_probe_scope(&observation)
                && existing.test_version == observation.test_version
        }) {
            if observation.observed_unix_ms >= existing.observed_unix_ms {
                *existing = observation;
            }
            return;
        }
        self.observations.push(observation);
    }

    /// Apply only observations scoped to this exact endpoint/model route.
    pub fn apply_to_endpoint(&self, endpoint: &mut Endpoint) {
        let source = endpoint.source.clone();
        let credential = endpoint.credential.clone();
        let provider_key = endpoint.provider_key.clone();
        let endpoint_id = endpoint.id.clone();
        let native_model_id = endpoint.model.clone();
        let observations = self.observations.iter().filter(|observation| {
            observation.source == source
                && observation.credential == credential
                && observation.provider_key == provider_key
                && observation.native_model_id == native_model_id
                && observation
                    .endpoint_id
                    .as_ref()
                    .is_none_or(|id| id == &endpoint_id)
        });
        apply_observations(&mut endpoint.capabilities, observations);
    }
}

impl ConformanceObservation {
    fn effective_reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.reasoning_effort.or(match self.test_id.as_str() {
            "reasoning-off" => Some(ReasoningEffort::None),
            "reasoning-minimal" => Some(ReasoningEffort::Minimal),
            "reasoning-low" => Some(ReasoningEffort::Low),
            "reasoning-medium" => Some(ReasoningEffort::Medium),
            "reasoning-high" => Some(ReasoningEffort::High),
            "reasoning-xhigh" => Some(ReasoningEffort::Xhigh),
            "reasoning-max" => Some(ReasoningEffort::Max),
            _ => None,
        })
    }

    fn matches_endpoint(&self, endpoint: &Endpoint) -> bool {
        self.source == endpoint.source
            && self.credential == endpoint.credential
            && self.provider_key == endpoint.provider_key
            && self.native_model_id == endpoint.model
            && self
                .endpoint_id
                .as_ref()
                .is_none_or(|endpoint_id| endpoint_id == &endpoint.id)
    }

    fn same_probe_scope(&self, other: &Self) -> bool {
        self.source == other.source
            && self.credential == other.credential
            && self.provider_key == other.provider_key
            && self.endpoint_id == other.endpoint_id
            && self.native_model_id == other.native_model_id
            && self.parameter == other.parameter
            && self.test_id == other.test_id
    }
}

fn schema_feature_probe_identity(test_id: &str) -> Option<(SchemaFeatureTransport, SchemaFeature)> {
    let rest = test_id.strip_prefix("schema-feature-")?;
    let (transport, feature_name) = if let Some(feature) = rest.strip_prefix("structured_outputs-")
    {
        (SchemaFeatureTransport::StructuredOutputs, feature)
    } else {
        let feature = rest.strip_prefix("tools-")?;
        (SchemaFeatureTransport::Tools, feature)
    };
    Some((transport, SchemaFeature::from_str_name(feature_name)?))
}

fn apply_observations<'a>(
    capabilities: &mut EndpointCapabilities,
    observations: impl Iterator<Item = &'a ConformanceObservation>,
) {
    let latest = latest_observations(observations);
    apply_parameter_and_reasoning_evidence(capabilities, latest.values().copied());
    apply_schema_feature_evidence(capabilities, latest.values().copied());
    apply_constraint_evidence(capabilities, &latest);
}

fn latest_observations<'a>(
    observations: impl Iterator<Item = &'a ConformanceObservation>,
) -> BTreeMap<(&'a str, &'a str), &'a ConformanceObservation> {
    let mut latest = BTreeMap::new();
    for observation in observations {
        let key = (observation.parameter.as_str(), observation.test_id.as_str());
        let replace = latest
            .get(&key)
            .is_none_or(|current: &&ConformanceObservation| {
                (observation.test_version, observation.observed_unix_ms)
                    > (current.test_version, current.observed_unix_ms)
            });
        if replace {
            latest.insert(key, observation);
        }
    }
    latest
}

fn apply_parameter_and_reasoning_evidence<'a>(
    capabilities: &mut EndpointCapabilities,
    observations: impl Iterator<Item = &'a ConformanceObservation>,
) {
    let mut passed = BTreeSet::new();
    let mut failed = BTreeSet::new();
    let mut reasoning_passed = BTreeSet::new();
    let mut reasoning_failed = BTreeSet::new();
    for observation in observations {
        if schema_feature_probe_identity(&observation.test_id).is_some() {
            continue;
        }
        if let Some(effort) = observation.effective_reasoning_effort() {
            match observation.outcome {
                ConformanceOutcome::Passed => {
                    reasoning_passed.insert(effort);
                    passed.insert("reasoning".to_owned());
                }
                ConformanceOutcome::Failed => {
                    reasoning_failed.insert(effort);
                }
                ConformanceOutcome::Inconclusive => {}
            }
            continue;
        }
        if matches!(
            observation.test_id.as_str(),
            "tool-schema-strict" | "tool-schema-strict-stress"
        ) {
            if observation.outcome == ConformanceOutcome::Passed {
                passed.extend(["tools".to_owned(), "tool_choice".to_owned()]);
            }
            continue;
        }
        match observation.outcome {
            ConformanceOutcome::Passed => {
                passed.insert(observation.parameter.clone());
                if observation.test_id == "tool-schema-loose" {
                    passed.insert("tool_choice".to_owned());
                }
            }
            ConformanceOutcome::Failed => {
                failed.insert(observation.parameter.clone());
                if observation.test_id == "tool-schema-loose" {
                    failed.insert("tool_choice".to_owned());
                }
            }
            ConformanceOutcome::Inconclusive => {}
        }
    }
    apply_parameter_sets(capabilities, &passed, failed);
    for effort in reasoning_passed.difference(&reasoning_failed) {
        capabilities.rejected_reasoning_efforts.remove(effort);
        capabilities.verified_reasoning_efforts.insert(*effort);
    }
    for effort in reasoning_failed {
        capabilities.verified_reasoning_efforts.remove(&effort);
        capabilities.rejected_reasoning_efforts.insert(effort);
    }
}

fn apply_parameter_sets(
    capabilities: &mut EndpointCapabilities,
    passed: &BTreeSet<String>,
    failed: BTreeSet<String>,
) {
    for parameter in passed.difference(&failed) {
        capabilities.rejected_parameters.remove(parameter);
        capabilities.verified_parameters.insert(parameter.clone());
    }
    for parameter in failed {
        capabilities.verified_parameters.remove(&parameter);
        capabilities.rejected_parameters.insert(parameter);
    }
}

fn apply_schema_feature_evidence<'a>(
    capabilities: &mut EndpointCapabilities,
    observations: impl Iterator<Item = &'a ConformanceObservation>,
) {
    let mut passed: BTreeMap<&str, BTreeSet<SchemaFeature>> = BTreeMap::new();
    let mut failed: BTreeMap<&str, BTreeSet<SchemaFeature>> = BTreeMap::new();
    for observation in observations {
        let Some((transport, feature)) = schema_feature_probe_identity(&observation.test_id) else {
            continue;
        };
        let target = if observation.outcome == ConformanceOutcome::Passed
            && observation.constraint_satisfied == Some(true)
        {
            &mut passed
        } else if observation.outcome == ConformanceOutcome::Inconclusive {
            continue;
        } else {
            &mut failed
        };
        target
            .entry(transport.parameter())
            .or_default()
            .insert(feature);
    }
    for (parameter, features) in passed {
        update_schema_feature_set(capabilities, parameter, features, true);
    }
    for (parameter, features) in failed {
        update_schema_feature_set(capabilities, parameter, features, false);
    }
}

fn update_schema_feature_set(
    capabilities: &mut EndpointCapabilities,
    parameter: &str,
    features: BTreeSet<SchemaFeature>,
    verified: bool,
) {
    let (add, remove) = if verified {
        (
            &mut capabilities.verified_schema_features,
            &mut capabilities.rejected_schema_features,
        )
    } else {
        (
            &mut capabilities.rejected_schema_features,
            &mut capabilities.verified_schema_features,
        )
    };
    let removal = remove.entry(parameter.to_owned()).or_default();
    for feature in &features {
        removal.remove(feature);
    }
    add.entry(parameter.to_owned())
        .or_default()
        .extend(features);
}

fn apply_constraint_evidence(
    capabilities: &mut EndpointCapabilities,
    latest: &BTreeMap<(&str, &str), &ConformanceObservation>,
) {
    for (parameter, simple_id, stress_id) in [
        (
            "structured_outputs",
            "strict-json-schema",
            "strict-json-schema-stress",
        ),
        ("tools", "tool-schema-strict", "tool-schema-strict-stress"),
    ] {
        apply_constraint_levels(
            capabilities,
            parameter,
            latest
                .values()
                .find(|observation| observation.test_id == simple_id)
                .copied(),
            latest
                .values()
                .find(|observation| observation.test_id == stress_id)
                .copied(),
        );
    }
}

fn apply_constraint_levels(
    capabilities: &mut EndpointCapabilities,
    parameter: &str,
    simple: Option<&ConformanceObservation>,
    stress: Option<&ConformanceObservation>,
) {
    let simple = simple.filter(|observation| observation.outcome == ConformanceOutcome::Passed);
    let stress = stress.filter(|observation| observation.outcome == ConformanceOutcome::Passed);
    let observed = match (
        simple.and_then(|observation| observation.constraint_satisfied),
        stress.and_then(|observation| observation.constraint_satisfied),
    ) {
        (_, Some(true)) => Some(ConstraintEnforcement::Enforced),
        (Some(true), Some(false)) => Some(ConstraintEnforcement::HintOnly),
        (Some(false), _) => Some(ConstraintEnforcement::Ineffective),
        _ => None,
    };
    if let Some(observed) = observed {
        capabilities
            .constraint_enforcement
            .insert(parameter.to_owned(), observed);
    } else {
        capabilities.constraint_enforcement.remove(parameter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EndpointStats, Pricing};

    fn endpoint(id: &str, model: &str) -> Endpoint {
        Endpoint {
            source: "openrouter".to_owned(),
            credential: None,
            provider_key: id.split('/').next().unwrap_or(id).to_owned(),
            id: id.to_owned(),
            provider: "provider".to_owned(),
            model: model.to_owned(),
            route_class: crate::ModelRouteClass::Standard,
            display_name: id.to_owned(),
            quantization: None,
            status: 0,
            context_length: None,
            max_prompt_tokens: None,
            max_completion_tokens: None,
            pricing: Pricing::default(),
            stats: EndpointStats::default(),
            workload_policy: crate::WorkloadPolicy::default(),
            capabilities: EndpointCapabilities {
                adapter_parameters: [
                    "chat_completions",
                    "system_role",
                    "response_format",
                    "structured_outputs",
                    "reasoning",
                    "reasoning_effort",
                    "seed",
                    "tools",
                    "tool_choice",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
                ..EndpointCapabilities::default()
            },
        }
    }

    fn observation(
        endpoint_id: &str,
        model: &str,
        parameter: &str,
        test_id: &str,
        version: u32,
        observed_unix_ms: u64,
        outcome: ConformanceOutcome,
    ) -> ConformanceObservation {
        ConformanceObservation {
            source: "openrouter".to_owned(),
            credential: None,
            provider_key: endpoint_id
                .split('/')
                .next()
                .unwrap_or(endpoint_id)
                .to_owned(),
            endpoint_id: Some(endpoint_id.to_owned()),
            native_model_id: model.to_owned(),
            canonical_id: None,
            parameter: parameter.to_owned(),
            reasoning_effort: None,
            reasoning_tokens: None,
            completion_tokens: None,
            reasoning_chars: None,
            reasoning_trace: None,
            response_content: None,
            tool_arguments: None,
            puzzle_correct: None,
            finish_reason: None,
            constraint_satisfied: None,
            test_id: test_id.to_owned(),
            test_version: version,
            outcome,
            observed_unix_ms,
            detail: None,
        }
    }

    #[test]
    fn schema_feature_evidence_is_transport_scoped_and_does_not_reject_base_transport() {
        let mut route = endpoint("provider/fp8", "model");
        route
            .capabilities
            .adapter_parameters
            .extend(["structured_outputs".to_owned(), "tools".to_owned()]);
        let mut passed = observation(
            "provider/fp8",
            "model",
            "structured_outputs",
            "schema-feature-structured_outputs-one_of",
            1,
            10,
            ConformanceOutcome::Passed,
        );
        passed.constraint_satisfied = Some(true);
        let mut failed = observation(
            "provider/fp8",
            "model",
            "tools",
            "schema-feature-tools-one_of",
            1,
            11,
            ConformanceOutcome::Failed,
        );
        failed.constraint_satisfied = Some(false);
        ConformanceProfile {
            observations: vec![passed, failed],
        }
        .apply_to_endpoint(&mut route);
        assert!(
            route
                .capabilities
                .verified_schema_features
                .get("structured_outputs")
                .is_some_and(|set| set.contains(&SchemaFeature::OneOf))
        );
        assert!(
            route
                .capabilities
                .rejected_schema_features
                .get("tools")
                .is_some_and(|set| set.contains(&SchemaFeature::OneOf))
        );
        assert!(!route.capabilities.rejected_parameters.contains("tools"));
    }

    #[test]
    fn schema_feature_probe_round_trips_as_structured_enum_variant() -> Result<(), serde_json::Error>
    {
        let probe = ConformanceProbe::SchemaFeature {
            transport: SchemaFeatureTransport::StructuredOutputs,
            feature: SchemaFeature::TuplePrefixItems,
        };
        let value = serde_json::to_value(probe)?;
        let decoded: ConformanceProbe = serde_json::from_value(value)?;
        assert_eq!(decoded, probe);
        assert_eq!(
            probe.test_id(),
            "schema-feature-structured_outputs-tuple_prefix_items"
        );
        Ok(())
    }

    #[test]
    fn newer_probe_result_replaces_older_result_for_same_test() {
        let mut route = endpoint("endpoint", "model");
        route
            .capabilities
            .supported_parameters
            .insert("structured_outputs".to_owned());
        ConformanceProfile {
            observations: vec![
                observation(
                    "endpoint",
                    "model",
                    "structured_outputs",
                    "strict-json-schema",
                    1,
                    10,
                    ConformanceOutcome::Failed,
                ),
                observation(
                    "endpoint",
                    "model",
                    "structured_outputs",
                    "strict-json-schema",
                    1,
                    20,
                    ConformanceOutcome::Passed,
                ),
            ],
        }
        .apply_to_endpoint(&mut route);
        assert!(route.capabilities.supports_parameter("structured_outputs"));
        assert!(
            route
                .capabilities
                .verified_parameters
                .contains("structured_outputs")
        );
        assert!(route.capabilities.rejected_parameters.is_empty());
    }

    #[test]
    fn failed_test_rejects_parameter_only_on_matching_route() {
        let mut target = endpoint("endpoint-a", "model");
        target
            .capabilities
            .supported_parameters
            .insert("tools".to_owned());
        let mut other = endpoint("endpoint-b", "model");
        other
            .capabilities
            .supported_parameters
            .insert("tools".to_owned());
        let profile = ConformanceProfile {
            observations: vec![
                observation(
                    "endpoint-a",
                    "model",
                    "tools",
                    "single-tool",
                    1,
                    10,
                    ConformanceOutcome::Passed,
                ),
                observation(
                    "endpoint-a",
                    "model",
                    "tools",
                    "streamed-tool-args",
                    1,
                    10,
                    ConformanceOutcome::Failed,
                ),
            ],
        };
        profile.apply_to_endpoint(&mut target);
        profile.apply_to_endpoint(&mut other);
        assert!(!target.capabilities.supports_parameter("tools"));
        assert!(other.capabilities.supports_parameter("tools"));
    }

    #[test]
    fn evidence_does_not_cross_execution_surfaces() {
        let mut route = endpoint("deepinfra/fp8", "model");
        route
            .capabilities
            .supported_parameters
            .insert("tools".to_owned());
        let mut observation = observation(
            "deepinfra/fp8",
            "model",
            "tools",
            "single-tool",
            1,
            10,
            ConformanceOutcome::Failed,
        );
        observation.source = "direct-deepinfra".to_owned();
        ConformanceProfile {
            observations: vec![observation],
        }
        .apply_to_endpoint(&mut route);
        assert!(route.capabilities.supports_parameter("tools"));
    }

    #[test]
    fn probe_error_detail_drops_provider_response_body() {
        let error = ProviderError::Execution {
            phase: crate::ExecutionFailurePhase::ResponseReceived,
            detail: "HTTP 429 Too Many Requests: {\"user_id\":\"secret-ish\"}".to_owned(),
            quota_observations: Vec::new(),
        };
        assert_eq!(probe_error_detail(&error), "HTTP 429");
        assert_eq!(
            classify_probe_error(ConformanceProbe::BasicChat, &error),
            ConformanceOutcome::Inconclusive
        );
    }

    #[test]
    fn inconclusive_result_does_not_override_existing_evidence() {
        let mut route = endpoint("endpoint", "model");
        route
            .capabilities
            .verified_parameters
            .insert("seed".to_owned());
        ConformanceProfile {
            observations: vec![observation(
                "endpoint",
                "model",
                "seed",
                "determinism",
                1,
                10,
                ConformanceOutcome::Inconclusive,
            )],
        }
        .apply_to_endpoint(&mut route);
        assert!(route.capabilities.verified_parameters.contains("seed"));
        assert!(route.capabilities.rejected_parameters.is_empty());
    }

    #[test]
    fn reasoning_off_probe_is_explicit_and_requests_trace() {
        let request = ConformanceProbe::ReasoningOff.request("model");
        assert_eq!(request.requirements.reasoning, ReasoningEffort::None);
        assert!(
            request
                .requirements
                .required_parameters
                .contains("reasoning")
        );
        assert!(request.include_reasoning);
    }

    #[test]
    fn reasoning_budget_probe_is_independent_from_effort() {
        let request = ConformanceProbe::ReasoningBudget256.request("model");
        assert_eq!(request.requirements.reasoning, ReasoningEffort::None);
        assert_eq!(request.requirements.reasoning_max_tokens, Some(256));
        assert!(request.include_reasoning);
        assert_eq!(
            ConformanceProbe::ReasoningBudget256.parameter(),
            "reasoning_max_tokens"
        );
    }

    #[test]
    fn reasoning_evidence_is_scoped_to_exact_effort() {
        let mut route = endpoint("endpoint", "model");
        route
            .capabilities
            .supported_parameters
            .insert("reasoning".to_owned());
        route.capabilities.reasoning = crate::ReasoningSupport::GatewayNormalized;

        let mut low = observation(
            "endpoint",
            "model",
            "reasoning",
            "reasoning-low",
            3,
            10,
            ConformanceOutcome::Passed,
        );
        low.reasoning_effort = Some(ReasoningEffort::Low);
        let mut high = observation(
            "endpoint",
            "model",
            "reasoning",
            "reasoning-high",
            3,
            10,
            ConformanceOutcome::Failed,
        );
        high.reasoning_effort = Some(ReasoningEffort::High);

        ConformanceProfile {
            observations: vec![low, high],
        }
        .apply_to_endpoint(&mut route);

        assert!(
            route
                .capabilities
                .verified_reasoning_efforts
                .contains(&ReasoningEffort::Low)
        );
        assert!(
            route
                .capabilities
                .rejected_reasoning_efforts
                .contains(&ReasoningEffort::High)
        );
        assert!(
            route
                .capabilities
                .supports_reasoning_effort(ReasoningEffort::Low, true)
        );
        assert!(
            !route
                .capabilities
                .supports_reasoning_effort(ReasoningEffort::High, false)
        );
        assert!(
            route
                .capabilities
                .supports_reasoning_effort(ReasoningEffort::Medium, false)
        );
    }

    #[test]
    fn historical_reasoning_probe_ids_recover_effort_scope() {
        let mut route = endpoint("endpoint", "model");
        route
            .capabilities
            .supported_parameters
            .insert("reasoning".to_owned());
        route.capabilities.reasoning = crate::ReasoningSupport::GatewayNormalized;
        ConformanceProfile {
            observations: vec![observation(
                "endpoint",
                "model",
                "reasoning",
                "reasoning-low",
                1,
                10,
                ConformanceOutcome::Failed,
            )],
        }
        .apply_to_endpoint(&mut route);
        assert!(
            route
                .capabilities
                .rejected_reasoning_efforts
                .contains(&ReasoningEffort::Low)
        );
        assert!(!route.capabilities.rejected_parameters.contains("reasoning"));
    }

    #[test]
    fn tool_stress_probe_proves_enforced_behavior() {
        let mut route = endpoint("endpoint", "model");
        route
            .capabilities
            .supported_parameters
            .extend(["tools".to_owned(), "tool_choice".to_owned()]);
        let mut loose = observation(
            "endpoint",
            "model",
            "tools",
            "tool-schema-loose",
            2,
            10,
            ConformanceOutcome::Passed,
        );
        loose.constraint_satisfied = Some(false);
        let mut strict = observation(
            "endpoint",
            "model",
            "tools",
            "tool-schema-strict",
            2,
            11,
            ConformanceOutcome::Passed,
        );
        strict.constraint_satisfied = Some(true);
        let mut stress = observation(
            "endpoint",
            "model",
            "tools",
            "tool-schema-strict-stress",
            1,
            12,
            ConformanceOutcome::Passed,
        );
        stress.constraint_satisfied = Some(true);
        ConformanceProfile {
            observations: vec![loose, strict, stress],
        }
        .apply_to_endpoint(&mut route);
        assert_eq!(
            route.capabilities.constraint_enforcement("tools"),
            Some(ConstraintEnforcement::Enforced)
        );
        assert!(route.capabilities.supports_parameter("tools"));
        assert!(route.capabilities.supports_parameter("tool_choice"));
    }

    #[test]
    fn strict_simple_success_without_stress_does_not_overclaim_constraint() {
        let mut route = endpoint("endpoint", "model");
        route
            .capabilities
            .supported_parameters
            .insert("structured_outputs".to_owned());
        let mut loose = observation(
            "endpoint",
            "model",
            "response_format",
            "json-schema-loose",
            2,
            10,
            ConformanceOutcome::Passed,
        );
        loose.constraint_satisfied = Some(true);
        let mut strict = observation(
            "endpoint",
            "model",
            "structured_outputs",
            "strict-json-schema",
            2,
            11,
            ConformanceOutcome::Passed,
        );
        strict.constraint_satisfied = Some(true);
        ConformanceProfile {
            observations: vec![loose, strict],
        }
        .apply_to_endpoint(&mut route);
        assert_eq!(
            route
                .capabilities
                .constraint_enforcement("structured_outputs"),
            None
        );
    }

    #[test]
    fn strict_schema_easy_violation_marks_control_ineffective() {
        let mut route = endpoint("endpoint", "model");
        route
            .capabilities
            .supported_parameters
            .insert("structured_outputs".to_owned());
        let mut strict = observation(
            "endpoint",
            "model",
            "structured_outputs",
            "strict-json-schema",
            2,
            11,
            ConformanceOutcome::Passed,
        );
        strict.constraint_satisfied = Some(false);
        ConformanceProfile {
            observations: vec![strict],
        }
        .apply_to_endpoint(&mut route);
        assert_eq!(
            route
                .capabilities
                .constraint_enforcement("structured_outputs"),
            Some(ConstraintEnforcement::Ineffective)
        );
    }

    #[test]
    fn strict_schema_stress_violation_after_easy_success_is_hint_only() {
        let mut route = endpoint("endpoint", "model");
        route
            .capabilities
            .supported_parameters
            .insert("structured_outputs".to_owned());
        let mut simple = observation(
            "endpoint",
            "model",
            "structured_outputs",
            "strict-json-schema",
            2,
            10,
            ConformanceOutcome::Passed,
        );
        simple.constraint_satisfied = Some(true);
        let mut stress = observation(
            "endpoint",
            "model",
            "structured_outputs",
            "strict-json-schema-stress",
            1,
            11,
            ConformanceOutcome::Passed,
        );
        stress.constraint_satisfied = Some(false);
        ConformanceProfile {
            observations: vec![simple, stress],
        }
        .apply_to_endpoint(&mut route);
        assert_eq!(
            route
                .capabilities
                .constraint_enforcement("structured_outputs"),
            Some(ConstraintEnforcement::HintOnly)
        );
    }

    #[test]
    fn strict_tool_rejection_does_not_reject_ordinary_tool_support() {
        let mut route = endpoint("endpoint", "model");
        route
            .capabilities
            .supported_parameters
            .extend(["tools".to_owned(), "tool_choice".to_owned()]);
        let mut loose = observation(
            "endpoint",
            "model",
            "tools",
            "tool-schema-loose",
            2,
            10,
            ConformanceOutcome::Passed,
        );
        loose.constraint_satisfied = Some(false);
        let strict = observation(
            "endpoint",
            "model",
            "tools",
            "tool-schema-strict",
            2,
            11,
            ConformanceOutcome::Failed,
        );
        ConformanceProfile {
            observations: vec![loose, strict],
        }
        .apply_to_endpoint(&mut route);
        assert!(route.capabilities.supports_parameter("tools"));
        assert!(route.capabilities.supports_parameter("tool_choice"));
        assert_eq!(route.capabilities.constraint_enforcement("tools"), None);
    }

    #[test]
    fn cached_probe_is_exactly_credential_and_endpoint_scoped() {
        let mut route = endpoint("endpoint", "model");
        route.credential = Some("account-a".to_owned());
        let mut cached = observation(
            "endpoint",
            "model",
            "chat_completions",
            "basic-chat",
            ConformanceProbe::BasicChat.test_version(),
            10,
            ConformanceOutcome::Passed,
        );
        cached.credential = Some("account-a".to_owned());
        let profile = ConformanceProfile {
            observations: vec![cached],
        };
        assert!(
            profile
                .cached_probe(&route, ConformanceProbe::BasicChat)
                .is_some()
        );
        route.credential = Some("account-b".to_owned());
        assert!(
            profile
                .cached_probe(&route, ConformanceProbe::BasicChat)
                .is_none()
        );
    }

    #[test]
    fn record_replaces_same_probe_version_instead_of_growing_forever() {
        let first = observation(
            "endpoint",
            "model",
            "chat_completions",
            "basic-chat",
            1,
            10,
            ConformanceOutcome::Inconclusive,
        );
        let mut second = first.clone();
        second.observed_unix_ms = 20;
        second.outcome = ConformanceOutcome::Passed;
        let mut profile = ConformanceProfile::default();
        profile.record(first);
        profile.record(second);
        assert_eq!(profile.observations.len(), 1);
        assert_eq!(profile.observations[0].outcome, ConformanceOutcome::Passed);
    }
}
