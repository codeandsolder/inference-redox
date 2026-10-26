use inference_redox::schema_feature_requirements;
use serde_json::Value;
use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("cases path required")?;
    let rows: Vec<Value> = serde_json::from_slice(&fs::read(path)?)?;
    for row in rows {
        let id = row.get("id").and_then(Value::as_str).ok_or("missing id")?;
        let schema = row.get("schema").ok_or("missing schema")?;
        let features = schema_feature_requirements(schema);
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({"id":id,"features":features}))?
        );
    }
    Ok(())
}
