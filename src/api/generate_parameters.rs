use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema)]
pub struct GenerateParameters {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = json!(1))]
    pub best_of: Option<i32>,

    #[serde(default = "default_true")]
    pub decoder_input_details: bool,

    #[serde(default = "default_true")]
    pub details: bool,

    #[serde(default)]
    pub do_sample: bool,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = json!(20))]
    pub max_new_tokens: Option<i32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = json!(1.03))]
    pub repetition_penalty: Option<f32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = json!(false))]
    pub return_full_text: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = json!(299792458))]
    pub seed: Option<i64>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schema(example = json!(vec!["photographer"]))]
    pub stop: Vec<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = json!(0.5))]
    pub temperature: Option<f32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = json!(10))]
    pub top_k: Option<i32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = json!(5))]
    pub top_n_tokens: Option<i32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = json!(0.95))]
    pub top_p: Option<f32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate: Option<i32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = json!(0.95))]
    pub typical_p: Option<f32>,

    #[serde(default)]
    pub watermark: bool,
}

fn default_true() -> bool {
    true
}
