use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, ToSchema, Clone)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Length,
    EosToken,
    StopSequence,
}

#[derive(Debug, Serialize, Deserialize, ToSchema, Clone)]
pub struct Token {
    pub id: i32,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprob: Option<f64>,
    pub special: bool,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct PrefillToken {
    pub id: i32,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprob: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BestOfSequence {
    pub finish_reason: FinishReason,
    pub generated_text: String,
    pub generated_tokens: i32,
    pub prefill: Vec<PrefillToken>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    pub tokens: Vec<Token>,
    pub top_tokens: Vec<Vec<Token>>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct Details {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of_sequences: Option<Vec<BestOfSequence>>,
    pub finish_reason: FinishReason,
    pub generated_tokens: i32,
    pub prefill: Vec<PrefillToken>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    pub tokens: Vec<Token>,
    pub top_tokens: Vec<Vec<Token>>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema, Clone)]
pub struct StreamDetails {
    pub finish_reason: FinishReason,
    pub generated_tokens: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema, Clone)]
pub struct StreamResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<StreamDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_text: Option<String>,
    pub token: Token,
    pub top_tokens: Vec<Token>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ErrorResponse {
    pub error: String,
    pub error_type: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
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
    pub temperature: Option<f64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = json!(10))]
    pub top_k: Option<i32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = json!(5))]
    pub top_n_tokens: Option<i32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = json!(0.95))]
    pub top_p: Option<f64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate: Option<i32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = json!(0.95))]
    pub typical_p: Option<f32>,

    #[serde(default)]
    pub watermark: bool,
}

#[derive(Deserialize, ToSchema, Debug)]
pub struct GenerateRequest {
    #[schema(example = "My name is John")]
    pub inputs: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<GenerateParameters>,
}

#[derive(Serialize, ToSchema)]
pub struct GenerateResponse {
    pub generated_text: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CompatGenerateRequest {
    #[schema(example = "My name is Olivier and I")]
    pub inputs: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<GenerateParameters>,

    #[serde(default)]
    pub stream: bool,
}

fn default_true() -> bool {
    true
}
