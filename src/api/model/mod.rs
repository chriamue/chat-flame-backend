//! This module defines the data structures used for API requests and responses.
//! These include various types of responses for text generation, error handling,
//! and information about the model and generation parameters.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Enumerates the reasons why text generation may finish.
#[derive(Debug, Serialize, Deserialize, ToSchema, Clone)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Generation finished due to reaching the maximum length.
    Length,
    /// Generation finished due to reaching the end-of-sequence token.
    EosToken,
    /// Generation finished due to reaching a stop sequence.
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_tokens: Option<Vec<Token>>,
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

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct Info {
    #[serde(rename = "docker_label", skip_serializing_if = "Option::is_none")]
    pub docker_label: Option<String>,

    #[serde(rename = "max_batch_total_tokens")]
    pub max_batch_total_tokens: i32,

    #[serde(rename = "max_best_of")]
    pub max_best_of: i32,

    #[serde(rename = "max_concurrent_requests")]
    pub max_concurrent_requests: i32,

    #[serde(rename = "max_input_length")]
    pub max_input_length: i32,

    #[serde(rename = "max_stop_sequences")]
    pub max_stop_sequences: i32,

    #[serde(rename = "max_total_tokens")]
    pub max_total_tokens: i32,

    #[serde(rename = "max_waiting_tokens")]
    pub max_waiting_tokens: i32,

    #[serde(rename = "model_device_type")]
    pub model_device_type: String,

    #[serde(rename = "model_dtype")]
    pub model_dtype: String,

    #[serde(rename = "model_id")]
    pub model_id: String,

    #[serde(rename = "model_pipeline_tag", skip_serializing_if = "Option::is_none")]
    pub model_pipeline_tag: Option<String>,

    #[serde(rename = "model_sha", skip_serializing_if = "Option::is_none")]
    pub model_sha: Option<String>,

    #[serde(rename = "sha", skip_serializing_if = "Option::is_none")]
    pub sha: Option<String>,

    #[serde(rename = "validation_workers")]
    pub validation_workers: i32,

    #[serde(rename = "version")]
    pub version: String,

    #[serde(rename = "waiting_served_ratio")]
    pub waiting_served_ratio: f32,
}
