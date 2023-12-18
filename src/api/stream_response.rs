use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema, Clone)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Length,
    EosToken,
    StopSequence,
}

#[derive(Serialize, Deserialize, ToSchema, Clone)]
pub struct Token {
    pub id: i32,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprob: Option<f64>,
    pub special: bool,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct PrefillToken {
    id: i32,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprob: Option<f64>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct BestOfSequence {
    finish_reason: FinishReason,
    generated_text: String,
    generated_tokens: i32,
    prefill: Vec<PrefillToken>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    tokens: Vec<Token>,
    top_tokens: Vec<Vec<Token>>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct Details {
    #[serde(skip_serializing_if = "Option::is_none")]
    best_of_sequences: Option<Vec<BestOfSequence>>,
    finish_reason: FinishReason,
    generated_tokens: i32,
    prefill: Vec<PrefillToken>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    tokens: Vec<Token>,
    top_tokens: Vec<Vec<Token>>,
}

#[derive(Serialize, Deserialize, ToSchema, Clone)]
pub struct StreamDetails {
    finish_reason: FinishReason,
    generated_tokens: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
}

#[derive(Serialize, Deserialize, ToSchema, Clone)]
pub struct StreamResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<StreamDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_text: Option<String>,
    pub token: Token,
    pub top_tokens: Vec<Token>,
}
