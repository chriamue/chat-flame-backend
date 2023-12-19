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
