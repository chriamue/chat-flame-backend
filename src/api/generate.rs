use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::config::Config;

use super::{
    generate_parameters::GenerateParameters, generate_stream::generate_stream_handler,
    generate_text::GenerateRequest, ErrorResponse,
};

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CompatGenerateRequest {
    #[schema(example = "My name is Olivier and I")]
    pub inputs: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<GenerateParameters>,

    #[serde(default)]
    pub stream: bool,
}

/// Generate tokens
#[utoipa::path(
    post,
    path = "/",
    request_body = CompatGenerateRequest,
    responses(
        (status = 200, description = "Generated Text", body = StreamResponse),
    ),
    tag = "Text Generation Inference"
)]
pub async fn generate_handler(
    config: State<Config>,
    Json(payload): Json<CompatGenerateRequest>,
) -> impl IntoResponse {
    if !payload.stream {
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            Json(ErrorResponse {
                error: "Use /generate endpoint if not streaming".to_string(),
                error_type: None,
            }),
        ));
    }
    Ok(generate_stream_handler(
        config,
        Json(GenerateRequest {
            inputs: payload.inputs,
            parameters: payload.parameters,
        }),
    )
    .await)
}
