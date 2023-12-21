use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};

use crate::{
    api::model::{CompatGenerateRequest, ErrorResponse, GenerateRequest},
    config::Config,
};

use super::generate_stream::generate_stream_handler;

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
