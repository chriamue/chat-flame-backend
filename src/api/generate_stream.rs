use axum::{response::IntoResponse, Json};
use axum_streams::StreamBodyAs;

use crate::llm::create_text_generation;

use super::generate::GenerateRequest;

/// Generate tokens
#[utoipa::path(
    post,
    path = "/generate_stream",
    request_body = GenerateRequest,
    responses(
        (status = 200, description = "Generated Text", body = StreamResponse),
    ),
    tag = "Text Generation Inference"
)]
pub async fn generate_stream_handler(Json(request): Json<GenerateRequest>) -> impl IntoResponse {
    let mut generator = create_text_generation().unwrap();
    let stream = generator.run_stream(&request.inputs, 50);

    StreamBodyAs::json_nl(stream)
}
