use axum::{http::StatusCode, Json};
use utoipa::OpenApi;

use super::{TextGenerationRequest, TextGenerationResponse};

/// Generate text based on the given prompt
///
/// This endpoint receives a prompt and returns the generated text.
#[utoipa::path(
    post,
    path = "/generate-text",
    request_body = TextGenerationRequest,
    responses(
        (status = 200, description = "Text generation successful", body = TextGenerationResponse),
        (status = 500, description = "Internal server error")
    )
)]
pub async fn generate_text_handler(
    Json(payload): Json<TextGenerationRequest>,
) -> Result<Json<TextGenerationResponse>, StatusCode> {
    // Call the Hugging Face API or perform the operation here
    let generated_text = "This is a dummy response".to_string();

    Ok(Json(TextGenerationResponse { generated_text }))
}

#[derive(OpenApi)]
#[openapi(paths(generate_text_handler),
components(
    schemas(TextGenerationRequest, TextGenerationResponse)
), tags((name = "Text Generation", description = "Text generation API")))]
pub struct ApiDoc;
