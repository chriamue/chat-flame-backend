use axum::{http::StatusCode, routing::post, Json, Router};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::api::{openapi::ApiDoc, TextGenerationRequest, TextGenerationResponse};

async fn generate_text_handler(
    Json(payload): Json<TextGenerationRequest>,
) -> Result<Json<TextGenerationResponse>, StatusCode> {
    // Call the Hugging Face API or perform the operation here
    let generated_text = "This is a dummy response".to_string();

    Ok(Json(TextGenerationResponse { generated_text }))
}

pub fn server() -> Router {
    let app = Router::new().route("/generate-text", post(generate_text_handler));
    app.merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
}
