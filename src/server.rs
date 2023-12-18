use axum::{
    routing::post,
    Router,
    Json,
    http::StatusCode,
};

use crate::api::{TextGenerationRequest, TextGenerationResponse};

async fn generate_text_handler(Json(payload): Json<TextGenerationRequest>) -> Result<Json<TextGenerationResponse>, StatusCode> {
    // Call the Hugging Face API or perform the operation here
    let generated_text = "This is a dummy response".to_string();

    Ok(Json(TextGenerationResponse { generated_text }))
}

pub fn server() -> Router {
    Router::new().route("/generate-text", post(generate_text_handler))
}
