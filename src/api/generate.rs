use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::llm::create_text_generation;

use super::ErrorResponse;

#[derive(Deserialize, ToSchema)]
pub struct TextGenerationRequest {
    pub inputs: String,
}

#[derive(Serialize, ToSchema)]
pub struct TextGenerationResponse {
    pub generated_text: String,
    // Add other fields as necessary
}

/// Generate tokens
#[utoipa::path(
    post,
    path = "/generate",
    request_body = TextGenerationRequest,
    responses(
        (status = 200, description = "Generated Text", body = TextGenerationResponse),
        (status = 422, description = "Input validation error", body = ErrorResponse, example = json!(ErrorResponse {error:"Input validation error".to_string(), error_type: None })),
        (status = 424, description = "Generation Error", body = ErrorResponse, example = json!(ErrorResponse {error:"Request failed during generation".to_string(), error_type: None })),
        (status = 429, description = "Model is overloaded", body = ErrorResponse, example = json!(ErrorResponse {error:"Model is overloaded".to_string(), error_type: None })),
        (status = 500, description = "Incomplete generation", body = ErrorResponse, example = json!(ErrorResponse {error:"Incomplete generation".to_string(), error_type: None }))
    ),
    tag = "Text Generation Inference"
)]
pub async fn generate_text_handler(
    Json(payload): Json<TextGenerationRequest>,
) -> Result<Json<TextGenerationResponse>, (StatusCode, Json<ErrorResponse>)> {
    let generator = create_text_generation();
    match generator {
        Ok(mut generator) => {
            let generated_text = generator.run(&payload.inputs, 50);
            match generated_text {
                Ok(generated_text) => match generated_text {
                    Some(text) => Ok(Json(TextGenerationResponse {
                        generated_text: text,
                    })),
                    None => Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: "Incomplete generation".to_string(),
                            error_type: None,
                        }),
                    )),
                },
                Err(_) => Err((
                    StatusCode::FAILED_DEPENDENCY,
                    Json(ErrorResponse {
                        error: "Request failed during generation".to_string(),
                        error_type: None,
                    }),
                )),
            }
        }
        Err(_) => Err((
            StatusCode::TOO_MANY_REQUESTS,
            Json(ErrorResponse {
                error: "Model is overloaded".to_string(),
                error_type: None,
            }),
        )),
    }
}
