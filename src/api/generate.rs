use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{config::Config, llm::create_text_generation};

use super::{generate_parameters::GenerateParameters, ErrorResponse};

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
    // Add other fields as necessary
}

/// Generate tokens
#[utoipa::path(
    post,
    path = "/generate",
    request_body = GenerateRequest,
    responses(
        (status = 200, description = "Generated Text", body = GenerateResponse),
        (status = 422, description = "Input validation error", body = ErrorResponse, example = json!(ErrorResponse {error:"Input validation error".to_string(), error_type: None })),
        (status = 424, description = "Generation Error", body = ErrorResponse, example = json!(ErrorResponse {error:"Request failed during generation".to_string(), error_type: None })),
        (status = 429, description = "Model is overloaded", body = ErrorResponse, example = json!(ErrorResponse {error:"Model is overloaded".to_string(), error_type: None })),
        (status = 500, description = "Incomplete generation", body = ErrorResponse, example = json!(ErrorResponse {error:"Incomplete generation".to_string(), error_type: None }))
    ),
    tag = "Text Generation Inference"
)]
pub async fn generate_text_handler(
    config: State<Config>,
    Json(payload): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, Json<ErrorResponse>)> {
    let temperature = match &payload.parameters {
        Some(parameters) => parameters.temperature,
        None => None,
    };
    let top_p: Option<f64> = match &payload.parameters {
        Some(parameters) => parameters.top_p,
        None => None,
    };
    let repeat_penalty: f32 = match &payload.parameters {
        Some(parameters) => parameters.repetition_penalty.unwrap_or(1.1),
        None => 1.1,
    };
    let repeat_last_n = match &payload.parameters {
        Some(parameters) => parameters.top_n_tokens.unwrap_or(64) as usize,
        None => 64,
    };
    let sample_len = match &payload.parameters {
        Some(parameters) => parameters.max_new_tokens.unwrap_or(50) as usize,
        None => 50,
    };

    let generator = create_text_generation(
        temperature,
        top_p,
        repeat_penalty,
        repeat_last_n,
        &config.cache_dir,
    );
    match generator {
        Ok(mut generator) => {
            let generated_text = generator.run(&payload.inputs, sample_len);
            match generated_text {
                Ok(generated_text) => match generated_text {
                    Some(text) => Ok(Json(GenerateResponse {
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
