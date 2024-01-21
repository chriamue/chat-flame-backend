use crate::{
    api::model::{ErrorResponse, GenerateRequest, GenerateResponse},
    llm::{generate_parameter::GenerateParameter, text_generation::create_text_generation},
    server::AppState,
};
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};

/// Asynchronous handler for generating text.
///
/// This function handles POST requests to the `/generate` endpoint. It takes a JSON payload
/// representing a `GenerateRequest` and uses the configuration and parameters specified to
/// generate text. The generated text is returned in a `GenerateResponse` if successful.
///
/// # Parameters
/// - `config`: Application state holding the global configuration.
/// - `Json(payload)`: JSON payload containing the input text and generation parameters.
///
/// # Responses
/// - `200 OK`: Successful text generation with `GenerateResponse`.
/// - `422 Unprocessable Entity`: Input validation error with `ErrorResponse`.
/// - `424 Failed Dependency`: Generation error with `ErrorResponse`.
/// - `429 Too Many Requests`: Model is overloaded with `ErrorResponse`.
/// - `500 Internal Server Error`: Incomplete generation with `ErrorResponse`.
///
/// # Usage
/// This endpoint is suitable for generating text based on given prompts and parameters.
/// It can be used in scenarios where batch text generation is required, such as content
/// creation, language modeling, or any application needing on-demand text generation.
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
    app_state: State<AppState>,
    Json(payload): Json<GenerateRequest>,
) -> impl IntoResponse {
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

    let config = app_state.config.clone();

    let mut generator = match &app_state.text_generation {
        Some(text_generation) => text_generation.clone(),
        None => create_text_generation(config.model, &config.cache_dir).unwrap(),
    };

    let parameter = GenerateParameter {
        temperature: temperature.unwrap_or_default(),
        top_p: top_p.unwrap_or_default(),
        max_new_tokens: sample_len,
        seed: 42,
        repeat_penalty,
        repeat_last_n,
    };

    let generated_text = generator.run(&payload.inputs, parameter);
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
