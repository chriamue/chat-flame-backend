use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

use crate::{
    api::model::{CompatGenerateRequest, ErrorResponse, GenerateRequest},
    server::AppState,
};

use super::{generate_stream::generate_stream_handler, generate_text_handler};

/// Handler for generating text tokens.
///
/// This endpoint accepts a `CompatGenerateRequest` and returns a stream of generated text
/// or a single text response based on the `stream` field in the request. If `stream` is true,
/// it returns a stream of `StreamResponse`. If `stream` is false, it returns `GenerateResponse`.
///
/// # Arguments
/// * `config` - State containing the application configuration.
/// * `payload` - JSON payload containing the input text and optional parameters.
///
/// # Responses
/// * `200 OK` - Successful generation of text.
/// * `501 Not Implemented` - Returned if streaming is not implemented.
#[utoipa::path(
    post,
    tag = "Text Generation Inference",
    path = "/",
    request_body = CompatGenerateRequest,
    responses(
        (status = 200, description = "Generated Text",
         content(
             ("application/json" = GenerateResponse),
             ("text/event-stream" = StreamResponse),
         )
        ),
        (status = 424, description = "Generation Error", body = ErrorResponse,
         example = json!({"error": "Request failed during generation"})),
        (status = 429, description = "Model is overloaded", body = ErrorResponse,
         example = json!({"error": "Model is overloaded"})),
        (status = 422, description = "Input validation error", body = ErrorResponse,
         example = json!({"error": "Input validation error"})),
        (status = 500, description = "Incomplete generation", body = ErrorResponse,
         example = json!({"error": "Incomplete generation"})),
    )
)]
pub async fn generate_handler(
    app_state: State<AppState>,
    Json(payload): Json<CompatGenerateRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    if payload.stream {
        Ok(generate_stream_handler(
            app_state,
            Json(GenerateRequest {
                inputs: payload.inputs,
                parameters: payload.parameters,
            }),
        )
        .await
        .into_response())
    } else {
        Ok(generate_text_handler(
            app_state,
            Json(GenerateRequest {
                inputs: payload.inputs,
                parameters: payload.parameters,
            }),
        )
        .await
        .into_response())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        routing::post,
        Router,
    };
    use serde_json::json;
    use tower::ServiceExt; // for `oneshot` method

    /// Test the generate_handler function for streaming enabled.
    #[ignore = "Will download model from HuggingFace"]
    #[tokio::test]
    async fn test_generate_handler_stream_enabled() {
        let state = AppState {
            config: Config::default(),
            text_generation: None,
        };
        let app = Router::new()
            .route("/", post(generate_handler))
            .with_state(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "inputs": "Hello, world!",
                            "stream": true
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    /// Test the generate_handler function for streaming disabled.
    #[tokio::test]
    #[ignore = "Will download model from HuggingFace"]
    async fn test_generate_handler_stream_disabled() {
        let state = AppState {
            config: Config::default(),
            text_generation: None,
        };
        let app = Router::new()
            .route("/", post(generate_handler))
            .with_state(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "inputs": "Hello, world!",
                            "stream": false
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
