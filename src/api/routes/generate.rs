use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};

use crate::{
    api::model::{CompatGenerateRequest, ErrorResponse, GenerateRequest},
    config::Config,
};

use super::generate_stream::generate_stream_handler;

/// Handler for generating text tokens.
///
/// This endpoint accepts a `CompatGenerateRequest` and returns a stream of generated text.
/// It requires the `stream` field in the request to be true. If `stream` is false,
/// the handler will return a `StatusCode::NOT_IMPLEMENTED` error.
///
/// # Arguments
/// * `config` - State containing the application configuration.
/// * `payload` - JSON payload containing the input text and optional parameters.
///
/// # Responses
/// * `200 OK` - Successful generation of text, returns a stream of `StreamResponse`.
/// * `501 Not Implemented` - Returned if `stream` field in request is false.
#[utoipa::path(
    post,
    path = "/",
    request_body = CompatGenerateRequest,
    responses(
        (status = 200, description = "Generated Text", body = StreamResponse),
        (status = 501, description = "Streaming not enabled", body = ErrorResponse),
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
        let app = Router::new()
            .route("/", post(generate_handler))
            .with_state(Config::default());

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
    async fn test_generate_handler_stream_disabled() {
        let app = Router::new()
            .route("/", post(generate_handler))
            .with_state(Config::default());

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

        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    }
}
