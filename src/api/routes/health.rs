//! This module contains the endpoint for the health check of the server.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

use crate::api::model::ErrorResponse;

/// Health check endpoint.
///
/// This endpoint checks the health of the server. It returns a success response if the server is healthy,
/// otherwise, it returns an error response indicating that the server is unhealthy.
#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Everything is working fine"),
        (status = 503, description = "Text generation inference is down", body = ErrorResponse, example = json!(ErrorResponse { error: String::from("unhealthy"), error_type: Some(String::from("healthcheck")) })),
    ),
    tag = "Text Generation Inference"
)]
pub async fn get_health_handler() -> impl IntoResponse {
    if check_server_health() {
        // Server is healthy
        Response::builder()
            .status(StatusCode::OK)
            .body("Everything is working fine".into())
            .unwrap()
    } else {
        // Server is unhealthy
        let error_response = ErrorResponse {
            error: "unhealthy".into(),
            error_type: Some("healthcheck".into()),
        };

        Json(error_response).into_response()
    }
}

fn check_server_health() -> bool {
    true // Dummy implementation, always returns true for now.
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::{Response, StatusCode},
    };

    /// Tests the `get_health_handler` function to ensure it returns the correct status.
    #[tokio::test]
    async fn test_get_health_handler() {
        let response: Response<Body> = get_health_handler().await.into_response();

        // Assertions to verify the correctness of the response.
        assert_eq!(response.status(), StatusCode::OK);

        // Extract the body from the response. Set a reasonable limit for the body size.
        let body_limit = 1024 * 1024; // 1 MB
        let body_bytes = to_bytes(response.into_body(), body_limit).await.unwrap();
        let body = String::from_utf8(body_bytes.to_vec()).unwrap();

        assert_eq!(body, "Everything is working fine");
    }
}
