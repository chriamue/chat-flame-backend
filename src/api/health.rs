use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

use super::ErrorResponse;

/// Health check endpoint
#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Everything is working fine"),
        (status = 503, description = "Text generation inference is down", body = ErrorResponse, example = json!(ErrorResponse { error: String::from("unhealthy"), error_type: String::from("healthcheck") })),
    )
)]
pub async fn health_check() -> impl IntoResponse {
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
            error_type: "healthcheck".into(),
        };

        Json(error_response).into_response()
    }
}

fn check_server_health() -> bool {
    true // Dummy implementation, always returns true for now.
}
