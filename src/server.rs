use axum::{
    routing::{get, post},
    Router,
};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::api::{generate::generate_text_handler, health::health_check, openapi::ApiDoc};

pub fn server() -> Router {
    Router::new()
        .route("/generate-text", post(generate_text_handler))
        .route("/health", get(health_check))
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
}
