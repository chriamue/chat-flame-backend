use axum::{
    routing::{get, post},
    Router,
};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::{
    api::{
        generate::generate_handler, generate_stream::generate_stream_handler,
        generate_text::generate_text_handler, health::health_check, openapi::ApiDoc,
    },
    config::Config,
};

pub fn server(config: Config) -> Router {
    let router = Router::new()
        .route("/", post(generate_handler))
        .route("/generate", post(generate_text_handler))
        .route("/health", get(health_check))
        .route("/generate_stream", post(generate_stream_handler))
        .with_state(config);

    let swagger_ui = SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi());

    router.merge(swagger_ui)
}
