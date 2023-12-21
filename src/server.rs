use axum::{
    routing::{get, post},
    Router,
};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::{
    api::{
        openapi::ApiDoc,
        routes::{generate_handler, generate_stream_handler, generate_text_handler},
        routes::{get_health_handler, get_info_handler},
    },
    config::Config,
};

pub fn server(config: Config) -> Router {
    let router = Router::new()
        .route("/", post(generate_handler))
        .route("/generate", post(generate_text_handler))
        .route("/health", get(get_health_handler))
        .route("/info", get(get_info_handler))
        .route("/generate_stream", post(generate_stream_handler))
        .with_state(config);

    let swagger_ui = SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi());

    router.merge(swagger_ui)
}
