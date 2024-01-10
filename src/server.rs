use axum::{
    response::Redirect,
    routing::{get, post},
    Router,
};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::{
    api::{
        openapi::ApiDoc,
        routes::{
            generate_handler, generate_model_handler, generate_stream_handler,
            generate_text_handler,
        },
        routes::{get_health_handler, get_info_handler},
    },
    config::Config,
};

/// Creates and configures the Axum web server with various routes and Swagger UI.
///
/// This function sets up all the necessary routes for the API and merges them
/// with the Swagger UI for easy API documentation and testing.
///
/// # Arguments
///
/// * `config` - Configuration settings for the server.
///
/// # Returns
///
/// An instance of `axum::Router` configured with all routes and the Swagger UI.
pub fn server(config: Config) -> Router {
    let router = Router::new()
        .route("/", get(|| async { Redirect::permanent("/swagger-ui") }))
        .route("/", post(generate_handler))
        .route("/generate", post(generate_text_handler))
        .route("/health", get(get_health_handler))
        .route("/info", get(get_info_handler))
        .route("/generate_stream", post(generate_stream_handler))
        .route("/model/:model", post(generate_model_handler))
        .with_state(config);

    let swagger_ui = SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi());

    router.merge(swagger_ui)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::ServiceExt; // for `oneshot` function

    #[tokio::test]
    async fn test_root_redirects_to_swagger_ui() {
        let config = Config::default();
        let app = server(config);

        let req = Request::builder()
            .method("GET")
            .uri("/")
            .body(Body::empty())
            .unwrap();

        let response = app.clone().oneshot(req).await.unwrap();

        // Verify that the response is a redirect to /swagger-ui.
        assert_eq!(response.status().as_u16(), 308);
        assert_eq!(response.headers().get("location").unwrap(), "/swagger-ui");
    }

    #[tokio::test]
    async fn test_swagger_ui_endpoint() {
        let config = Config::default();
        let app = server(config);

        let req = Request::builder()
            .method("GET")
            .uri("/swagger-ui/index.html")
            .body(Body::empty())
            .unwrap();

        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
