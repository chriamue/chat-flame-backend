use super::model::{
    CompatGenerateRequest, FinishReason, GenerateParameters, GenerateRequest, GenerateResponse,
    Info, StreamDetails, StreamResponse, Token,
};
use crate::{api::model::ErrorResponse, llm::models::Models};
use utoipa::OpenApi;

/// Represents the API documentation for the text generation inference service.
///
/// This struct uses `utoipa::OpenApi` to provide a centralized documentation of the API endpoints
/// and their associated request and response models. It is used to generate OpenAPI specification
/// for the service, which can be served as a Swagger UI or other OpenAPI-compatible documentation tools.
#[derive(OpenApi)]
#[openapi(
    // List of API endpoints to be included in the documentation.
    paths(
        super::routes::generate::generate_handler,
        super::routes::generate_text::generate_text_handler,
        super::routes::generate_stream::generate_stream_handler,
        super::routes::model::generate_model_handler,
        super::routes::health::get_health_handler,
        super::routes::info::get_info_handler
    ),
    // Schema components for requests and responses used across the API.
    components(
        schemas(
            CompatGenerateRequest,
            GenerateRequest,
            GenerateResponse,
            GenerateParameters,
            ErrorResponse,
            StreamResponse,
            StreamDetails,
            Token,
            FinishReason,
            Info,
            Models
        )
    ),
    // Metadata and description of the API tags.
    tags(
        (name = "Text Generation Inference", description = "Text generation Inference API")
    )
)]
pub struct ApiDoc;

#[cfg(test)]
mod tests {
    use super::*;
    use utoipa::OpenApi;

    #[test]
    fn api_doc_contains_all_endpoints() {
        let api_doc = ApiDoc::openapi();
        let paths = api_doc.paths.paths;
        assert!(paths.contains_key("/"));
        assert!(paths.contains_key("/generate"));
        assert!(paths.contains_key("/generate_stream"));
        assert!(paths.contains_key("/health"));
        assert!(paths.contains_key("/info"));
    }
}
