use utoipa::OpenApi;

use crate::api::{
    generate::{TextGenerationRequest, TextGenerationResponse},
    ErrorResponse,
};

#[derive(OpenApi)]
#[openapi(
    paths(super::generate::generate_text_handler, super::health::health_check),
    components(
        schemas(TextGenerationRequest, TextGenerationResponse, ErrorResponse)
    ),
    tags((name = "Text Generation Inference", description = "Text generation Inference API"))
)]
pub struct ApiDoc;
