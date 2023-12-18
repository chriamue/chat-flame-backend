use utoipa::OpenApi;

use crate::api::{
    generate::{TextGenerationRequest, TextGenerationResponse},
    ErrorResponse,
};

use super::generate_parameters::GenerateParameters;

#[derive(OpenApi)]
#[openapi(
    paths(super::generate::generate_text_handler, super::health::health_check),
    components(
        schemas(TextGenerationRequest, TextGenerationResponse, GenerateParameters, ErrorResponse)
    ),
    tags((name = "Text Generation Inference", description = "Text generation Inference API"))
)]
pub struct ApiDoc;
