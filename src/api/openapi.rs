use utoipa::OpenApi;

use crate::api::{
    generate_text::{GenerateRequest, GenerateResponse},
    ErrorResponse,
};

use super::{
    generate::CompatGenerateRequest,
    generate_parameters::GenerateParameters,
    stream_response::{FinishReason, StreamDetails, StreamResponse, Token},
};

#[derive(OpenApi)]
#[openapi(
    paths(super::generate_text::generate_text_handler, super::generate_stream::generate_stream_handler, super::health::health_check),
    components(
        schemas(CompatGenerateRequest, GenerateRequest, GenerateResponse, GenerateParameters, ErrorResponse, StreamResponse,
            StreamDetails, Token, FinishReason)
    ),
    tags((name = "Text Generation Inference", description = "Text generation Inference API"))
)]
pub struct ApiDoc;
