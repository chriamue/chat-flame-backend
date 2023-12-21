use utoipa::OpenApi;

use crate::api::model::ErrorResponse;

use super::model::{
    CompatGenerateRequest, FinishReason, GenerateParameters, GenerateRequest, GenerateResponse,
    Info, StreamDetails, StreamResponse, Token,
};

#[derive(OpenApi)]
#[openapi(
    paths(super::generate_text::generate_text_handler, super::generate_stream::generate_stream_handler, super::routes::health::get_health_handler, super::routes::info::get_info_handler),
    components(
        schemas(CompatGenerateRequest, GenerateRequest, GenerateResponse, GenerateParameters, ErrorResponse, StreamResponse,
            StreamDetails, Token, FinishReason, Info)
    ),
    tags((name = "Text Generation Inference", description = "Text generation Inference API"))
)]
pub struct ApiDoc;
