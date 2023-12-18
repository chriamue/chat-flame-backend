pub mod openapi;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Deserialize, ToSchema)]
pub struct TextGenerationRequest {
    pub prompt: String,
    // Add other fields as defined in the API
}

#[derive(Serialize, ToSchema)]
pub struct TextGenerationResponse {
    pub generated_text: String,
    // Add other fields as necessary
}
