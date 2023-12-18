use serde::{Serialize, Deserialize};

#[derive(Deserialize)]
pub struct TextGenerationRequest {
    pub prompt: String,
    // Add other fields as defined in the API
}

#[derive(Serialize)]
pub struct TextGenerationResponse {
    pub generated_text: String,
    // Add other fields as necessary
}
