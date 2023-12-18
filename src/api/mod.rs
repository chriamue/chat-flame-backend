pub mod generate;
pub mod generate_parameters;
pub mod generate_stream;
pub mod health;
pub mod openapi;
pub mod stream_response;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ErrorResponse {
    error: String,
    error_type: Option<String>,
}
