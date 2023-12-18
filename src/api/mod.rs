pub mod generate;
pub mod health;
pub mod openapi;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ErrorResponse {
    error: String,
    error_type: String,
}
