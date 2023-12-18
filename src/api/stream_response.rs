use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct StreamResponse {
    // Define the fields according to your schema
    pub generated_text: String,
}
