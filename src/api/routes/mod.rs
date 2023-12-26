/// Module containing all route handlers.
///
/// This module organizes the different API endpoints and their associated handlers.
/// Each route corresponds to a specific functionality of the text generation inference API.
///
/// # Modules
/// * `generate` - Handles requests for token generation with streaming capability.
/// * `generate_stream` - Handles streaming requests for text generation.
/// * `generate_text` - Handles requests for generating text without streaming.
/// * `health` - Provides a health check endpoint.
/// * `info` - Provides information about the text generation inference service.
pub mod generate; // Module for handling token generation with streaming.
pub mod generate_stream; // Module for handling streaming text generation requests.
pub mod generate_text; // Module for handling text generation requests.
pub mod health; // Module for the health check endpoint.
pub mod info; // Module for the service information endpoint.

// Public exports of route handlers for ease of access.
pub use generate::generate_handler;
pub use generate_stream::generate_stream_handler;
pub use generate_text::generate_text_handler;
pub use health::get_health_handler;
pub use info::get_info_handler;
