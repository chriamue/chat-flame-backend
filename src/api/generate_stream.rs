use axum::{
    response::{sse::Event, IntoResponse, Sse},
    Json,
};
use futures::stream::StreamExt;

use crate::llm::create_text_generation;

use super::generate::GenerateRequest;

/// Generate tokens
#[utoipa::path(
    post,
    path = "/generate_stream",
    request_body = GenerateRequest,
    responses(
        (status = 200, description = "Generated Text", body = StreamResponse),
    ),
    tag = "Text Generation Inference"
)]
pub async fn generate_stream_handler(Json(request): Json<GenerateRequest>) -> impl IntoResponse {
    let mut generator = create_text_generation().unwrap();
    let stream = generator.run_stream(&request.inputs, 50);

    let event_stream = stream.map(|response| -> Result<Event, std::convert::Infallible> {
        let data = serde_json::to_string(&response)
            .unwrap_or_else(|_| "Error serializing response".to_string());
        Ok(Event::default().data(data))
    });
    Sse::new(event_stream)
}
