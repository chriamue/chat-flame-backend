use crate::api::model::GenerateRequest;
use crate::{config::Config, llm::create_text_generation};
use axum::{
    extract::State,
    response::{sse::Event, IntoResponse, Sse},
    Json,
};
use futures::stream::StreamExt;
use log::debug;
use std::vec;

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
pub async fn generate_stream_handler(
    config: State<Config>,
    Json(payload): Json<GenerateRequest>,
) -> impl IntoResponse {
    debug!("Received request: {:?}", payload);
    let temperature = match &payload.parameters {
        Some(parameters) => parameters.temperature,
        None => None,
    };
    let top_p: Option<f64> = match &payload.parameters {
        Some(parameters) => parameters.top_p,
        None => None,
    };
    let repeat_penalty: f32 = match &payload.parameters {
        Some(parameters) => parameters.repetition_penalty.unwrap_or(1.1),
        None => 1.1,
    };
    let repeat_last_n = match &payload.parameters {
        Some(parameters) => parameters.top_n_tokens.unwrap_or(64) as usize,
        None => 64,
    };
    let sample_len = match &payload.parameters {
        Some(parameters) => parameters.max_new_tokens.unwrap_or(50) as usize,
        None => 50,
    };

    let stop_tokens = match &payload.parameters {
        Some(parameters) => parameters.stop.clone(),
        None => vec!["</s>".to_string()],
    };

    let mut generator = create_text_generation(
        config.model,
        temperature,
        top_p,
        repeat_penalty,
        repeat_last_n,
        &config.cache_dir,
    )
    .unwrap();
    let stream = generator.run_stream(&payload.inputs, sample_len, Some(stop_tokens));

    let event_stream = stream.map(|response| -> Result<Event, std::convert::Infallible> {
        let data = serde_json::to_string(&response)
            .unwrap_or_else(|_| "Error serializing response".to_string());
        Ok(Event::default().data(data))
    });
    Sse::new(event_stream)
}
