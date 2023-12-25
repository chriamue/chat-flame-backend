use crate::llm::generate_parameter::GenerateParameter;
use crate::llm::text_generation::create_text_generation;
use crate::{api::model::GenerateRequest, config::Config};
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
        None => vec!["<|endoftext|>".to_string(), "</s>".to_string()],
    };

    let mut generator = create_text_generation(config.model, &config.cache_dir).unwrap();

    let parameter = GenerateParameter {
        temperature: temperature.unwrap_or_default(),
        top_p: top_p.unwrap_or_default(),
        max_new_tokens: sample_len,
        seed: 42,
        repeat_penalty,
        repeat_last_n,
    };

    let stream = generator.run_stream(&payload.inputs, parameter, Some(stop_tokens));

    let event_stream = stream.map(|response| -> Result<Event, std::convert::Infallible> {
        let data = serde_json::to_string(&response)
            .unwrap_or_else(|_| "Error serializing response".to_string());
        Ok(Event::default().data(data))
    });
    Sse::new(event_stream)
}
