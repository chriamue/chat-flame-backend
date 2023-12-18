use axum::{response::IntoResponse, Json};
use axum_streams::StreamBodyAs;
use futures::{stream, Stream};
use tokio_stream::StreamExt;

use super::{generate::GenerateRequest, stream_response::StreamResponse};

fn source_test_stream() -> impl Stream<Item = StreamResponse> {
    // Simulating a stream with a plain vector and throttling to show how it works
    stream::iter(vec![
        StreamResponse {
            generated_text: "hello world".to_string()
        };
        100
    ])
    .throttle(std::time::Duration::from_millis(50))
}

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
    StreamBodyAs::json_nl(source_test_stream())
}
