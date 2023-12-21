use crate::{api::model::Info, config::Config};
use axum::{extract::State, http::StatusCode, Json};

/// Endpoint to get model information
#[utoipa::path(
    get,
    path = "/info",
    responses(
        (status = 200, description = "Served model info", body = Info),
    ),
    tag = "Text Generation Inference"
)]
pub async fn info_handler(config: State<Config>) -> Result<Json<Info>, StatusCode> {
    let version = env!("CARGO_PKG_VERSION");
    let model_info = Info {
        docker_label: None,
        max_batch_total_tokens: 2048,
        max_best_of: 1,
        max_concurrent_requests: 1,
        max_input_length: 1024,
        max_stop_sequences: 4,
        max_total_tokens: 2048,
        max_waiting_tokens: 32,
        model_device_type: "cpu".to_string(),
        model_dtype: "float16".to_string(),
        model_id: config.model.tokenizer_repo().to_string(),
        model_pipeline_tag: Some("text-generation".to_string()),
        model_sha: None,
        sha: None,
        validation_workers: 2,
        version: version.to_string(),
        waiting_served_ratio: 1.2,
    };

    Ok(Json(model_info))
}
