//! This module contains the endpoint for retrieving model information.

use crate::{api::model::Info, server::AppState};
use axum::{extract::State, http::StatusCode, Json};

/// Endpoint to get model information.
///
/// This endpoint provides detailed information about the model used in the application,
/// including its configuration and capabilities.
#[utoipa::path(
    get,
    path = "/info",
    responses(
        (status = 200, description = "Served model info", body = Info),
    ),
    tag = "Text Generation Inference"
)]
pub async fn get_info_handler(app_state: State<AppState>) -> Result<Json<Info>, StatusCode> {
    let config = &app_state.config;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Config, llm::models::Models};

    #[tokio::test]
    async fn test_get_info_handler() {
        let test_config = Config {
            port: 8080,
            cache_dir: None,
            model: Models::default(),
            keep_in_memory: None,
        };

        let state = State(AppState {
            config: test_config.clone(),
            text_generation: None,
        });
        let response = get_info_handler(state).await.unwrap();
        let info = response.0;
        assert_eq!(info.max_batch_total_tokens, 2048);
        assert_eq!(info.max_best_of, 1);
        assert_eq!(info.max_concurrent_requests, 1);
        assert_eq!(info.max_input_length, 1024);
        assert_eq!(info.max_stop_sequences, 4);
        assert_eq!(info.max_total_tokens, 2048);
        assert_eq!(info.max_waiting_tokens, 32);
        assert_eq!(info.model_device_type, "cpu");
        assert_eq!(info.model_dtype, "float16");
        assert_eq!(info.model_id, test_config.model.tokenizer_repo());
    }
}
