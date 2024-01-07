//! Model Loader Module.
//!
//! This module contains functions for loading model weights and tokenizers for text generation.
//! It supports various models and uses the Hugging Face Hub for downloading model files.

use std::path::PathBuf;

use crate::llm::Model;

use super::models::Models;
use anyhow::{Error as E, Result};
use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::Device;
use candle_transformers::models::quantized_llama::ModelWeights;
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use log::{debug, info};
use tokenizers::Tokenizer;

/// Formats the size in bytes into a human-readable string.
fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}

/// Creates and loads model weights from the Hugging Face Hub.
///
/// # Arguments
///
/// * `model` - The model enum specifying the model to load.
/// * `cache_dir` - Optional directory for caching downloaded models.
///
/// # Returns
///
/// Returns a result containing a tuple of `ModelWeights` and `Device`,
/// or an error if loading fails.
pub fn create_model(
    model: Models,
    cache_dir: &Option<PathBuf>,
) -> Result<(Model, Device), Box<dyn std::error::Error>> {
    info!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    let revision = "main".to_string();

    let start = std::time::Instant::now();
    let api = match cache_dir {
        Some(cache_dir) => ApiBuilder::default()
            .with_cache_dir(cache_dir.clone())
            .build()?,
        None => Api::new()?,
    };

    let model_path = model.repo_path();

    debug!("model paths: {:?}", model_path);

    let repo = api.repo(Repo::with_revision(
        model_path.0.to_string(),
        RepoType::Model,
        revision,
    ));

    let model_path = &repo.get(model_path.1)?;
    let mut file = std::fs::File::open(model_path)?;
    info!("retrieved the model files in {:?}", start.elapsed());

    let model = match model_path.extension().and_then(|v| v.to_str()) {
        Some("gguf") => {
            let content = gguf_file::Content::read(&mut file)?;
            let mut total_size_in_bytes = 0;
            for (_, tensor) in content.tensor_infos.iter() {
                let elem_count = tensor.shape.elem_count();
                total_size_in_bytes +=
                    elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.blck_size();
            }
            debug!(
                "loaded {:?} tensors ({}) in {:.2}s",
                content.tensor_infos.len(),
                &format_size(total_size_in_bytes),
                start.elapsed().as_secs_f32(),
            );
            match model {
                Models::L7b
                | Models::L13b
                | Models::L7bChat
                | Models::L13bChat
                | Models::L7bCode
                | Models::L13bCode
                | Models::L34bCode
                | Models::Leo7b
                | Models::Leo13b => Model::Llama(ModelWeights::from_gguf(content, &mut file)?),
                Models::Mixtral
                | Models::MixtralInstruct
                | Models::Mistral7b
                | Models::Mistral7bInstruct
                | Models::Zephyr7bAlpha
                | Models::Zephyr7bBeta
                | Models::L70b
                | Models::L70bChat
                | Models::OpenChat35
                | Models::Starling7bAlpha => {
                    Model::Llama(ModelWeights::from_gguf(content, &mut file)?)
                }
                Models::PhiV1 | Models::PhiV1_5 | Models::PhiV2 | Models::PhiHermes => {
                    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                        model_path,
                    )?;
                    Model::MixFormer(candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM::new_v2(&candle_transformers::models::mixformer::Config::v2(), vb)?)
                }
            }
        }
        Some("ggml" | "bin") | Some(_) | None => {
            let content = ggml_file::Content::read(&mut file)?;
            let mut total_size_in_bytes = 0;
            for (_, tensor) in content.tensors.iter() {
                let elem_count = tensor.shape().elem_count();
                total_size_in_bytes +=
                    elem_count * tensor.dtype().type_size() / tensor.dtype().blck_size();
            }
            debug!(
                "loaded {:?} tensors ({}) in {:.2}s",
                content.tensors.len(),
                &format_size(total_size_in_bytes),
                start.elapsed().as_secs_f32(),
            );
            debug!("params: {:?}", content.hparams);
            let default_gqa = match model {
                Models::L7b
                | Models::L13b
                | Models::L7bChat
                | Models::L13bChat
                | Models::L7bCode
                | Models::L13bCode
                | Models::L34bCode
                | Models::Leo7b
                | Models::Leo13b => 1,
                Models::Mixtral
                | Models::MixtralInstruct
                | Models::Mistral7b
                | Models::Mistral7bInstruct
                | Models::Zephyr7bAlpha
                | Models::Zephyr7bBeta
                | Models::L70b
                | Models::L70bChat
                | Models::OpenChat35
                | Models::Starling7bAlpha => 8,
                Models::PhiHermes | Models::PhiV1 | Models::PhiV1_5 | Models::PhiV2 => 4,
            };

            match model {
                Models::L7b
                | Models::L13b
                | Models::L7bChat
                | Models::L13bChat
                | Models::L7bCode
                | Models::L13bCode
                | Models::L34bCode
                | Models::Leo7b
                | Models::Leo13b => Model::Llama(ModelWeights::from_ggml(content, default_gqa)?),
                Models::Mixtral
                | Models::MixtralInstruct
                | Models::Mistral7b
                | Models::Mistral7bInstruct
                | Models::Zephyr7bAlpha
                | Models::Zephyr7bBeta
                | Models::L70b
                | Models::L70bChat
                | Models::OpenChat35
                | Models::Starling7bAlpha => {
                    Model::Llama(ModelWeights::from_ggml(content, default_gqa)?)
                }
                Models::PhiV1 | Models::PhiV1_5 | Models::PhiV2 | Models::PhiHermes => {
                    Model::Llama(ModelWeights::from_ggml(content, default_gqa)?)
                }
            }
        }
    };
    Ok((model, Device::Cpu))
}

/// Creates and loads a tokenizer from the Hugging Face Hub.
///
/// # Arguments
///
/// * `model` - The model enum specifying the tokenizer to load.
///
/// # Returns
///
/// Returns a result containing the `Tokenizer`,
/// or an error if loading fails.
pub fn create_tokenizer(model: Models) -> Result<Tokenizer, Box<dyn std::error::Error>> {
    let tokenizer_path = {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = model.tokenizer_repo();
        let api = api.model(repo.to_string());
        api.get("tokenizer.json")?
    };
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(999), "999B");
        assert_eq!(format_size(1000), "1.00KB");
        assert_eq!(format_size(1000000), "1.00MB");
        assert_eq!(format_size(1000000000), "1.00GB");
    }
}
