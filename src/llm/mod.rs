// source: https://github.com/huggingface/candle/blob/main/candle-examples/examples/mistral/main.rs

pub mod dummy_text_generator;
pub mod models;
mod text_generation;
pub mod text_generator;

pub use dummy_text_generator::DummyTextGenerator;
pub use text_generator::TextGenerator;

use std::path::PathBuf;

use anyhow::{Error as E, Result};
use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::Device;
use candle_transformers::models::quantized_llama::ModelWeights;
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use log::info;
pub use text_generation::TextGeneration;
use tokenizers::Tokenizer;

use self::models::Models;

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

pub fn create_model(
    model: Models,
    cache_dir: &Option<PathBuf>,
) -> Result<(ModelWeights, Device), Box<dyn std::error::Error>> {
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

    let repo = api.repo(Repo::with_revision(
        model_path.0.to_string(),
        RepoType::Model,
        revision,
    ));

    let model_path = &repo.get(model_path.1)?;
    let mut file = std::fs::File::open(&model_path)?;
    info!("retrieved the model files in {:?}", start.elapsed());

    let model = match model_path.extension().and_then(|v| v.to_str()) {
        Some("gguf") => {
            let model = gguf_file::Content::read(&mut file)?;
            let mut total_size_in_bytes = 0;
            for (_, tensor) in model.tensor_infos.iter() {
                let elem_count = tensor.shape.elem_count();
                total_size_in_bytes +=
                    elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.blck_size();
            }
            println!(
                "loaded {:?} tensors ({}) in {:.2}s",
                model.tensor_infos.len(),
                &format_size(total_size_in_bytes),
                start.elapsed().as_secs_f32(),
            );
            ModelWeights::from_gguf(model, &mut file)?
        }
        Some("ggml" | "bin") | Some(_) | None => {
            let content = ggml_file::Content::read(&mut file)?;
            let mut total_size_in_bytes = 0;
            for (_, tensor) in content.tensors.iter() {
                let elem_count = tensor.shape().elem_count();
                total_size_in_bytes +=
                    elem_count * tensor.dtype().type_size() / tensor.dtype().blck_size();
            }
            println!(
                "loaded {:?} tensors ({}) in {:.2}s",
                content.tensors.len(),
                &format_size(total_size_in_bytes),
                start.elapsed().as_secs_f32(),
            );
            println!("params: {:?}", content.hparams);
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
            };
            ModelWeights::from_ggml(content, default_gqa)?
        }
    };
    Ok((model, Device::Cpu))
}

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

pub fn create_text_generation(
    model: Models,
    temperature: Option<f64>,
    top_p: Option<f64>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    cache_dir: &Option<PathBuf>,
) -> Result<TextGeneration, Box<dyn std::error::Error>> {
    let tokenizer = create_tokenizer(model)?;
    let model = create_model(model, cache_dir)?;

    let device = Device::Cpu;
    let seed: u64 = 299792458;

    Ok(TextGeneration::new(
        model.0,
        tokenizer,
        seed,
        temperature,
        top_p,
        repeat_penalty,
        repeat_last_n,
        &device,
    ))
}
