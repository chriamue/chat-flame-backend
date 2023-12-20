// source: https://github.com/huggingface/candle/blob/main/candle-examples/examples/mistral/main.rs

mod text_generation;
use std::path::PathBuf;

use anyhow::{Error as E, Result};
use candle_core::Device;
use candle_transformers::models::mistral::{Config, Model as Mistral};
use candle_transformers::models::quantized_mistral::Model as QMistral;
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use log::info;
pub use text_generation::TextGeneration;
use tokenizers::Tokenizer;

pub enum Model {
    // unimplemented
    Mistral(Mistral),
    Quantized(QMistral),
}

pub fn create_model(
    cache_dir: &Option<PathBuf>,
) -> Result<(Model, Device), Box<dyn std::error::Error>> {
    info!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    let model_id = "lmz/candle-mistral".to_string();
    let revision = "main".to_string();

    let start = std::time::Instant::now();
    let api = match cache_dir {
        Some(cache_dir) => ApiBuilder::default()
            .with_cache_dir(cache_dir.clone())
            .build()?,
        None => Api::new()?,
    };
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

    let filename = &repo.get("model-q4k.gguf")?;
    info!("retrieved the model files in {:?}", start.elapsed());

    let config = Config::config_7b_v0_1(false);

    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(filename)?;
    let model = QMistral::new(&config, vb)?;
    Ok((Model::Quantized(model), Device::Cpu))
}

pub fn create_tokenizer() -> Result<Tokenizer, Box<dyn std::error::Error>> {
    let model_id = "lmz/candle-mistral".to_string();
    let revision = "main".to_string();
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    Ok(tokenizer)
}

pub fn create_text_generation(
    temperature: Option<f64>,
    top_p: Option<f64>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    cache_dir: &Option<PathBuf>,
) -> Result<TextGeneration, Box<dyn std::error::Error>> {
    let model = create_model(cache_dir)?;
    let tokenizer = create_tokenizer()?;

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
