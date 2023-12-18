// source: https://github.com/huggingface/candle/blob/main/candle-examples/examples/mistral/main.rs

use anyhow::{Error as E, Result};

use candle_core::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mistral::{Config, Model as Mistral};
use candle_transformers::models::quantized_mistral::Model as QMistral;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

pub enum Model {
    // unimplemented
    Mistral(Mistral),
    Quantized(QMistral),
}

pub struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<Option<String>> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the </s> token"),
        };
        let start_gen = std::time::Instant::now();
        let mut generated_text = String::new();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = match &mut self.model {
                Model::Mistral(m) => m.forward(&input, start_pos)?,
                Model::Quantized(m) => m.forward(&input, start_pos)?,
            };
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                generated_text.push_str(&t);

                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(Some(generated_text))
    }
}

pub fn create_model() -> Result<(Model, Device), Box<dyn std::error::Error>> {
    let model_id = "lmz/candle-mistral".to_string();
    let revision = "main".to_string();

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

    let filename = &repo.get("model-q4k.gguf")?;
    println!("retrieved the files in {:?}", start.elapsed());

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

pub fn create_text_generation() -> Result<TextGeneration, Box<dyn std::error::Error>> {
    let model = create_model()?;
    let tokenizer = create_tokenizer()?;

    let device = Device::Cpu;
    let seed: u64 = 299792458;
    let temperature: Option<f64> = None;
    let top_p: Option<f64> = None;
    let repeat_penalty: f32 = 1.1;
    let repeat_last_n: usize = 64;

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
