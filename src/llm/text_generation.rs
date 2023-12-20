use crate::{
    api::model::{FinishReason, StreamDetails, StreamResponse, Token},
    llm::token_generator::TokenGenerator,
};

use anyhow::{Error as E, Result};
use candle_core::Device;
use candle_transformers::{generation::LogitsProcessor, models::quantized_llama::ModelWeights};
use futures::Stream;
use log::{debug, info, trace};
use rayon::prelude::*;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;

use super::token_output_stream::TokenOutputStream;

pub struct TextGeneration {
    model: Arc<Mutex<ModelWeights>>,
    tokenizer: Arc<Mutex<TokenOutputStream>>,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: ModelWeights,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        _device: &Device,
    ) -> Self {
        let _logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(Mutex::new(TokenOutputStream::new(tokenizer))),
            repeat_penalty,
            repeat_last_n,
        }
    }

    fn process_tokens<F>(
        tokenizer: &Mutex<TokenOutputStream>,
        model: &Mutex<ModelWeights>,
        prompt: String,
        sample_len: usize,
        stop_tokens: Option<Vec<String>>,
        mut handle_token: F,
    ) -> Result<()>
    where
        F: FnMut(String, usize, bool) -> Result<()>,
    {
        let mut tokenizer = tokenizer.try_lock().unwrap();
        let mut model = model.try_lock().unwrap();
        let mut generated_tokens = vec![];

        tokenizer.clear();
        let binding = tokenizer.tokenizer().encode(prompt, true).map_err(E::msg)?;
        generated_tokens.extend(binding.get_ids().to_vec());

        let repeat_penalty = 1.1;
        let repeat_last_n = 64;
        let mut logits_processor = LogitsProcessor::new(42, None, None);
        let start_gen = std::time::Instant::now();

        let mut token_generator = TokenGenerator::new();
        token_generator.set_stop_tokens(stop_tokens, &mut tokenizer);

        for _ in 0..sample_len {
            let next_token = {
                token_generator.next(
                    &generated_tokens,
                    &mut logits_processor,
                    &mut model,
                    repeat_penalty,
                    repeat_last_n,
                )?
            };

            if let Some(token) = next_token {
                generated_tokens.push(token);
                if token_generator.is_stop_token(&token) {
                    break;
                }
            }
        }

        let generated_text: Vec<String> = generated_tokens
            .par_iter()
            .filter_map(|&token| tokenizer.decode(&[token]).ok())
            .collect();

        for (index, text) in generated_text.iter().enumerate() {
            handle_token(text.clone(), index, false)?;
        }

        info!(
            "{} tokens generated ({:.2} token/s)",
            generated_tokens.len(),
            generated_tokens.len() as f64 / start_gen.elapsed().as_secs_f64(),
        );

        Ok(())
    }
    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<Option<String>> {
        let mut generated_text = String::new();
        Self::process_tokens(
            &self.tokenizer,
            &self.model,
            prompt.to_string(),
            sample_len,
            None,
            |text, _, is_final| {
                if is_final {
                    Ok(())
                } else {
                    generated_text.push_str(&text);
                    Ok(())
                }
            },
        )?;
        Ok(Some(generated_text))
    }

    pub fn run_stream(
        &mut self,
        prompt: &str,
        sample_len: usize,
        stop_tokens: Option<Vec<String>>,
    ) -> impl Stream<Item = StreamResponse> {
        debug!("Running stream for {}: {}", sample_len, prompt);
        let (tx, rx) = tokio::sync::mpsc::channel(32);

        let tokenizer = self.tokenizer.clone();
        let model = self.model.clone();
        let prompt = prompt.to_string();
        let sample_len = sample_len as usize;
        let repeat_last_n = self.repeat_last_n;
        let repeat_penalty = self.repeat_penalty;

        tokio::spawn(async move {
            let mut tokenizer = tokenizer.try_lock().unwrap();
            let mut model = model.try_lock().unwrap();
            tokenizer.clear();
            let mut tokens = tokenizer
                .tokenizer()
                .encode(prompt, true)
                .map_err(E::msg)
                .unwrap()
                .get_ids()
                .to_vec();

            let mut generated_text = String::new();
            let mut logits_processor = LogitsProcessor::new(42, None, None);
            let start_gen = std::time::Instant::now();

            let mut token_generator = TokenGenerator::new();
            token_generator.set_stop_tokens(stop_tokens, &mut tokenizer);

            let mut generated_tokens = 0;

            for index in 0..sample_len {
                let next_token = token_generator
                    .next(
                        &tokens,
                        &mut logits_processor,
                        &mut model,
                        repeat_penalty,
                        repeat_last_n,
                    )
                    .unwrap()
                    .unwrap();
                tokens.push(next_token);
                generated_tokens += 1;

                if let Some(t) = tokenizer.next_token(next_token).unwrap() {
                    if token_generator.is_stop_token(&next_token) {
                        tx.send(StreamResponse {
                            generated_text: Some(generated_text.clone()),
                            details: Some(StreamDetails {
                                finish_reason: FinishReason::EosToken,
                                generated_tokens: index as i32,
                                seed: None,
                            }),
                            token: Token {
                                text: t.clone(),
                                logprob: Some(1.0),
                                special: true,
                                id: index as i32,
                            },
                            top_tokens: Vec::new(),
                        })
                        .await
                        .unwrap();
                        let dt = start_gen.elapsed();
                        info!(
                            "\n{generated_tokens} tokens generated ({:.2} token/s)",
                            generated_tokens as f64 / dt.as_secs_f64(),
                        );
                        return;
                    }
                    generated_text.push_str(&t);
                    trace!("{t}");
                    tx.send(StreamResponse {
                        generated_text: None,
                        details: None,
                        token: Token {
                            text: t.clone(),
                            logprob: Some(1.0),
                            special: false,
                            id: index as i32,
                        },
                        top_tokens: vec![Token {
                            text: t.clone(),
                            logprob: Some(1.0),
                            special: false,
                            id: index as i32,
                        }],
                    })
                    .await
                    .unwrap();
                }
                tx.send(StreamResponse {
                    generated_text: Some(generated_text.clone()),
                    details: Some(StreamDetails {
                        finish_reason: FinishReason::Length,
                        generated_tokens: sample_len as i32,
                        seed: None,
                    }),
                    token: Token {
                        text: "".to_string(),
                        logprob: Some(1.0),
                        special: true,
                        id: sample_len as i32,
                    },
                    top_tokens: Vec::new(),
                })
                .await
                .unwrap();
            }
            let dt = start_gen.elapsed();
            info!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)",
                generated_tokens as f64 / dt.as_secs_f64(),
            );
        });

        ReceiverStream::new(rx)
    }
}
