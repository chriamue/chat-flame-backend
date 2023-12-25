use crate::{
    api::model::{FinishReason, StreamDetails, StreamResponse, Token},
    llm::token_generator::TokenGenerator,
};

use crate::llm::generate_parameter::GenerateParameter;
use anyhow::{Error as E, Result};
use candle_core::Device;
use candle_transformers::{generation::LogitsProcessor, models::quantized_llama::ModelWeights};
use futures::Stream;
use log::{debug, info, trace};
use std::{collections::HashSet, sync::Arc};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;

use super::{
    text_generator::{self, TextGenerator},
    token_generator::{TokenGenerator2, TokenGeneratorTrait},
    token_output_stream::TokenOutputStream,
    TextGeneratorTrait,
};

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

    pub fn run(&mut self, prompt: &str, parameter: GenerateParameter) -> Result<Option<String>> {
        let locked_tokenizer = self.tokenizer.try_lock().unwrap();
        let locked_model = self.model.try_lock().unwrap();

        let stop_tokens = vec!["<|endoftext|>", "</s>"];

        let eos_tokens: HashSet<u32> = stop_tokens
            .into_iter()
            .map(|token| {
                locked_tokenizer
                    .tokenizer()
                    .token_to_id(token)
                    .unwrap_or_default()
            })
            .collect::<HashSet<u32>>();

        let model = Box::new(locked_model.clone());
        let sampler = Box::new(LogitsProcessor::new(
            parameter.seed,
            Some(parameter.temperature),
            Some(parameter.top_p),
        ));

        let token_generator: Box<dyn TokenGeneratorTrait> =
            Box::new(TokenGenerator2::new(eos_tokens, parameter, model, sampler));

        let mut text_generator = TextGenerator::new(
            TokenOutputStream::new(locked_tokenizer.tokenizer().clone()),
            token_generator,
        );

        text_generator.init(prompt.to_string())?;

        let start_gen = std::time::Instant::now();
        let mut token_count = 0;

        let mut generated_text = String::new();
        while let Ok(result) = text_generator.next() {
            token_count += 1;
            match result {
                text_generator::TextGeneratorResult::Token((text, _)) => {
                    generated_text.push_str(&text);
                }
                text_generator::TextGeneratorResult::Finish(_) => {
                    break;
                }
            }
        }

        info!(
            "{} tokens generated ({:.2} token/s)",
            token_count,
            token_count as f64 / start_gen.elapsed().as_secs_f64(),
        );

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
                            top_tokens: None,
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
                        top_tokens: None,
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
                    top_tokens: None,
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
