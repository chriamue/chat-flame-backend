use crate::{
    api::model::{FinishReason, StreamDetails, StreamResponse, Token},
    llm::{
        self,
        text_generator::{TextGeneratorResult, TextGeneratorTrait},
    },
};

use crate::llm::generate_parameter::GenerateParameter;
use anyhow::Result;
use candle_core::Device;
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::LogitsProcessor;
use futures::Stream;
use log::{info, trace};
use std::{collections::HashSet, path::PathBuf, sync::Arc};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;

use super::{
    loader::{create_model, create_tokenizer},
    models::Models,
    text_generator::{self, TextGenerator},
    token_generator::{TokenGenerator, TokenGeneratorTrait},
    Model,
};

#[derive(Clone)]
pub struct TextGeneration {
    model: Arc<Mutex<Model>>,
    tokenizer: Arc<Mutex<TokenOutputStream>>,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(model: Model, tokenizer: Tokenizer, _device: &Device) -> Self {
        Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(Mutex::new(TokenOutputStream::new(tokenizer))),
        }
    }

    pub fn run(&mut self, prompt: &str, parameter: GenerateParameter) -> Result<Option<String>> {
        info!(
            "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
            parameter.temperature, parameter.repeat_penalty, parameter.repeat_last_n
        );

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
            Box::new(TokenGenerator::new(eos_tokens, parameter, model, sampler));

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
        parameter: GenerateParameter,
        _stop_tokens: Option<Vec<String>>,
    ) -> impl Stream<Item = StreamResponse> {
        info!(
            "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
            parameter.temperature, parameter.repeat_penalty, parameter.repeat_last_n
        );

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

        let (tx, rx) = tokio::sync::mpsc::channel(32);

        let prompt = prompt.to_string();
        let parameter = parameter.clone();

        let tokenizer = locked_tokenizer.tokenizer().clone();

        tokio::spawn(async move {
            let token_generator: Box<dyn TokenGeneratorTrait> = Box::new(TokenGenerator::new(
                eos_tokens,
                parameter.clone(),
                model,
                sampler,
            ));

            let mut text_generator =
                TextGenerator::new(TokenOutputStream::new(tokenizer), token_generator);

            text_generator.init(prompt.to_string()).unwrap();

            let start_gen = std::time::Instant::now();
            let mut token_count = 0;
            let mut generated_text = String::new();

            for index in 0..parameter.max_new_tokens {
                if let Ok(t) = text_generator.next() {
                    match t {
                        TextGeneratorResult::Token((text, _)) => {
                            token_count += 1;
                            generated_text.push_str(&text);
                            trace!("{text}");
                            tx.send(StreamResponse {
                                generated_text: None,
                                details: None,
                                token: Token {
                                    text: text.clone(),
                                    logprob: Some(1.0),
                                    special: false,
                                    id: index as i32,
                                },
                                top_tokens: None,
                            })
                            .await
                            .unwrap();
                        }
                        TextGeneratorResult::Finish(reason) => {
                            match reason {
                                llm::FinishReason::Length => {
                                    tx.send(StreamResponse {
                                        generated_text: Some(generated_text.clone()),
                                        details: Some(StreamDetails {
                                            finish_reason: FinishReason::Length,
                                            generated_tokens: index as i32,
                                            seed: Some(parameter.seed as i64),
                                        }),
                                        token: Token {
                                            text: "".to_string(),
                                            logprob: Some(1.0),
                                            special: true,
                                            id: index as i32,
                                        },
                                        top_tokens: None,
                                    })
                                    .await
                                    .unwrap();
                                }
                                _ => {
                                    tx.send(StreamResponse {
                                        generated_text: Some(generated_text.clone()),
                                        details: Some(StreamDetails {
                                            finish_reason: FinishReason::EosToken,
                                            generated_tokens: index as i32,
                                            seed: Some(parameter.seed as i64),
                                        }),
                                        token: Token {
                                            text: "".to_string(),
                                            logprob: Some(1.0),
                                            special: true,
                                            id: index as i32,
                                        },
                                        top_tokens: None,
                                    })
                                    .await
                                    .unwrap();
                                }
                            }
                            break;
                        }
                    }
                }
            }
            let dt = start_gen.elapsed();
            info!(
                "\n{token_count} tokens generated ({:.2} token/s)",
                token_count as f64 / dt.as_secs_f64(),
            );
        });

        ReceiverStream::new(rx)
    }
}

pub fn create_text_generation(
    model: Models,
    cache_dir: &Option<PathBuf>,
) -> Result<TextGeneration, Box<dyn std::error::Error>> {
    let tokenizer = create_tokenizer(model).expect("Failed to create tokenizer");
    let model = create_model(model, cache_dir).expect("Failed to create model");

    let device = Device::Cpu;

    Ok(TextGeneration::new(model.0, tokenizer, &device))
}
