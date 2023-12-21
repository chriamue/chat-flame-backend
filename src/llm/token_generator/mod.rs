use std::collections::HashSet;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::{generation::LogitsProcessor, models::quantized_llama::ModelWeights};

use super::{
    generate_parameter::GenerateParameter, model_processor::ModelProcessor, sampler::Sampler,
    token_output_stream::TokenOutputStream,
};

mod dummy;

#[derive(Default)]
pub struct TokenGenerator {
    index: usize,
    stop_tokens: HashSet<u32>,
}

impl TokenGenerator {
    pub fn new() -> Self {
        Self {
            index: 0,
            stop_tokens: HashSet::new(),
        }
    }

    pub fn next(
        &mut self,
        tokens: &[u32],
        logits_processor: &mut LogitsProcessor,
        model: &mut ModelWeights,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<Option<u32>> {
        let context_size = if self.index > 0 { 1 } else { tokens.len() };
        let start_pos = tokens.len().saturating_sub(context_size);
        let ctxt = &tokens[start_pos..];
        let input = Tensor::new(ctxt, &Device::Cpu)?.unsqueeze(0)?;
        let logits = model.forward(&input, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = {
            let start_at = tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &tokens[start_at..],
            )?
        };
        self.index += 1;
        Ok(Some(logits_processor.sample(&logits)?))
    }

    pub fn reset(&mut self) {
        self.index = 0;
        self.stop_tokens.clear();
    }

    pub fn set_stop_tokens(
        &mut self,
        stop_tokens: Option<Vec<String>>,
        tokenizer: &mut TokenOutputStream,
    ) {
        self.stop_tokens = stop_tokens
            .unwrap_or_default()
            .iter()
            .filter_map(|token| tokenizer.get_token(token))
            .collect();
    }

    pub fn is_stop_token(&self, token: &u32) -> bool {
        self.stop_tokens.contains(token)
    }
}

#[derive(Debug, PartialEq)]
pub enum FinishReason {
    Length,
    EosToken,
    StopSequence,
}

pub type TokenProbability = (u32, f32);

#[derive(Debug, PartialEq)]
pub enum TokenGeneratorResult {
    Token(TokenProbability),
    Finish(FinishReason),
}

pub trait TokenGeneratorTrait {
    fn next(&mut self) -> Result<TokenGeneratorResult>;
}

pub struct TokenGenerator2 {
    index: usize,
    stop_tokens: HashSet<u32>,
    parameter: GenerateParameter,
    prompt_tokens: Vec<u32>,
    sampler: Box<dyn Sampler>,
    model: Box<dyn ModelProcessor>,
    next_token: Option<u32>,
}

impl TokenGenerator2 {
    pub fn new(
        prompt_tokens: Vec<u32>,
        stop_tokens: HashSet<u32>,
        parameter: GenerateParameter,
        model: Box<dyn ModelProcessor>,
        sampler: Box<dyn Sampler>,
    ) -> Self {
        let mut token_generator = Self {
            index: 0,
            stop_tokens,
            parameter,
            prompt_tokens,
            model,
            sampler,
            next_token: None,
        };
        token_generator.next_token = Some(token_generator.next_token().unwrap_or_default());
        token_generator
    }
}

impl TokenGenerator2 {
    fn next_token(&mut self) -> Result<u32> {
        let next_token = {
            let input = Tensor::new(self.prompt_tokens.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            self.sampler.sample(&logits)?
        };
        Ok(next_token)
    }
}

impl TokenGeneratorTrait for TokenGenerator2 {
    fn next(&mut self) -> Result<TokenGeneratorResult> {
        if self.index >= self.parameter.max_new_tokens {
            return Ok(TokenGeneratorResult::Finish(FinishReason::Length));
        }

        let next_token = {
            let input = Tensor::new(&[self.next_token.unwrap()], &Device::Cpu)?.unsqueeze(0)?;
            let logits = self
                .model
                .forward(&input, self.prompt_tokens.len() + self.index)?;
            let logits = logits.squeeze(0)?;
            self.sampler.sample(&logits)?
        };

        // todo: repeat penalty

        if self.stop_tokens.contains(&next_token) {
            return Ok(TokenGeneratorResult::Finish(FinishReason::EosToken));
        }
        self.next_token = Some(next_token);
        self.index += 1;
        Ok(TokenGeneratorResult::Token((next_token, 1.0)))
    }
}

#[cfg(test)]
mod tests {
    use crate::llm::{model_processor::DummyModelProcessor, sampler::DummySampler};

    use super::*;

    #[test]
    fn test_token_generator_finish() {
        let mut token_generator = TokenGenerator2::new(
            vec![0, 1, 2],
            HashSet::new(),
            GenerateParameter { max_new_tokens: 10 },
            Box::new(DummyModelProcessor::new()),
            Box::new(DummySampler::new()),
        );
        // starting at 1 because model processor and sampler run already in the new function.
        for index in 1..11 {
            assert_eq!(
                token_generator.next().unwrap(),
                TokenGeneratorResult::Token((index, 1.0))
            );
        }
        assert_eq!(
            token_generator.next().unwrap(),
            TokenGeneratorResult::Finish(FinishReason::Length)
        );
    }

    #[test]
    fn test_token_generator_eos_token() {
        let stop_token = 3;
        let mut token_generator = TokenGenerator2::new(
            vec![0, 1, 2],
            vec![stop_token].into_iter().collect(),
            GenerateParameter { max_new_tokens: 10 },
            Box::new(DummyModelProcessor::new()),
            Box::new(DummySampler::new()),
        );
        for index in 1..3 {
            assert_eq!(
                token_generator.next().unwrap(),
                TokenGeneratorResult::Token((index, 1.0))
            );
        }
        assert_eq!(
            token_generator.next().unwrap(),
            TokenGeneratorResult::Finish(FinishReason::EosToken)
        );
    }
}
