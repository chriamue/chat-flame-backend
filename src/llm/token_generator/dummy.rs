use crate::llm::{
    generate_parameter::GenerateParameter,
    model_processor::{DummyModelProcessor, ModelProcessor},
    sampler::{DummySampler, Sampler},
};
use anyhow::Result;
use candle_core::{Device, Tensor};

use super::{FinishReason, TokenGeneratorResult, TokenGeneratorTrait};

/// A dummy implementation of the `TokenGeneratorTrait` used for testing purposes.
///
/// This token generator produces a predefined sequence of tokens based on the index counter.
/// It's a simplified version intended for unit testing without involving complex models or samplers.
pub struct DummyTokenGenerator {
    parameter: GenerateParameter,
    index: usize,
    sampler: Box<dyn Sampler>,
    model: Box<dyn ModelProcessor>,
}

unsafe impl Send for DummyTokenGenerator {}

impl DummyTokenGenerator {
    /// Creates a new instance of `DummyTokenGenerator` with specified generation parameters.
    ///
    /// # Arguments
    ///
    /// * `parameter` - Parameters that control the generation process such as max tokens, temperature, etc.
    ///
    /// # Returns
    ///
    /// A new instance of `DummyTokenGenerator`.
    pub fn new(parameter: GenerateParameter) -> Self {
        let sampler = Box::new(DummySampler::new());
        let model = Box::new(DummyModelProcessor::new());
        DummyTokenGenerator {
            parameter,
            index: 0,
            sampler,
            model,
        }
    }
}

impl TokenGeneratorTrait for DummyTokenGenerator {
    fn init(&mut self, _prompt_tokens: Vec<u32>) -> Result<()> {
        self.index = 0;
        Ok(())
    }
    fn next(&mut self) -> Result<TokenGeneratorResult> {
        self.index += 1;
        if self.index > self.parameter.max_new_tokens {
            return Ok(TokenGeneratorResult::Finish(FinishReason::Length));
        }
        let tensor = Tensor::new(&[0.0], &Device::Cpu).unwrap();
        let logits = self.model.forward(&tensor, self.index).unwrap();
        let token = self.sampler.sample(&logits).unwrap();
        Ok(TokenGeneratorResult::Token((token, 1.0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_token_generator_with_zero_max_tokens() {
        let mut token_generator = DummyTokenGenerator::new(GenerateParameter {
            max_new_tokens: 0,
            ..Default::default()
        });
        assert_eq!(
            token_generator.next().unwrap(),
            TokenGeneratorResult::Finish(FinishReason::Length)
        );
    }

    #[test]
    fn test_dummy_token_generator_with_repeat_penalty() {
        let mut token_generator = DummyTokenGenerator::new(GenerateParameter {
            max_new_tokens: 5,
            repeat_penalty: 1.5, // Non-default value
            ..Default::default()
        });
        // Just verify that it can still produce tokens normally
        for index in 0..5 {
            assert_eq!(
                token_generator.next().unwrap(),
                TokenGeneratorResult::Token((index, 1.0))
            );
        }
    }

    #[test]
    fn test_dummy_token_generator_with_high_max_tokens() {
        let mut token_generator = DummyTokenGenerator::new(GenerateParameter {
            max_new_tokens: 1000, // High value
            ..Default::default()
        });
        // Run through multiple iterations and ensure it stops correctly
        for _ in 0..1000 {
            if let TokenGeneratorResult::Finish(_) = token_generator.next().unwrap() {
                break;
            }
        }
    }

    #[test]
    fn test_dummy_token_generator_initialization() {
        let mut token_generator = DummyTokenGenerator::new(Default::default());
        token_generator.init(vec![1, 2, 3]).unwrap(); // Initial set of tokens
        assert_eq!(
            token_generator.next().unwrap(),
            TokenGeneratorResult::Token((0, 1.0))
        );
        token_generator.init(vec![4, 5, 6]).unwrap(); // Re-initialize with new tokens
        assert_eq!(
            token_generator.next().unwrap(),
            TokenGeneratorResult::Token((1, 1.0))
        );
    }
}
