use crate::llm::{
    generate_parameter::GenerateParameter,
    model_processor::{DummyModelProcessor, ModelProcessor},
    sampler::{DummySampler, Sampler},
};
use anyhow::Result;
use candle_core::{Device, Tensor};

use super::{FinishReason, TokenGeneratorResult, TokenGeneratorTrait};

pub struct DummyTokenGenerator {
    parameter: GenerateParameter,
    index: usize,
    sampler: Box<dyn Sampler>,
    model: Box<dyn ModelProcessor>,
}

unsafe impl Send for DummyTokenGenerator {}

impl DummyTokenGenerator {
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
    fn test_dummy_token_generator() {
        let mut token_generator = DummyTokenGenerator::new(GenerateParameter {
            max_new_tokens: 10,
            ..Default::default()
        });
        for index in 0..10 {
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
}
