use candle_core::{Device, Tensor};

use crate::llm::{
    generate_parameter::GenerateParameter,
    sampler::{DummySampler, Sampler},
};

use super::{FinishReason, TokenGeneratorResult, TokenGeneratorTrait};

struct DummyTokenGenerator {
    parameter: GenerateParameter,
    index: usize,
    sampler: Box<dyn Sampler>,
}

impl DummyTokenGenerator {
    pub fn new(parameter: GenerateParameter) -> Self {
        let sampler = Box::new(DummySampler::new());
        DummyTokenGenerator {
            parameter,
            index: 0,
            sampler,
        }
    }
}

impl TokenGeneratorTrait for DummyTokenGenerator {
    fn next(&mut self) -> TokenGeneratorResult {
        self.index += 1;
        if self.index > self.parameter.max_new_tokens {
            return TokenGeneratorResult::Finish(FinishReason::Length);
        }
        let token = self
            .sampler
            .sample(&Tensor::new(&[1.0], &Device::Cpu).unwrap())
            .unwrap();
        TokenGeneratorResult::Token((token, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_token_generator() {
        let mut token_generator =
            DummyTokenGenerator::new(GenerateParameter { max_new_tokens: 10 });
        for index in 0..10 {
            assert_eq!(
                token_generator.next(),
                TokenGeneratorResult::Token((index, 1.0))
            );
        }
        assert_eq!(
            token_generator.next(),
            TokenGeneratorResult::Finish(FinishReason::Length)
        );
    }
}
