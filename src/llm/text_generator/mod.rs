use super::{
    token_generator::{TokenGeneratorResult, TokenGeneratorTrait},
    token_output_stream::TokenOutputStream,
    FinishReason,
};
use anyhow::Result;
mod dummy_text_generator;

pub type TextProbability = (String, f32);

#[derive(Debug, PartialEq)]
pub enum TextGeneratorResult {
    Token(TextProbability),
    Finish(FinishReason),
}

/// Trait for text generation functionality.
pub trait TextGeneratorTrait {
    fn init(&mut self, prompt: String) -> Result<()>;
    fn next(&mut self) -> Result<TextGeneratorResult>;
}

pub struct TextGenerator {
    tokenizer: TokenOutputStream,
    token_generator: Box<dyn TokenGeneratorTrait>,
}

impl TextGenerator {
    pub fn new(
        tokenizer: TokenOutputStream,
        token_generator: Box<dyn TokenGeneratorTrait>,
    ) -> Self {
        Self {
            tokenizer,
            token_generator,
        }
    }
}

impl TextGeneratorTrait for TextGenerator {
    fn init(&mut self, prompt: String) -> Result<()> {
        let prompt_tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?;
        self.token_generator
            .init(prompt_tokens.get_ids().to_vec())?;
        Ok(())
    }

    fn next(&mut self) -> Result<TextGeneratorResult> {
        let token = self.token_generator.next()?;
        match token {
            TokenGeneratorResult::Token((token, probability)) => {
                let text = self.tokenizer.next_token(token)?;
                match text {
                    Some(text) => Ok(TextGeneratorResult::Token((text, probability))),
                    None => Ok(TextGeneratorResult::Token(("".to_string(), 1.0))),
                }
            }
            TokenGeneratorResult::Finish(reason) => Ok(TextGeneratorResult::Finish(reason)),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::llm::{
        generate_parameter::GenerateParameter, token_generator::dummy::DummyTokenGenerator,
    };

    use super::*;

    #[test]
    fn test_text_generator() {
        let mut text_generator = TextGenerator::new(
            TokenOutputStream::new(tokenizers::tokenizer::Tokenizer::new(
                tokenizers::models::bpe::BPE::default(),
            )),
            Box::new(DummyTokenGenerator::new(GenerateParameter {
                max_new_tokens: 10,
                ..Default::default()
            })),
        );
        text_generator.init("Hello World".to_string()).unwrap();
        for _ in 0..10 {
            assert!(match text_generator.next().unwrap() {
                TextGeneratorResult::Token((_, _)) => true,
                _ => false,
            });
        }
        assert_eq!(
            text_generator.next().unwrap(),
            TextGeneratorResult::Finish(FinishReason::Length)
        );
    }
}
