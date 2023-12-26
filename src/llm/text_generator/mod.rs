use super::{
    token_generator::{TokenGeneratorResult, TokenGeneratorTrait},
    FinishReason,
};
use anyhow::Result;
use candle_examples::token_output_stream::TokenOutputStream;
mod dummy_text_generator;

/// Represents the probability associated with a piece of generated text.
pub type TextProbability = (String, f32);

/// Enumerates possible results from a text generation process.
///
/// This enum is used to encapsulate the outcomes of text generation, including
/// both the generation of a new token and the conclusion of the generation process.
#[derive(Debug, PartialEq)]
pub enum TextGeneratorResult {
    /// Represents a generated piece of text along with its probability.
    ///
    /// The `String` is the generated text, and the `f32` is the probability associated with it.
    Token(TextProbability),

    /// Indicates the completion of the text generation process.
    ///
    /// This variant is used when the generation process reaches an end, either due to reaching
    /// a specified limit or encountering a stopping condition.
    Finish(FinishReason),
}

/// A trait defining the core functionality for text generation.
///
/// This trait encapsulates the necessary methods for initializing the generation process with a
/// prompt and then producing text iteratively.
pub trait TextGeneratorTrait {
    /// Initializes the text generation process with a given prompt.
    ///
    /// This method sets up the necessary state for text generation based on the provided prompt.
    ///
    /// # Arguments
    ///
    /// * `prompt` - A `String` that serves as the starting point for text generation.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure of the initialization process.
    fn init(&mut self, prompt: String) -> Result<()>;

    /// Generates the next piece of text in the sequence.
    ///
    /// This method should be called iteratively to generate text progressively.
    /// It provides the next piece of text based on the current state of the generator.
    ///
    /// # Returns
    ///
    /// A `Result` wrapping a `TextGeneratorResult`, which can be either a generated token
    /// or an indication that the generation process has finished.
    fn next(&mut self) -> Result<TextGeneratorResult>;
}

/// Handles the text generation process.
///
/// This struct is responsible for managing the token generation and converting tokens into text.
pub struct TextGenerator {
    /// The tokenizer used to encode the prompt and decode the generated tokens.
    tokenizer: TokenOutputStream,

    /// The token generator that produces tokens based on the model's output.
    token_generator: Box<dyn TokenGeneratorTrait>,
}

impl TextGenerator {
    /// Constructs a new `TextGenerator`.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - Tokenizer for encoding prompts and decoding generated tokens.
    /// * `token_generator` - Token generator that provides the logic for generating tokens.
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
