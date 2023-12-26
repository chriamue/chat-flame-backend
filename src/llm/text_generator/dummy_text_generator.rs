use crate::llm::FinishReason;

use super::{TextGeneratorResult, TextGeneratorTrait};
use anyhow::Result;
/// A basic implementation of the `TextGeneratorTrait` for testing and demonstration purposes.
///
/// This struct uses a simple approach for text generation, primarily intended to serve as a placeholder
/// or for testing the framework without involving complex models.
pub struct DummyTextGenerator {
    // The internal text that will be used for generation.
    text: String,
}

impl DummyTextGenerator {
    /// Constructs a new `DummyTextGenerator` with a given text.
    ///
    /// # Arguments
    ///
    /// * `text` - The initial text to be used by the generator.
    pub fn new(text: String) -> Self {
        Self { text }
    }
}

impl TextGeneratorTrait for DummyTextGenerator {
    /// Initializes the generator with a given prompt.
    ///
    /// This method sets the internal text to the provided prompt.
    ///
    /// # Arguments
    ///
    /// * `prompt` - A `String` serving as the initial text.
    ///
    /// # Returns
    ///
    /// Always returns `Ok(())` as there is no complex initialization process.
    fn init(&mut self, prompt: String) -> Result<()> {
        self.text = prompt;
        Ok(())
    }

    /// Generates the next piece of text.
    ///
    /// For the `DummyTextGenerator`, this method returns the entire internal text at once and
    /// then signifies completion in subsequent calls.
    ///
    /// # Returns
    ///
    /// A `Result` wrapping a `TextGeneratorResult`, which is either the entire text as a token
    /// or an indication that the generation process has finished.
    fn next(&mut self) -> Result<TextGeneratorResult> {
        if !self.text.is_empty() {
            let text = std::mem::take(&mut self.text);
            Ok(TextGeneratorResult::Token((text, 1.0)))
        } else {
            Ok(TextGeneratorResult::Finish(FinishReason::Length))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_text_generator() {
        let mut generator = DummyTextGenerator::new("Hello World".to_string());
        generator.init("Test".to_string()).unwrap();

        // First call should return the entire text.
        match generator.next().unwrap() {
            TextGeneratorResult::Token((text, _)) => assert_eq!(text, "Test"),
            _ => panic!("Unexpected result on first call to next"),
        }

        // Subsequent calls should indicate that the generation process has finished.
        assert_eq!(
            generator.next().unwrap(),
            TextGeneratorResult::Finish(FinishReason::Length)
        );
    }
}
