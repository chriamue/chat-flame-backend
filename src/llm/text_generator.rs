use futures::stream::Stream;
use std::pin::Pin;

/// Trait for text generation functionality.
pub trait TextGenerator {
    /// Generates text synchronously.
    fn generate(
        &mut self,
        prompt: &str,
        sample_len: usize,
        stop_tokens: Option<Vec<String>>,
    ) -> Result<Option<String>, Box<dyn std::error::Error>>;

    /// Generates text as a stream asynchronously.
    fn generate_stream(
        &mut self,
        prompt: &str,
        sample_len: usize,
        stop_tokens: Option<Vec<String>>,
    ) -> Pin<Box<dyn Stream<Item = Result<String, Box<dyn std::error::Error>>> + '_>>;
}
