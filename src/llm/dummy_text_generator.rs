use futures::stream::{self, Stream};
use std::{collections::HashSet, pin::Pin};

use super::TextGenerator;

pub struct DummyTextGenerator {
    text: String,
}

impl DummyTextGenerator {
    pub fn new(text: String) -> Self {
        Self { text }
    }
}

impl TextGenerator for DummyTextGenerator {
    fn generate(
        &mut self,
        _prompt: &str,
        sample_len: usize,
        stop_tokens: Option<Vec<String>>,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let mut generated_text = String::new();
        let mut token_count = 0;

        let stop_token_set: HashSet<String> = stop_tokens.unwrap_or_default().into_iter().collect();

        let mut current_token = String::new();
        for ch in self.text.chars() {
            if ch.is_whitespace() {
                if !current_token.is_empty() && token_count < sample_len {
                    if stop_token_set.contains(&current_token) {
                        break;
                    }
                    generated_text.push_str(&current_token);
                    current_token.clear();
                    token_count += 1;
                }
                if token_count < sample_len {
                    generated_text.push(ch);
                    token_count += 1;
                }
            } else {
                current_token.push(ch);
            }
        }

        if !current_token.is_empty() && token_count < sample_len {
            if stop_token_set.contains(&current_token) {
                return Ok(Some(generated_text));
            }
            generated_text.push_str(&current_token);
        }

        Ok(Some(generated_text))
    }

    fn generate_stream(
        &mut self,
        _prompt: &str,
        _sample_len: usize,
        _stop_tokens: Option<Vec<String>>,
    ) -> Pin<Box<dyn Stream<Item = Result<String, Box<dyn std::error::Error>>> + '_>> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();

        for ch in self.text.chars() {
            if ch.is_whitespace() {
                if !current_token.is_empty() {
                    tokens.push(Ok(current_token.clone()));
                    current_token.clear();
                }
                tokens.push(Ok(ch.to_string()));
            } else {
                current_token.push(ch);
            }
        }

        if !current_token.is_empty() {
            tokens.push(Ok(current_token));
        }

        let stream = stream::iter(tokens);
        Box::pin(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[test]
    fn test_generate_with_sample_len_and_stop_tokens() {
        let mut generator = DummyTextGenerator::new("Hello world this is a test".to_string());

        let result = generator.generate("", 3, None).unwrap();
        assert_eq!(result, Some("Hello world".to_string()));

        generator = DummyTextGenerator::new("Hello world this is a test".to_string());

        let stop_tokens = vec!["this".to_string()];
        let result = generator.generate("", 10, Some(stop_tokens)).unwrap();
        assert_eq!(result, Some("Hello world ".to_string()));
    }

    #[tokio::test]
    async fn test_generate_stream() {
        let mut generator = DummyTextGenerator::new("Test stream text".to_string());
        let mut stream = generator.generate_stream("dummy prompt", 10, None);

        let mut results = Vec::new();
        while let Some(result) = stream.next().await {
            results.push(result.unwrap());
        }

        assert_eq!(results, vec!["Test", " ", "stream", " ", "text"]);
    }
}
