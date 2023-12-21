use super::{TextGeneratorResult, TextGeneratorTrait};
use anyhow::Result;

pub struct DummyTextGenerator {
    text: String,
}

impl DummyTextGenerator {
    pub fn new(text: String) -> Self {
        Self { text }
    }
}

impl TextGeneratorTrait for DummyTextGenerator {
    fn init(&mut self, prompt: String) -> Result<()> {
        self.text = prompt;
        Ok(())
    }

    fn next(&mut self) -> Result<TextGeneratorResult> {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
