//! Sampler module for text generation.
//!
//! This module contains the `Sampler` trait and its implementations which are
//! used for sampling tokens based on the output logits from a language model.

use candle_core::{Result, Tensor};
use candle_transformers::generation::LogitsProcessor;

/// A trait for sampling a token based on logits output.
///
/// This trait defines a method for sampling a single token from a distribution
/// represented by logits.
pub trait Sampler {
    /// Samples a token based on provided logits.
    ///
    /// # Arguments
    ///
    /// * `logits` - A reference to a tensor containing logits output from the model.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the sampled token's ID.
    fn sample(&mut self, logits: &Tensor) -> Result<u32>;
}

/// Implementation of `Sampler` for the `LogitsProcessor` from `candle_transformers`.
impl Sampler for LogitsProcessor {
    fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        Self::sample(self, logits)
    }
}

/// A dummy implementation of `Sampler` for testing purposes.
///
/// This sampler sequentially returns incrementing integers as tokens.
pub struct DummySampler {
    index: usize,
}

impl DummySampler {
    /// Creates a new `DummySampler`.
    pub fn new() -> Self {
        Self { index: 0 }
    }
}

/// Provides a default instance of `DummySampler`.
impl Default for DummySampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Implementation of `Sampler` for `DummySampler`.
impl Sampler for DummySampler {
    fn sample(&mut self, _logits: &Tensor) -> Result<u32> {
        self.index += 1;
        Ok(self.index as u32 - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Tests the `DummySampler` to ensure it returns incrementing integers.
    #[test]
    fn test_dummy_sampler() {
        let mut sampler = DummySampler::new();
        assert_eq!(
            sampler
                .sample(&Tensor::new(&[1.0], &Device::Cpu).unwrap())
                .unwrap(),
            0
        );
        assert_eq!(
            sampler
                .sample(&Tensor::new(&[1.0], &Device::Cpu).unwrap())
                .unwrap(),
            1
        );
    }
}
