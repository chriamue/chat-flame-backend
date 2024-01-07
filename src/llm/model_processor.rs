//! Model Processor module for text generation.
//!
//! This module contains the `ModelProcessor` trait and its implementations
//! which are used for processing input tensors and generating output tensors
//! representing logits from a language model.

use super::Model;
use candle_core::{Result, Tensor};

/// A trait for processing model inputs and generating outputs.
///
/// This trait defines a method for processing input tensors through a model
/// and generating output tensors.
pub trait ModelProcessor {
    /// Processes an input tensor and generates an output tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - A reference to the input tensor.
    /// * `index_pos` - The position index for processing.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the output tensor.
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor>;
}

impl ModelProcessor for Model {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        match self {
            Model::Llama(model) => model.forward(x, index_pos),
            Model::MixFormer(model) => model.forward(x),
        }
    }
}

/// A dummy implementation of `ModelProcessor` for testing purposes.
///
/// This processor simulates model outputs by returning incrementing tensors.
pub struct DummyModelProcessor {
    index: usize,
}

impl DummyModelProcessor {
    /// Creates a new `DummyModelProcessor`.
    pub fn new() -> Self {
        Self { index: 0 }
    }
}

/// Provides a default instance of `DummyModelProcessor`.
impl Default for DummyModelProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Implementation of `ModelProcessor` for `DummyModelProcessor`.
impl ModelProcessor for DummyModelProcessor {
    fn forward(&mut self, x: &Tensor, _index_pos: usize) -> Result<Tensor> {
        self.index += 1;
        let y = Tensor::new(&[self.index as f32 - 1.0], x.device())?;
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Tests the `DummyModelProcessor` to ensure it returns incrementing tensors.
    #[test]
    fn test_dummy_model_processor() {
        let mut model_processor = DummyModelProcessor::new();
        let x = Tensor::new(&[0.0], &Device::Cpu).unwrap();

        for index in 0..10 {
            let y = model_processor
                .forward(&x, index)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(y, vec![index as f32]);
        }
    }
}
