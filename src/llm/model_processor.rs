use candle_core::{Result, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;

pub trait ModelProcessor {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor>;
}

impl ModelProcessor for ModelWeights {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        Self::forward(self, x, index_pos)
    }
}

pub struct DummyModelProcessor {
    index: usize,
}

impl DummyModelProcessor {
    pub fn new() -> Self {
        Self { index: 0 }
    }
}

impl ModelProcessor for DummyModelProcessor {
    fn forward(&mut self, x: &Tensor, _index_pos: usize) -> Result<Tensor> {
        self.index += 1;
        let y = Tensor::new(&[self.index as f32 - 1.0], &x.device())?;
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

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
