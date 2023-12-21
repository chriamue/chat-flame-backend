use candle_core::{Result, Tensor};
use candle_transformers::generation::LogitsProcessor;

pub trait Sampler {
    fn sample(&mut self, logits: &Tensor) -> Result<u32>;
}

impl Sampler for LogitsProcessor {
    fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        Self::sample(self, logits)
    }
}

pub struct DummySampler {
    index: usize,
}

impl DummySampler {
    pub fn new() -> Self {
        Self { index: 0 }
    }
}

impl Sampler for DummySampler {
    fn sample(&mut self, _logits: &Tensor) -> Result<u32> {
        self.index += 1;
        Ok(self.index as u32 - 1)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

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
