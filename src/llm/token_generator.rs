use std::collections::HashSet;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::{generation::LogitsProcessor, models::quantized_llama::ModelWeights};

#[derive(Default)]
pub struct TokenGenerator {
    index: usize,
    stop_tokens: HashSet<u32>,
}

impl TokenGenerator {
    pub fn new() -> Self {
        Self {
            index: 0,
            stop_tokens: HashSet::new(),
        }
    }

    pub fn next(
        &mut self,
        tokens: &[u32],
        logits_processor: &mut LogitsProcessor,
        model: &mut ModelWeights,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<Option<u32>> {
        let context_size = if self.index > 0 { 1 } else { tokens.len() };
        let start_pos = tokens.len().saturating_sub(context_size);
        let ctxt = &tokens[start_pos..];
        let input = Tensor::new(ctxt, &Device::Cpu)?.unsqueeze(0)?;
        let logits = model.forward(&input, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = {
            let start_at = tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &tokens[start_at..],
            )?
        };
        self.index += 1;
        Ok(Some(logits_processor.sample(&logits)?))
    }

    pub fn reset(&mut self) {
        self.index = 0;
        self.stop_tokens.clear();
    }

    pub fn set_stop_tokens(
        &mut self,
        stop_tokens: Option<Vec<String>>,
        tokenizer: &mut TokenOutputStream,
    ) {
        self.stop_tokens = stop_tokens
            .unwrap_or_default()
            .iter()
            .filter_map(|token| tokenizer.get_token(token))
            .collect();
    }

    pub fn is_stop_token(&self, token: &u32) -> bool {
        self.stop_tokens.contains(token)
    }
}
