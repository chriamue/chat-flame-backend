// source: https://github.com/huggingface/candle/blob/main/candle-examples/examples/mistral/main.rs
pub mod generate_parameter;
pub mod loader;
pub mod model_processor;
pub mod models;
pub mod sampler;
pub mod text_generation;
pub mod text_generator;
pub mod token_generator;

#[derive(Debug, PartialEq)]
pub enum FinishReason {
    Length,
    EosToken,
    StopSequence,
}
