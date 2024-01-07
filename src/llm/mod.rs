//! # Large Language Model (LLM) Module
//!
//! This module contains components necessary for loading and processing large language models.
//! It includes utilities for handling model parameters, loading models, generating tokens,
//! and other functionalities essential for text generation.

/// Parameters for text generation.
///
/// This module defines the parameters used to control the behavior of text generation,
/// such as the maximum number of new tokens to generate, temperature settings, and others.
pub mod generate_parameter;

/// Module for loading models.
///
/// Provides functionality to load model weights and other necessary components for language models.
pub mod loader;

/// Processor for language models.
///
/// Handles the processing of input data through the model, including forward passes
/// and manipulation of outputs.
pub mod model_processor;

/// Enumerations for supported models.
///
/// Defines the various language models supported by this application.
pub mod models;

/// Sampling utilities for language models.
///
/// Includes implementations for sampling methods used in text generation, such as
/// temperature-based sampling.
pub mod sampler;

/// Main text generation logic.
///
/// Central module for generating text using the language models. It orchestrates
/// the interaction between the tokenizer, model, and sampling methods.
pub mod text_generation;

/// Generator for text generation.
///
/// Manages the generation of text by iteratively producing tokens and constructing
/// the final output text.
pub mod text_generator;

/// Token generator utilities.
///
/// Provides the core functionality for generating individual tokens during the text
/// generation process.
pub mod token_generator;

/// Enumeration representing the reason why text generation was finished.
///
/// Indicates whether the generation stopped due to reaching the maximum length,
/// encountering an end-of-sequence token, or hitting a specified stop sequence.
#[derive(Debug, PartialEq)]
pub enum FinishReason {
    /// Generation stopped because the maximum length was reached.
    Length,

    /// Generation stopped due to the model producing an end-of-sequence token.
    EosToken,

    /// Generation stopped because a specified stop sequence was encountered.
    StopSequence,
}

#[derive(Clone)]
pub enum Model {
    Llama(candle_transformers::models::quantized_llama::ModelWeights),
    MixFormer(candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM),
}
