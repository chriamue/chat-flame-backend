//! Generate Parameters Module.
//!
//! This module defines parameters used for controlling text generation.

use serde::{Deserialize, Serialize};

/// Parameters used to generate samples.
///
/// This struct defines various settings that influence the behavior of the text generation process,
/// such as token limits, sampling temperature, and repeat penalties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateParameter {
    /// Maximum number of new tokens to generate.
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,

    /// Seed used for deterministic generation.
    #[serde(default = "default_seed")]
    pub seed: u64,

    /// Temperature for sampling.
    #[serde(default = "default_temperature")]
    pub temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[serde(default = "default_top_p")]
    pub top_p: f64,

    /// Penalty for repeating tokens.
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,

    /// The number of last tokens to consider for applying the repeat penalty.
    #[serde(default = "default_repeat_last_n")]
    pub repeat_last_n: usize,
}

fn default_max_new_tokens() -> usize {
    50
}

fn default_seed() -> u64 {
    299792458
}

fn default_temperature() -> f64 {
    1.0
}

fn default_top_p() -> f64 {
    0.9
}

fn default_repeat_penalty() -> f32 {
    1.0
}

fn default_repeat_last_n() -> usize {
    64
}

impl Default for GenerateParameter {
    fn default() -> Self {
        serde_json::from_str("{}").unwrap_or_else(|_| GenerateParameter {
            max_new_tokens: default_max_new_tokens(),
            seed: default_seed(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            repeat_penalty: default_repeat_penalty(),
            repeat_last_n: default_repeat_last_n(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_generate_parameter() {
        let param = GenerateParameter::default();
        assert_eq!(param.max_new_tokens, default_max_new_tokens());
        assert_eq!(param.seed, default_seed());
        assert_eq!(param.temperature, default_temperature());
        assert_eq!(param.top_p, default_top_p());
        assert_eq!(param.repeat_penalty, default_repeat_penalty());
        assert_eq!(param.repeat_last_n, default_repeat_last_n());
    }
}
