/// Parameters used to generate samples
#[derive(Debug, Clone, Default)]
pub struct GenerateParameter {
    /// Maximum number of tokens to generate
    pub max_new_tokens: usize,
    /// The seed used to generate samples
    pub seed: u64,
    /// The temperature used to generate samples
    pub temperature: f64,
    /// Nucleus sampling probability cutoff
    pub top_p: f64,
}
