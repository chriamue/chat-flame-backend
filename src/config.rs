use serde::Deserialize;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use crate::llm::models::Models;

/// Configuration for the chat-flame-backend application.
///
/// It includes settings for the server port, cache directory, and model information.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct Config {
    /// Port number on which the server will listen.
    pub port: u16,

    /// Optional path to the directory where cache files are stored.
    pub cache_dir: Option<PathBuf>,

    /// Model to be used by the server.
    pub model: Models,

    /// Whether to keep the default model in memory.
    pub keep_in_memory: Option<bool>,
}

/// Loads the application configuration from a YAML file.
///
/// # Arguments
///
/// * `file_path` - Path to the YAML configuration file.
///
/// # Returns
///
/// This function returns `Config` on success or an error if the file cannot be read
/// or the contents cannot be parsed into a `Config`.
///
/// # Examples
///
/// ```
/// use chat_flame_backend::config::load_config;
/// let config = load_config("path/to/config.yml").unwrap_or_default();
/// println!("Server will run on port: {}", config.port);
/// ```
pub fn load_config(file_path: &str) -> Result<Config, Box<dyn std::error::Error>> {
    let mut file = File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let config: Config = serde_yaml::from_str(&contents)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_config() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            "port: 8080\ncache_dir: /tmp\nmodel: 7b-open-chat-3.5"
        )
        .unwrap();

        let config_path = temp_file.path().to_str().unwrap();
        let config = load_config(config_path).unwrap();

        assert_eq!(config.port, 8080);
        assert_eq!(config.cache_dir, Some(PathBuf::from("/tmp")));
        assert_eq!(config.model, Models::OpenChat35);
        assert_eq!(config.keep_in_memory, None);
    }
}
