use serde::Deserialize;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub port: u16,
    pub cache_dir: Option<PathBuf>,
}

pub fn load_config(file_path: &str) -> Result<Config, Box<dyn std::error::Error>> {
    let mut file = File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let config: Config = serde_yaml::from_str(&contents)?;
    Ok(config)
}
