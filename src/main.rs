use chat_flame_backend::{
    config::{load_config, Config},
    server::server,
};
use log::{error, info};
use std::net::SocketAddr;
use structopt::StructOpt;

/// Backend server for chat applications using Candle AI framework.
/// This server is optimized for CPU-only environments and provides a robust, efficient, and scalable service.
#[derive(StructOpt, Debug)]
#[structopt(name = "ChatFlameBackend")]
struct Opt {
    /// Sets a custom config file. If not specified, 'config.yml' is used as the default.
    #[structopt(
        short,
        long,
        default_value = "config.yml",
        help = "Specify the path to the configuration file"
    )]
    config: String,
    /// Optional text prompt for immediate text generation. If provided, runs text generation instead of starting the server.
    #[structopt(short, long)]
    prompt: Option<String>,
}

async fn generate_text(prompt: String, config: Config) {
    info!("Generating text for prompt: {}", prompt);
    let mut text_generation =
        chat_flame_backend::llm::create_text_generation(None, None, 0.0, 0, &config.cache_dir)
            .unwrap();
    let generated_text = text_generation.run(&prompt, 50).unwrap();
    println!("{}", generated_text.unwrap_or_default());
}

async fn start_server(config: Config) {
    info!("Starting server");
    info!("preload model");
    let _ = chat_flame_backend::llm::create_model(&config.cache_dir);

    info!("Running on port: {}", config.port);
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    let app = server(config);

    info!("Server running at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[tokio::main]
async fn main() {
    pretty_env_logger::init();
    let opt = Opt::from_args();

    match load_config(&opt.config) {
        Ok(config) => {
            info!("Loaded config: {:?}", config);
            if let Some(prompt) = opt.prompt {
                generate_text(prompt, config).await;
                return;
            } else {
                start_server(config).await;
            }
        }
        Err(e) => {
            error!("Failed to load config: {}", e);
            std::process::exit(1);
        }
    }
}
