use chat_flame_backend::{
    config::{load_config, Config},
    llm::models::Models,
    server::server,
};
use clap::Parser;
use log::{error, info};
use std::net::SocketAddr;

/// Backend server for chat applications using Candle AI framework.
/// This server is optimized for CPU-only environments and provides a robust, efficient, and scalable service.
#[derive(Parser, Debug)]
#[command(name = "ChatFlameBackend")]
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

    /// Optional length of the generated text. If not provided, defaults to 50.
    #[structopt(short, long)]
    sample_len: Option<usize>,

    /// Optional model to use for text generation. If not provided, defaults to 7b.
    #[structopt(long)]
    model: Option<Models>,
}

async fn generate_text(prompt: String, sample_len: Option<usize>, model: Models, config: Config) {
    info!("Generating text for prompt: {}", prompt);
    let mut text_generation = chat_flame_backend::llm::create_text_generation(
        model,
        None,
        None,
        0.0,
        0,
        &config.cache_dir,
    )
    .unwrap();
    let generated_text = text_generation
        .run(&prompt, sample_len.unwrap_or(150))
        .unwrap();
    println!("{}", generated_text.unwrap_or_default());
}

async fn start_server(model: Models, config: Config) {
    info!("Starting server");
    info!("preload model");
    let _ = chat_flame_backend::llm::create_model(model, &config.cache_dir);

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
    let opt = Opt::parse();

    match load_config(&opt.config) {
        Ok(config) => {
            info!("Loaded config: {:?}", config);
            if let Some(prompt) = opt.prompt {
                generate_text(
                    prompt,
                    opt.sample_len,
                    opt.model.unwrap_or_default(),
                    config,
                )
                .await;
                return;
            } else {
                start_server(opt.model.unwrap_or(config.model), config).await;
            }
        }
        Err(e) => {
            error!("Failed to load config: {}", e);
            std::process::exit(1);
        }
    }
}
