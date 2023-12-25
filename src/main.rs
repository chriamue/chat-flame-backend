use chat_flame_backend::{
    config::{load_config, Config},
    llm::{generate_parameter::GenerateParameter, models::Models},
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

    /// The temperature used to generate samples, use 0 for greedy sampling.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Optional model to use for text generation. If not provided, defaults to 7b-open-chat-3.5.
    #[structopt(long)]
    model: Option<Models>,
}

async fn generate_text(
    prompt: String,
    parameter: GenerateParameter,
    model: Models,
    config: Config,
) {
    info!("Generating text for prompt: {}", prompt);
    let mut text_generation =
        chat_flame_backend::llm::create_text_generation(model, None, None, &config.cache_dir)
            .unwrap();

    let generated_text = text_generation.run(&prompt, parameter).unwrap();
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
                let parameter = GenerateParameter {
                    temperature: opt.temperature,
                    top_p: opt.top_p.unwrap_or_default(),
                    max_new_tokens: opt.sample_len.unwrap_or(50),
                    seed: opt.seed,
                    repeat_penalty: opt.repeat_penalty,
                    repeat_last_n: opt.repeat_last_n,
                };

                generate_text(prompt, parameter, opt.model.unwrap_or_default(), config).await;
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
