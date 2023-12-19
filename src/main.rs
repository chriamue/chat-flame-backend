use chat_flame_backend::{config::load_config, server::server};
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
}

#[tokio::main]
async fn main() {
    let opt = Opt::from_args();

    match load_config(&opt.config) {
        Ok(config) => {
            // Initialize your application with config
            println!("Running on port: {}", config.port);

            let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
            let app = server(config);

            println!("Server running at http://{}", addr);

            let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
            axum::serve(listener, app).await.unwrap();
        }
        Err(e) => {
            eprintln!("Failed to load config: {}", e);
            std::process::exit(1);
        }
    }
}
