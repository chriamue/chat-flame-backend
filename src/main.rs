use std::net::SocketAddr;

use axum::Server;
use structopt::StructOpt;
use chat_flame_backend::{config::load_config, server::server};
use tokio::signal;

/// Backend server for chat applications using Candle AI framework.
/// This server is optimized for CPU-only environments and provides a robust, efficient, and scalable service.
#[derive(StructOpt, Debug)]
#[structopt(name = "ChatFlameBackend")]
struct Opt {
    /// Sets a custom config file. If not specified, 'config.yml' is used as the default.
    #[structopt(short, long, default_value = "config.yml", help = "Specify the path to the configuration file")]
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
            let app = server();

            println!("Server running at http://{}", addr);
            Server::bind(&addr)
                .serve(app.into_make_service())
                .with_graceful_shutdown(shutdown_signal())
                .await
                .unwrap();
        }
        Err(e) => {
            eprintln!("Failed to load config: {}", e);
            std::process::exit(1);
        }
    }
}

async fn shutdown_signal() {
    let _ = signal::ctrl_c().await;
    println!("Shutting down server...");
}
