FROM rust:1.74 as builder

RUN USER=root cargo new --bin chat-flame-backend
WORKDIR /chat-flame-backend

COPY ./Cargo.lock* ./
COPY ./Cargo.toml ./Cargo.toml

RUN cargo build --release
RUN rm src/*.rs

COPY ./src ./src

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt install -y openssl

COPY --from=builder /chat-flame-backend/target/release/chat-flame-backend .
COPY ./config.yml .

# Docker Image Labels
LABEL org.opencontainers.image.title "Chat Flame Backend"
LABEL org.opencontainers.image.description "A backend inference service for chat applications using Rust and Axum."
LABEL org.opencontainers.image.version "0.1.0"
LABEL org.opencontainers.image.authors "Christian M <chriamue@gmail.com>"
LABEL org.opencontainers.image.source "https://github.com/chriamue/chat-flame-backend"
LABEL org.opencontainers.image.licenses "MIT"

CMD ["./chat-flame-backend"]
