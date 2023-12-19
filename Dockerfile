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

CMD ["./chat-flame-backend"]
