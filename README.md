# chat-flame-backend
ChatFlameBackend is an innovative backend solution for chat applications, leveraging the power of the Candle AI framework with a focus on the Mistral model

## Quickstart

### Installation

```bash
cargo build --release
```

### Running

```bash
cargo run --release
```

### Testing

```bash
cargo test
```

or with curl

```bash
curl -X POST http://localhost:8080/generate-text \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Your text prompt here"}'
```

or the stream endpoint

```bash
curl -X POST -H "Content-Type: application/json" -d '{"inputs": "Your input text"}' http://localhost:8080/generate_stream
```

### Docker

```bash
docker-compose up --build
```

Visit http://localhost:8080/swagger-ui for the swagger ui.


## Todo

- [ ] implement api for https://huggingface.github.io/text-generation-inference/#/
- [ ] model configuration
- [ ] generate stream
- [x] docker image and docker-compose
