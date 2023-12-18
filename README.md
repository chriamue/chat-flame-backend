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

## Todo

- [ ] implement api for https://huggingface.github.io/text-generation-inference/#/
- [ ] model configuration
- [ ] generate stream