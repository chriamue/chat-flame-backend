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

### Docker

```bash
docker-compose up --build
```

Visit http://localhost:8080/swagger-ui for the swagger ui.

## Testing

### Test using the shell

```bash
cargo test
```

or with curl

```bash
curl -X POST http://localhost:8080/generate \
     -H "Content-Type: application/json" \
     -d '{"inputs": "Your text prompt here"}'
```

or the stream endpoint

```bash
curl -X POST -H "Content-Type: application/json" -d '{"inputs": "Your input text"}' http://localhost:8080/generate_stream
```

### Test using python

You can find a detailed documentation on how to use the python client on [huggingface](https://huggingface.co/docs/text-generation-inference/basic_tutorials/consuming_tgi#inference-client).

```bash
virtualenv .venv
source .venv/bin/activate
pip install huggingface-hub
python test.py
```


## Supported Models

- [x] [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] Zephyr
- [x] OpenChat
- [x] Starling
- [ ] GPT-Neo
- [ ] GPT-J
- [ ] Llama


### Mistral

["lmz/candle-mistral"](https://huggingface.co/lmz/candle-mistral)

## Performance

The following table shows the performance metrics of the model on different systems:

| Model            | System                     | Tokens per Second |
| ---------------- | -------------------------- | ----------------- |
| 7b-open-chat-3.5 | AMD 7900X3D (12 Core) 64GB | 9.4 tokens/s      |
| 7b-open-chat-3.5 | AMD 5600G (8 Core VM) 16GB | 2.8 tokens/s      |
| 13b (llama2 13b) | AMD 7900X3D (12 Core) 64GB | 5.2 tokens/s      |

## Todo

- [x] implement api for https://huggingface.github.io/text-generation-inference/#/
- [x] model configuration
- [x] generate stream
- [x] docker image and docker-compose
- [ ] add tests
- [ ] add documentation
