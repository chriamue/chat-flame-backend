# chat-flame-backend

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Doc](https://img.shields.io/badge/Docs-online-green.svg)](https://blog.chriamue.de/chat-flame-backend/chat_flame_backend/)
[![codecov](https://codecov.io/gh/chriamue/chat-flame-backend/graph/badge.svg?token=MNHB75EJ2Z)](https://codecov.io/gh/chriamue/chat-flame-backend)

ChatFlameBackend is an innovative backend solution for chat applications, leveraging the power of the Candle AI framework with a focus on the Mistral model

## Quickstart

### Installation

```bash
cargo build --release
```

### Running

Run the server

```bash
cargo run --release
```

Run one of the models

```bash
cargo run --release -- --model phi-v2 --prompt 'write me fibonacci in rust'
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

## Architecture

The backend is written in rust. The models are loaded using the [candle](https://github.com/huggingface/candle) framework.
To serve the models on an http endpoint, axum is used.
Utoipa is used to provide a swagger ui for the api.

## Supported Models

- [x] [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] Zephyr
- [x] OpenChat
- [x] Starling
- [x] [Phi](https://huggingface.co/microsoft/phi-2) (Phi-1, Phi-1.5, Phi-2)
- [ ] GPT-Neo
- [ ] GPT-J
- [ ] Llama

### Mistral

["lmz/candle-mistral"](https://huggingface.co/lmz/candle-mistral)

### Phi

["microsoft/phi-2"](https://huggingface.co/microsoft/phi-2)

## Performance

The following table shows the performance metrics of the model on different systems:

| Model            | System                     | Tokens per Second |
| ---------------- | -------------------------- | ----------------- |
| 7b-open-chat-3.5 | AMD 7900X3D (12 Core) 64GB | 9.4 tokens/s      |
| 7b-open-chat-3.5 | AMD 5600G (8 Core VM) 16GB | 2.8 tokens/s      |
| 13b (llama2 13b) | AMD 7900X3D (12 Core) 64GB | 5.2 tokens/s      |
| phi-2            | AMD 7900X3D (12 Core) 64GB | 20.6 tokens/s     |
| phi-2            | AMD 5600G (8 Core VM) 16GB | 5.3 tokens/s      |
| phi-2            | Apple M2 (10 Core) 16GB    | 24.0 tokens/s     |

### Hint

The performance of the model is highly dependent on the memory bandwidth of the system.
While getting 20.6 tokens/s for the Phi-2 Model on a AMD 7900X3D with 64GB of DDR5-4800 memory,
the performance could be increased to
21.8 tokens/s by overclocking the memory to DDR5-5600.

## Todo

- [x] implement api for https://huggingface.github.io/text-generation-inference/#/
- [x] model configuration
- [x] generate stream
- [x] docker image and docker-compose
- [ ] add tests
- [ ] add documentation
- [ ] fix stop token
