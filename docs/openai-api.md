# OpenAI-Compatible API

MLX Forge includes an OpenAI-compatible API server for serving fine-tuned models.

## Start the Server

```bash
# Pre-load a model
mlx-forge serve --model Qwen/Qwen3-0.6B --port 8000

# Serve a fine-tuned model from a training run
mlx-forge serve --model <run-id>

# Serve with an adapter
mlx-forge serve --model Qwen/Qwen3-0.6B --adapter path/to/checkpoint
```

## Endpoints

### `POST /v1/chat/completions`

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is MLX?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

### `POST /v1/completions`

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "The capital of France is",
    "max_tokens": 50
  }'
```

### `GET /v1/models`

```bash
curl http://localhost:8000/v1/models
```

## Streaming

Set `"stream": true` for Server-Sent Events:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

## Integration Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen3-0.6B",
    api_key="not-needed",
)
response = llm.invoke("What is fine-tuning?")
```

## Stop Sequences

```json
{
  "stop": ["END", "\n\n"]
}
```

The server checks generated text against stop strings and truncates output at the first match.
