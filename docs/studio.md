# Studio UI

MLX Forge Studio is a browser-based interface for managing training runs, datasets, and inference.

## Launch

```bash
mlx-forge studio
```

Opens at [http://localhost:8741](http://localhost:8741).

## Pages

### Dashboard
Overview of recent runs, system status, and quick actions.

### Experiments
List all training runs with status, loss curves, and metrics. Click a run to see details.

### Run Detail
Full metrics dashboard for a single run:
- Live loss chart (updates during training via WebSocket)
- Metric cards: train loss, val loss, throughput, memory, tokens
- Configuration viewer
- Checkpoint list with export/push-to-hub actions

### Playground
Interactive text generation with any loaded model or adapter:
- Select model and optional adapter
- Adjust temperature, top-p, max tokens
- Chat or completion mode

### Data Library
Browse and manage datasets:
- Curated catalog with 20+ datasets
- Download, inspect, and delete datasets
- Import from HuggingFace Hub

### Settings
Configure training defaults and Studio preferences.

## API

Studio's REST API is available at `/api/v1/` and `/api/v2/`:

- `GET /api/v1/runs` — List runs
- `GET /api/v1/runs/{id}` — Run details
- `POST /api/v1/training/start` — Start training
- `POST /api/v1/runs/{id}/export` — Export model
- `POST /api/v1/runs/{id}/push-to-hub` — Push to HuggingFace
- `GET /api/v2/queue` — Job queue status
- `POST /api/v2/queue/submit` — Submit training job
