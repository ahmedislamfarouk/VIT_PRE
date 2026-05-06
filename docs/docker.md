# Docker

## Dockerfile

The project includes a `Dockerfile` based on `python:3.10-slim` for reproducible execution.

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "enhanced_distilled.py"]
```

## Build

```bash
docker build -t amtd .
```

## Run

```bash
docker run --gpus all amtd
```

## Notes

- The Docker image includes all Python dependencies (`timm`, `torch`, `torchvision`, `kagglehub`, `scikit-learn`, `matplotlib`, `tqdm`, `Pillow`, `numpy`)
- Optuna is not in `requirements.txt` but is installed separately in scripts; add `optuna` if running HPO inside Docker
- Volume mount checkpoints and results for persistence:
  ```bash
  docker run --gpus all -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd)/results:/app/results amtd
  ```
