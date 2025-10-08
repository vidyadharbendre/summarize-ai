# Summarize-AI

A modular text summarization app with OOP, FastAPI backend, and Gradio frontend.

---

## Features

- Modular summarizers: T5 & BART models (easily extendable)
- Clean, SOLID OOP design with factory pattern
- FastAPI backend API with model selection
- Gradio frontend calls backend API
- Dockerized for cross-platform compatibility (Mac M1 and Windows NVIDIA)

---

## Prerequisites

- Docker & Docker Compose installed
- NVIDIA GPU users: Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- Apple Silicon (M1/M2) users: Docker Desktop for Mac (ARM version)

---

## Setup & Run

1. Clone the repo
2. docker compose up --build
3. CTRL C - stop
4. Restart: docker compose up

Rebuild and restart:

docker compose down

docker compose up --build

Then test:

Open http://127.0.0.1:7860 (try 127.0.0.1 instead of localhost)

Open http://127.0.0.1:8000/health and http://127.0.0.1:8001/docs




