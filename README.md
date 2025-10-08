# 📝 Summarize-AI

A modern, modular text summarization application built with **Object-Oriented Programming (OOP)** principles, featuring a **FastAPI** backend and **Gradio** frontend interface.

![Python](https://img.shields.io/badge/Python-3.11.13-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-red)

---

## 🌟 Features

### **AI-Powered Summarization**
- **Multiple Models**: Choose between T5-small and BART-large-cnn models
- **Adjustable Length**: Control summary length with intuitive sliders
- **Real-time Processing**: Get summaries in seconds

### **Professional Architecture**
- **Clean Code**: Follows SOLID principles and design patterns
- **Factory Pattern**: Easy to extend with new summarization models  
- **RESTful API**: Well-documented FastAPI backend
- **Modern UI**: Interactive Gradio web interface

### **Production Ready**
- **Dockerized**: Cross-platform compatibility (Mac M1, Windows, Linux)
- **Microservices**: Separate backend and frontend containers
- **Model Caching**: Persistent model storage across restarts
- **Error Handling**: Robust error management and logging

---

## 🎯 Quick Demo

1. **Enter your text** in the input box
2. **Select a model** (T5 or BART)
3. **Adjust length** with min/max sliders
4. **Get instant summaries**!

*Example: Transform a 500-word article into a concise 50-word summary*

---

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have:

- **Docker Desktop** installed ([Download here](https://www.docker.com/products/docker-desktop/))
- **Git** installed ([Download here](https://git-scm.com/downloads))

**Platform-Specific Requirements:**

| Platform | Additional Requirements |
|----------|------------------------|
| **🍎 macOS (M1/M2)** | [Docker Desktop for Mac ARM](https://desktop.docker.com/mac/main/arm64/Docker.dmg) |
| **🖥️ Windows** | [Docker Desktop for Windows](https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe) |
| **🐧 Linux** | [Docker Engine](https://docs.docker.com/engine/install/) + [Docker Compose](https://docs.docker.com/compose/install/) |
| **🎮 GPU Users** | [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/summarize-ai.git
   cd summarize-ai
   ```

2. **Build and start the application**
   ```bash
   docker compose up --build
   ```
   
   *First run takes 5-10 minutes to download ML models*

3. **Access the application**
   
   🎨 **Web Interface**: [http://localhost:7860](http://localhost:7860)
   
   🔧 **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
   
   💚 **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

4. **Stop the application**
   ```bash
   # Press Ctrl+C in the terminal, then:
   docker compose down
   ```

### Restart & Rebuild

```bash
# Quick restart (preserves models)
docker compose up

# Full rebuild (downloads models again)
docker compose down
docker compose up --build
```

---

## 📖 Usage Guide

### Web Interface

1. **Open** [http://localhost:7860](http://localhost:7860)
2. **Paste your text** in the input field (supports long articles)
3. **Choose model**:
   - **T5-small**: Faster, good for general text
   - **BART**: Better quality, slower processing
4. **Adjust sliders**:
   - **Max Length**: 50-500 words
   - **Min Length**: 10-200 words
5. **Click Submit** and get your summary!

### API Usage

```bash
# Example API call
curl -X POST "http://localhost:8000/summarize?model_type=t5" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Your long text here...",
       "max_length": 100,
       "min_length": 30
     }'
```

**Response:**
```json
{
  "summary": "Your concise summary here..."
}
```

### Supported Models

| Model | Speed | Quality | Best For |
|-------|--------|---------|----------|
| **T5-small** | ⚡ Fast | ⭐⭐⭐ Good | General text, quick summaries |
| **BART-large-cnn** | 🐌 Slower | ⭐⭐⭐⭐⭐ Excellent | News articles, detailed summaries |

---

## 🏗️ Architecture

```
┌─────────────────┐    HTTP    ┌─────────────────┐
│   Gradio UI     │ ◄──────► │   FastAPI       │
│   (Frontend)    │   :7860    │   (Backend)     │
│                 │            │   :8000         │
└─────────────────┘            └─────────────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │  Transformers   │
                               │     Models      │
                               │   (T5, BART)    │
                               └─────────────────┘
```

### File Structure

```
summarize-ai/
│
├── 🐳 docker-compose.yml       # Multi-container orchestration
├── 📚 README.md                # This file
├── 🚫 .gitignore              # Excludes models and cache
│
├── 🖥️ backend/                 # FastAPI microservice
│   ├── 📦 Dockerfile          # Backend container config
│   ├── 📋 requirements.txt    # Python dependencies
│   └── 📂 app/
│       ├── 🏭 factory.py      # Factory pattern implementation
│       ├── 🔌 api.py          # FastAPI routes and endpoints
│       ├── 📝 summarizer.py   # Model implementations
│       └── 🏗️ base.py         # Abstract base classes
│
└── 🎨 frontend/                # Gradio interface
    ├── 📦 Dockerfile          # Frontend container config
    ├── 📋 requirements.txt    # Python dependencies
    └── 📂 app/
        └── 🖼️ interface.py    # Gradio web interface
```

---

## 🔧 Development

### Adding New Models

1. **Extend the base class** in `backend/app/summarizer.py`:
   ```python
   class GPTSummarizer(BaseSummarizer):
       def __init__(self, model_name: str = "gpt2"):
           self.pipeline = pipeline("summarization", model=model_name)
       
       def summarize(self, text: str, max_length: int = 100, min_length: int = 30) -> str:
           # Your implementation here
   ```

2. **Register in factory** (`backend/app/factory.py`):
   ```python
   elif model_type == "gpt":
       return GPTSummarizer()
   ```

3. **Update the frontend** (`frontend/app/interface.py`):
   ```python
   MODEL_CHOICES = {
       "T5-small": "t5",
       "BART": "bart", 
       "GPT-2": "gpt"  # Add new option
   }
   ```

### Running Tests

```bash
# Run backend tests
docker exec summarize-ai-backend-1 python -m pytest

# Run frontend tests  
docker exec summarize-ai-frontend-1 python -m pytest
```

### Viewing Logs

```bash
# All logs
docker compose logs

# Backend only
docker compose logs backend -f

# Frontend only
docker compose logs frontend -f
```

---

## 🐛 Troubleshooting

### Common Issues

**🔴 "Connection reset by peer"**
```bash
# Solution: Check if containers are running
docker compose ps

# Restart if needed
docker compose down && docker compose up --build
```

**🔴 "Docker daemon not running"**
```bash
# Solution: Start Docker Desktop
# Mac: Open Docker Desktop app
# Windows: Start Docker Desktop from Start menu
```

**🔴 "Models not loading"**
```bash
# Solution: Clear cache and rebuild
docker compose down -v  # Removes volumes
docker compose up --build
```

**🔴 "Port already in use"**
```bash
# Solution: Stop conflicting services
lsof -ti:7860 | xargs kill -9  # Kill process on port 7860
lsof -ti:8000 | xargs kill -9  # Kill process on port 8000
```

### Performance Optimization

**For faster startup:**
- Keep containers running: `docker compose up` (not `--build`)
- Models are cached after first download

**For GPU acceleration:**
- Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Update `docker-compose.yml` with GPU support

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** this repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Fork and clone your fork
git clone https://github.com/yourusername/summarize-ai.git
cd summarize-ai

# Create feature branch
git checkout -b feature/new-model

# Make your changes and test
docker compose up --build

# Commit and push
git add .
git commit -m "Add new summarization model"
git push origin feature/new-model
```

---

## 📋 Requirements

### System Requirements
- **Memory**: 4GB RAM minimum (8GB+ recommended)
- **Storage**: 2GB free space (for Docker images and models)
- **Network**: Internet connection (for model downloads on first run)

### Supported Platforms
- ✅ **macOS** (Intel & Apple Silicon)
- ✅ **Windows** 10/11 (with WSL2)
- ✅ **Linux** (Ubuntu, Debian, CentOS, etc.)

---

## 📝 API Reference

### POST /summarize

**Parameters:**
```json
{
  "text": "string (required)",
  "max_length": "integer (50-500, default: 150)",
  "min_length": "integer (10-200, default: 30)"
}
```

**Query Parameters:**
- `model_type`: `t5` | `bart` (default: `t5`)

**Response:**
```json
{
  "summary": "Generated summary text..."
}
```

### GET /health

**Response:**
```json
{
  "status": "ok"
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Hugging Face](https://huggingface.co/)** for the Transformers library
- **[FastAPI](https://fastapi.tiangolo.com/)** for the excellent web framework  
- **[Gradio](https://gradio.app/)** for the intuitive UI components
- **[Docker](https://docker.com/)** for containerization platform

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/summarize-ai?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/summarize-ai?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/summarize-ai)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/summarize-ai)

---

**⭐ If this project helped you, please give it a star!**

**🐛 Found a bug? [Open an issue](https://github.com/yourusername/summarize-ai/issues)**

**💡 Have an idea? [Start a discussion](https://github.com/vidyadharbendre/summarize-ai/discussions)**