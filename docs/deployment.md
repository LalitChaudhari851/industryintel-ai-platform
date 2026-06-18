# Deployment & Configuration Guide

This guide explains how to configure, run, and deploy the **AI Industry Intelligence Platform** using Docker Compose or a local python setup.

---

## 1. Environment Variables Configuration

Copy `.env.example` to `.env` and configure key variables:

```bash
cp .env.example .env
```

### Essential Settings
- `TAVILY_API_KEY`: Required for Researcher agent web scraping. Get one from [Tavily Search API](https://tavily.com).
- `OLLAMA_BASE_URL`: Base URL for local Ollama inference. Defaults to `http://localhost:11434` (external local run) or `http://ollama:11434` (inside Docker Compose).

### Observability Settings (LangSmith - Optional)
To enable agent tracing:
```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2-your-api-key
LANGSMITH_PROJECT=ai-industry-intelligence
```

---

## 2. Docker Compose Deployment (Recommended)

The platform provides a fully containerized **3-service stack** containing:
1. **Ollama**: Preconfigured to automatically download and serve `qwen3:8b` (primary) and `llama3.1:8b` (fallback).
2. **Backend**: FastAPI web service handling graphs and database logic on port `8000`.
3. **Frontend**: Streamlit interactive dashboard UI on port `8501`.

### Build & Start Stack
Run:
```bash
docker compose up --build
```

The services will be available at:
- **Streamlit Frontend**: [http://localhost:8501](http://localhost:8501)
- **FastAPI API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

*Note: The first launch will download the local BGE embedding and cross-encoder reranking models (~1.5 GB total) and pull the Ollama LLM weights. Cold start might take several minutes depending on internet connection speed.*

---

## 3. Local Non-Docker Deployment (Development Mode)

If you prefer to run the system natively:

### Step 3.1: Install & Launch Ollama
1. Download and install [Ollama](https://ollama.com).
2. Start the Ollama server:
   ```bash
   ollama serve
   ```
3. Pull required models:
   ```bash
   ollama pull qwen3:8b
   ollama pull llama3.1:8b
   ```

### Step 3.2: Launch FastAPI Backend
1. Install package dependencies:
   ```bash
   pip install -e .
   ```
2. Start FastAPI server using uvicorn:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Step 3.3: Launch Streamlit Frontend
1. Launch Streamlit server:
   ```bash
   streamlit run frontend/app.py --server.port 8501
   ```
2. Open [http://localhost:8501](http://localhost:8501) in your browser.
