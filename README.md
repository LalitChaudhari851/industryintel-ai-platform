---
title: AI Industry Intelligence Platform
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# AI Industry Intelligence Platform

Autonomous multi-agent research swarm designed to analyze industries, markets, competitors, technologies, and trends. Operates entirely on local models to guarantee data privacy and sovereignty.

---

## 🌟 Key Features

1. **Local Inference Swarm**: Uses **Qwen3-8B-Instruct** (via Ollama) as the primary cognitive engine for agent reasoning and **Llama 3.1 8B Instruct** as the automatic fallback engine.
2. **FAISS Semantic Memory**: Chunks and indexes collected web sources dynamically using local **BAAI/bge-base-en-v1.5** embeddings, allowing subsequent agent steps (like Analyst and Critic) to query historical context.
3. **BGE Reranking**: Utilizes the **BAAI/bge-reranker-base** cross-encoder model to score the relevance of web sources before indexing.
4. **Self-Correcting Critique Loop**: The **Critic Agent** audits analysis findings against citations, routing back to the research phase if quality metrics fall below a 70% threshold.
5. **SSE Progress Stream**: FastAPI publishes real-time agent updates using Server-Sent Events (SSE).
6. **Premium Streamlit Dashboard**: A complete, responsive UI with glassmorphic cards, active agent workflow maps, and bibliography reference views.
7. **Factual Quality Evaluation Framework**: A standalone command-line evaluation framework calculating domain diversity, citation density, grounding confidence, and LLM-as-judge accuracy.

---

## 🛠️ Architecture Overview

For in-depth details of design patterns, refer to [docs/architecture.md](docs/architecture.md).

```
+------------------+     HTTP     +------------------+     Graph     +------------------+
|   Streamlit UI   | -----------> | FastAPI Backend  | ------------> |  LangGraph Flow  |
|  (Port 8501)     |              |  (Port 8000)     |               | (Planner->Writer)|
+------------------+              +------------------+               +------------------+
                                                                               |
                                                                               v
                                                                     +------------------+
                                                                     |   Ollama LLM     |
                                                                     |   FAISS + BGE    |
                                                                     +------------------+
```

---

## 🚀 Quick Start (Docker Compose)

The easiest way to start the entire system (Ollama + Backend + Frontend) is via Docker Compose:

1. Copy `.env.example` to `.env` and add your **Tavily API Key**:
   ```bash
   cp .env.example .env
   ```
2. Start the services:
   ```bash
   docker compose up --build
   ```
3. Open your browser:
   - **Streamlit Frontend Dashboard**: [http://localhost:8501](http://localhost:8501)
   - **FastAPI backend Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

*Note: The initial run will download local embeddings and LLM weights (~5.5 GB total). Please be patient.*

---

## 💻 Native Development Setup (Non-Docker)

Refer to [docs/deployment.md](docs/deployment.md) for full instructions on running the FastAPI backend and Streamlit frontend natively on your system.

---

## 📊 Quality Evaluation Framework

To run batch evaluations of the agent swarm against the golden test cases:

1. Set your `TAVILY_API_KEY` env var.
2. Execute the evaluation runner:
   ```bash
   python -m evaluation.runner
   ```
This will run the test cases, apply local metric formulas, invoke the LLM accuracy judge, and produce a summary quality report in your console.
