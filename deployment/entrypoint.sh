#!/bin/bash

# Start Ollama service in the background
echo "Starting Ollama server..."
ollama serve &

# Wait for Ollama server to respond
echo "Waiting for Ollama server to start..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 2
done

# Pull configured models
echo "Pulling Qwen3 (primary model)..."
ollama pull qwen3:8b

echo "Pulling Llama 3.1 (fallback model)..."
ollama pull llama3.1:8b

echo "Models loaded successfully. Keeping Ollama server in foreground."
wait
