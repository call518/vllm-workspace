#!/bin/bash

# Check available disk space on both relevant volumes
ROOT_SPACE=$(df -h / | awk 'NR==2 {print $4}')
DOCKER_SPACE=$(df -h /opt/dlami/nvme | awk 'NR==2 {print $4}')
echo "Available space on root: $ROOT_SPACE"
echo "Available space on Docker volume: $DOCKER_SPACE"

# Check if Docker daemon is running
docker info > /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "Docker daemon is not running. Starting Docker..."
  sudo systemctl start docker
fi

# Clean up Docker resources before running
echo "Cleaning up Docker resources..."
docker system prune -f

# Get absolute paths
CURRENT_DIR=$(pwd)
MODEL_DIR="${CURRENT_DIR}/models"
CONFIG_DIR="${CURRENT_DIR}/config"

# Create a chat template file
TEMPLATE_PATH="${CONFIG_DIR}/qwen_template.json"
echo '{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "<|im_start|>user\n{{query}}<|im_end|>"},
    {"role": "assistant", "content": "<|im_start|>assistant\n{{response}}<|im_end|>"}
  ]
}' > "$TEMPLATE_PATH"

# Run vLLM Docker container for Qwen2.5-14B-Instruct model
echo "Starting vLLM container..."
docker run -it \
    --runtime nvidia \
    --gpus all \
    --network="host" \
    --ipc=host \
    -v "${MODEL_DIR}:/models" \
    -v "${CONFIG_DIR}:/config" \
    vllm/vllm-openai:latest \
    --model "/models/Qwen2.5-14B-Instruct/Qwen2.5-14B-Instruct-Q4_K_M.gguf" \
    --dtype auto \
    --tensor-parallel-size 1 \
    --host "0.0.0.0" \
    --port 5000 \
    --gpu-memory-utilization 0.9 \
    --served-model-name "VLLMQwen2.5-14B" \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --max-model-len 8192 \
    --chat-template /config/qwen_template.json \
    --generation-config /config 