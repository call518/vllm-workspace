#!/bin/bash

# Usage: ./run_vllm.sh [MODEL_NAME]
# Example: ./run_vllm.sh Qwen/Qwen2.5-3B-Instruct

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: $0 [MODEL_NAME]"
  echo "  MODEL_NAME: HuggingFace model ID (default: Qwen/Qwen2.5-3B-Instruct)"
  exit 0
fi

MODEL_NAME=${1:-"Qwen/Qwen2.5-3B-Instruct"}

# Check available disk space
ROOT_SPACE=$(df -h / | awk 'NR==2 {print $4}')
echo "Available space on root: $ROOT_SPACE"
#DOCKER_SPACE=$(df -h /var/lib/docker | awk 'NR==2 {print $4}')
#echo "Available space on Docker volume: $DOCKER_SPACE"

# Check if Docker daemon is running
docker info > /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "Docker daemon is not running. Starting Docker..."
  sudo systemctl start docker
fi

# Clean up Docker resources before running
echo "Cleaning up Docker resources..."
sudo docker system prune -f

# Create HuggingFace cache directory for model downloads
HF_CACHE_DIR="${HOME}/.cache/huggingface"
mkdir -p "${HF_CACHE_DIR}"

# Set HUGGING_FACE_HUB_TOKEN
HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)

# Run vLLM Docker container
echo "Starting vLLM container..."
DOCKER_IMAGE="vllm/vllm-openai:v0.9.2"
#DOCKER_IMAGE="vllm/vllm-openai:v0.9.1"
#DOCKER_IMAGE="vllm/vllm-openai:v0.9.0.1"

#sudo docker run -d \
#    --name vLLM-Workspace \
#    --runtime nvidia \
#    --gpus all \
#    -p 5000:5000 \
#    --ipc=host \
#    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
#    -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
#    ${DOCKER_IMAGE} \
#    --model "${MODEL_NAME}" \
#    --dtype auto \
#    --tensor-parallel-size 1 \
#    --host "0.0.0.0" \
#    --port 5000 \
#    --gpu-memory-utilization 1.0 \
#    --served-model-name "${MODEL_NAME}" \
#    --max-num-batched-tokens 8192 \
#    --max-num-seqs 256 \
#    --max-model-len 8192 \
#    --trust-remote-code

docker_args=(
    --name vLLM-Workspace \
    --runtime nvidia \
    --gpus all \
    #--network="host" \
    -p 5000:5000 \
    --ipc=host \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
    ${DOCKER_IMAGE} \
    # --load-format gguf \
    --model "${MODEL_NAME}" \
    --dtype auto \
    --host "0.0.0.0" \
    --port 5000 \
    --gpu-memory-utilization 0.9 \
    #--cpu-offload-gb 16 \
    --served-model-name "${MODEL_NAME}" \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --max-model-len 8192 \
)

sudo docker run -d "${docker_args[@]}"
