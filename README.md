# Setting Up vLLM with Qwen2.5-14B-Instruct on AWS

This guide walks through the process of setting up and running the Qwen2.5-14B-Instruct model using vLLM on AWS EC2 instances, with solutions for common storage and configuration issues.

## Prerequisites

- AWS EC2 instance with NVIDIA GPU (recommended: g4dn.xlarge or higher)
- NVIDIA drivers installed
- Docker and NVIDIA Container Toolkit installed
- At least 100GB of storage space (preferably more)

## Initial Setup

### 1. Create EC2 Instance

Use an AWS Deep Learning AMI (DLAMI) with GPU support to simplify driver installation:

```bash
# Example instance types
# g4dn.xlarge - 1 GPU, 4 vCPUs, 16 GB RAM
# g4dn.2xlarge - 1 GPU, 8 vCPUs, 32 GB RAM
# g5.xlarge - 1 GPU, 4 vCPUs, 16 GB RAM
```

Attach a large EBS volume (300GB recommended) to your instance.

### 2. Configure Docker for NVIDIA Support

Ensure Docker and NVIDIA Container Toolkit are properly installed:

```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker installation
docker --version

# Check NVIDIA Docker integration
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## Storage Configuration

AWS instances often have multiple volumes with different sizes. To handle large models, configure Docker to use the larger storage volume:

### 1. Identify Available Volumes

```bash
# Check disk layout
lsblk

# Check available space
df -h
```

Typically AWS DLAMIs have two volumes:
- NVMe root volume (e.g., `/dev/nvme0n1`)
- Larger NVMe storage volume (e.g., `/dev/nvme1n1` mounted at `/opt/dlami/nvme`)

### 2. Configure Docker to Use Larger Volume

```bash
# Create a directory for Docker data
sudo mkdir -p /opt/dlami/nvme/docker

# Configure Docker to use this directory
sudo nano /etc/docker/daemon.json
```

Add the `data-root` setting to your `daemon.json` file:

```json
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "data-root": "/opt/dlami/nvme/docker"
}
```

Restart Docker to apply changes:

```bash
sudo systemctl restart docker
```

### 3. Verify Docker Storage Location

```bash
# Check Docker's root directory
docker info | grep "Docker Root Dir"

# Should show: Docker Root Dir: /opt/dlami/nvme/docker
```

## Running the vLLM Container

### 1. Create Directory Structure

```bash
# Create directories for models and configuration
mkdir -p models/Qwen2.5-14B-Instruct
mkdir -p config
```

### 2. Download the Model

Download the quantized GGUF version of Qwen2.5-14B-Instruct model:

```bash
# Navigate to the models directory
cd models/Qwen2.5-14B-Instruct

# Download the model (adjust URL as needed)
wget https://huggingface.co/TheBloke/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf
```

## EC2 Instance Management

### After Starting/Restarting EC2 Instance

When you start or restart your EC2 instance, you need to reconfigure Docker to use the correct storage location and permissions. Use the provided setup script:

```bash
# First, run the setup script
./setup_docker.sh

# Then start your vLLM container
./run_vllm.sh
```

The setup script:
1. Configures Docker to use the larger NVMe volume
2. Sets proper permissions for Docker directories
3. Ensures Docker daemon is configured correctly
4. Pulls the latest vLLM Docker image
5. Cleans up any stale Docker resources

This setup needs to be run after each EC2 instance restart because:
- Docker service restarts with default settings
- Storage permissions might need to be reestablished
- Docker daemon configuration needs to be verified

### 2. Run the container

Open a shell script to run the container:

```bash
chmod +x run_vllm.sh
./run_vllm.sh
```

## Troubleshooting

### "No space left on device" Error

If you encounter space issues despite configuring Docker to use the larger volume:

1. Verify Docker is using the correct storage location:
   ```bash
   docker info | grep "Docker Root Dir"
   ```

2. Clean Docker resources completely:
   ```bash
   docker system prune -a -f --volumes
   ```

3. Check available space on both volumes:
   ```bash
   df -h
   ```

4. Try pulling the image separately:
   ```bash
   docker pull vllm/vllm-openai:latest
   ```

Here are the commands to fix this problem:

```bash
# Stop Docker service
sudo systemctl stop docker

# Create a new directory for Docker data
sudo mkdir -p /opt/dlami/nvme/docker

# Move existing Docker data (optional, but preserves existing images)
sudo rsync -aP /var/lib/docker/ /opt/dlami/nvme/docker/

# Create or edit Docker daemon config
sudo vim /etc/docker/daemon.json

# Add this configuration:
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "data-root": "/opt/dlami/nvme/docker"
}

# Restart Docker service
sudo systemctl start docker

# Verify Docker is using the new location
docker info | grep "Docker Root Dir"
```

### GGUF Model Loading Issues

If you encounter issues loading the GGUF model:

1. Make sure you're using absolute paths for volume mounts:
   ```bash
   CURRENT_DIR=$(pwd)
   MODEL_DIR="${CURRENT_DIR}/models"
   CONFIG_DIR="${CURRENT_DIR}/config"
   ```

2. Try the direct Hugging Face approach if GGUF isn't working:
   ```bash
   docker run -it \
       --runtime nvidia \
       --gpus all \
       --network="host" \
       --ipc=host \
       -v "${CONFIG_DIR}:/config" \
       vllm/vllm-openai:latest \
       --model "Qwen/Qwen2.5-14B-Instruct" \
       --quantization awq \
       --host "0.0.0.0" \
       --port 5000 \
       --gpu-memory-utilization 0.9 \
       --served-model-name "VLLMQwen2.5-14B" \
       --max-num-batched-tokens 8192 \
       --max-num-seqs 256 \
       --max-model-len 8192 \
       --generation-config /config
   ```

3. Check vLLM documentation for specific GGUF loading requirements:
   [vLLM GGUF Support](https://docs.vllm.ai/en/latest/models/supported_models.html)

### GPU Issues

If Docker can't access the GPUs:

1. Verify NVIDIA drivers are working:
   ```bash
   nvidia-smi
   ```

2. Check NVIDIA Container Toolkit configuration:
   ```bash
   sudo nano /etc/docker/daemon.json
   ```
   
   Ensure it contains the NVIDIA runtime configuration.

3. Restart Docker:
   ```bash
   sudo systemctl restart docker
   ```

## Using the API

Once the server is running, you can make API calls to generate text:

### Test the Completions Endpoint

```bash
curl http://localhost:5000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "VLLMQwen2.5-14B",
    "prompt": "Write a short poem about artificial intelligence:",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Test the Chat Completions Endpoint

```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "VLLMQwen2.5-14B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What are the three laws of robotics?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Check Available Models

```bash
curl http://localhost:5000/v1/models
```

### Test Server Health

```bash
curl http://localhost:5000/health
```

### Test With Streaming Response

For streaming responses (similar to how ChatGPT provides tokens incrementally):

```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "VLLMQwen2.5-14B",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": true
  }'
```

You can also access the OpenAPI documentation by opening http://localhost:5000/docs in your browser, which will show you all the available endpoints and parameters.

## Performance Tuning

Adjust these parameters in the `run_vllm.sh` script based on your hardware:

- `--gpu-memory-utilization`: Value between 0 and 1 (default: 0.9)
- `--max-num-batched-tokens`: Increase for higher throughput, decrease if out of memory
- `--max-num-seqs`: Maximum number of sequences in the batch
- `--max-model-len`: Maximum sequence length
- `--tensor-parallel-size`: Set to number of GPUs if using multiple GPUs

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen2.5 Model Information](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)
