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

### 2. Create Run Script

Create a shell script to run the container:

```bash
nano run_vllm.sh
```

Add the following content:

```bash
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
    --generation-config /config
```

Make the script executable:

```bash
chmod +x run_vllm.sh
```

### 3. Download the Model

Download the quantized GGUF version of Qwen2.5-14B-Instruct model:

```bash
# Navigate to the models directory
cd models/Qwen2.5-14B-Instruct

# Download the model (adjust URL as needed)
wget https://huggingface.co/TheBloke/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf
```

### 4. Create Configuration File

Create a basic configuration file for generation parameters:

```bash
nano config/config.json
```

Add the following content:

```json
{
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 1024
}
```

### 5. Run the Container

```bash
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

## Utility Scripts and GUI

For easier interaction with the vLLM API, several utility scripts have been created:

### 1. Test Query Script

Create a script to test the chat completion API with pretty formatting:

```bash
#!/bin/bash

# Test vLLM chat completion API
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-14B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }' | jq
```

Save as `test-query.sh` and make it executable with `chmod +x test-query.sh`.

### 2. List Models Script

Create a script to list available models with readable output:

```bash
#!/bin/bash

# List available models in vLLM server
curl http://localhost:5000/v1/models | jq
```

Save as `list-models.sh` and make it executable with `chmod +x list-models.sh`.

### 3. Gradio GUI Interface

A simple web interface using Gradio can be created to interact with vLLM:

```bash
#!/bin/bash

# Install required packages if not already installed
pip install gradio openai

# Create a simple Gradio UI for vLLM
cat > vllm_gradio_ui.py << 'EOF'
import gradio as gr
import argparse
from openai import OpenAI

def format_history(history):
    formatted_history = [{
        "role": "system", 
        "content": "You are a helpful AI assistant."
    }]
    
    for human, assistant in history:
        formatted_history.append({"role": "user", "content": human})
        formatted_history.append({"role": "assistant", "content": assistant})
    
    return formatted_history

def predict(message, history, client, model_name, temperature):
    # Format history to OpenAI chat format
    formatted_history = format_history(history)
    formatted_history.append({"role": "user", "content": message})
    
    # Send request to OpenAI API (vLLM server)
    stream = client.chat.completions.create(
        model=model_name,
        messages=formatted_history,
        temperature=temperature,
        stream=True
    )
    
    # Collect the response
    full_response = ""
    for chunk in stream:
        chunk_content = chunk.choices[0].delta.content or ""
        full_response += chunk_content
        yield full_response

def main():
    parser = argparse.ArgumentParser(description='Gradio UI for vLLM')
    parser.add_argument('--model', type=str, default="Qwen2.5-14B-Instruct", help='Model name')
    parser.add_argument('--api-url', type=str, default="http://localhost:5000/v1", help='API base URL')
    parser.add_argument('--api-key', type=str, default="", help='API key')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    parser.add_argument('--port', type=int, default=7860, help='Port for Gradio UI')
    
    args = parser.parse_args()
    
    # Create OpenAI client
    client = OpenAI(api_key=args.api_key or "not-needed", base_url=args.api_url)
    
    # Create Gradio interface
    with gr.Blocks(title="vLLM Chat Interface") as demo:
        gr.Markdown(f"# Chat with {args.model}")
        gr.Markdown("A simple chat interface powered by vLLM")
        
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(placeholder="Type your message here...", container=False)
        
        with gr.Row():
            submit_btn = gr.Button("Send")
            clear_btn = gr.Button("Clear")
        
        # Temperature slider
        temp_slider = gr.Slider(
            minimum=0.0, maximum=1.0, value=args.temperature, step=0.1, 
            label="Temperature", info="Higher values make output more random"
        )
            
        def respond(message, chat_history, temperature):
            if not message.strip():
                return "", chat_history
            
            chat_history.append((message, ""))
            return "", chat_history, predict(message, chat_history[:-1], client, args.model, temperature)
        
        def update_chatbot(history, response):
            history[-1] = (history[-1][0], response)
            return history
            
        def clear_conversation():
            return []
        
        # Set up event handlers
        msg.submit(
            respond, 
            [msg, chatbot, temp_slider], 
            [msg, chatbot], 
            queue=False
        ).then(
            update_chatbot,
            [chatbot, respond.outputs[-1]],
            [chatbot]
        )
        
        submit_btn.click(
            respond, 
            [msg, chatbot, temp_slider], 
            [msg, chatbot], 
            queue=False
        ).then(
            update_chatbot,
            [chatbot, respond.outputs[-1]],
            [chatbot]
        )
        
        clear_btn.click(clear_conversation, None, chatbot)
    
    # Launch the interface
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)

if __name__ == "__main__":
    main()
EOF

echo "Created vllm_gradio_ui.py"
echo "Starting Gradio UI for vLLM..."

# Run the Gradio UI
python vllm_gradio_ui.py "$@"
```

Save as `gui-demo.sh` and make it executable with `chmod +x gui-demo.sh`.

### Running the GUI

To run the web interface:

```bash
./gui-demo.sh
```

You can customize the settings with command-line parameters:

```bash
./gui-demo.sh --model "Qwen2.5-14B-Instruct" --api-url "http://localhost:5000/v1" --temperature 0.7 --port 7860
```

### Third-Party UIs

Several third-party UI options are also available for vLLM:

1. **nextjs-vllm-ui** - A beautiful ChatGPT-like interface
   - GitHub: https://github.com/yoziru/nextjs-vllm-ui
   - Run with Docker: `docker run --rm -d -p 3000:3000 -e VLLM_URL=http://host.docker.internal:5000 ghcr.io/yoziru/nextjs-vllm-ui:latest`

2. **Open WebUI** - A full-featured web interface that works with vLLM
   - Can be configured to use vLLM as the backend instead of Ollama
   - Example Docker command:
     ```bash
     docker run -d -p 3000:8080 \
       --name open-webui \
       --restart always \
       --env=OPENAI_API_BASE_URL=http://<your-ip>:5000/v1 \
       --env=OPENAI_API_KEY=your-api-key \
       --env=ENABLE_OLLAMA_API=false \
       ghcr.io/open-webui/open-webui:main
     ```

3. **vllm-ui** - A simple Gradio-based interface designed for Vision Language Models
   - GitHub: https://github.com/sammcj/vlm-ui

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
