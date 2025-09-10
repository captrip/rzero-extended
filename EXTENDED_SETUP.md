# R-Zero Extended Model Support

This extension adds support for Docker model runners and Ollama instead of OpenAI GPT for evaluation in the R-Zero framework.

## 🚀 New Features

- **Docker Model Runner Support**: Use llama.cpp-based Docker containers via OpenAI-compatible API
- **Ollama Integration**: Use locally hosted Ollama models for evaluation  
- **Configurable Model Providers**: YAML-based configuration for easy switching between providers
- **Fallback Support**: Automatic fallback to other providers if primary fails
- **Extended Results Checker**: Enhanced evaluation script with multi-provider support

## 📋 Setup Instructions

### 1. Install Dependencies

The required dependencies are already in `requirements.txt`. Make sure you have:
- `requests` (for HTTP API calls)
- `pyyaml` (for configuration files)

### 2. Configure Model Providers

#### Option A: Docker Model Runner (llama.cpp)

1. Start your Docker container with llama.cpp server:
```bash
# Example Docker command (adjust as needed)
docker run -p 12434:8080 -v /path/to/models:/models \
  your-llamacpp-image \
  --model /models/llama-3.2-3b-instruct.gguf \
  --host 0.0.0.0 --port 8080
```

2. The API will be available at: `http://localhost:12434/engines/llama.cpp/v1`

#### Option B: Ollama Setup

1. Install Ollama: https://ollama.ai/
2. Pull your desired model:
```bash
ollama pull llama3.2
```
3. Ollama runs on `http://localhost:11434` by default

### 3. Update Configuration

Edit `config/llm_config.yaml` to configure your preferred providers:

```yaml
evaluation_models:
  primary:
    provider: "ollama"  # or "docker" 
    config:
      base_url: "http://localhost:11434"
      model_name: "llama3.2"
  
  fallback:
    - provider: "docker"
      config:
        base_url: "http://localhost:12434/engines/llama.cpp/v1"
        model_name: "llama3.2"
```

## 🔧 Usage

### Extended Results Evaluation

Use the new extended results checker instead of the original:

```bash
# Instead of results_recheck.py, use:
python evaluation/results_recheck_extended.py --model_name "Qwen/Qwen2.5-7B-Instruct"

# With custom config:
python evaluation/results_recheck_extended.py \
  --model_name "Qwen/Qwen2.5-7B-Instruct" \
  --config /path/to/custom_config.yaml
```

### Direct API Usage

```python
from llm_clients import DockerModelRunner, OllamaClient, ModelFactory

# Docker model runner
docker_client = DockerModelRunner(
    base_url="http://localhost:12434/engines/llama.cpp/v1",
    model_name="llama3.2"
)

# Ollama client
ollama_client = OllamaClient(
    base_url="http://localhost:11434", 
    model_name="llama3.2"
)

# Using factory with config
client = ModelFactory.create_client("ollama", {
    "base_url": "http://localhost:11434",
    "model_name": "llama3.2"
})

# Generate response
messages = [{"role": "user", "content": "What is 2+2?"}]
response = client.generate_response(messages)
```

### Test Your Setup

Run the example script to verify everything works:

```bash
python examples/llm_usage_example.py
```

## 📊 Model Provider Comparison

| Provider | Pros | Cons | Use Case |
|----------|------|------|----------|
| **Docker Runner** | Full control, reproducible | Requires Docker setup | Production, specific models |
| **Ollama** | Easy local setup | Limited to supported models | Development, quick testing |
| **OpenAI Compatible** | Standardized API | May need API keys | Cloud services, hosted models |

## 🔍 Configuration Options

### Provider Types
- `openai`: OpenAI-compatible APIs
- `docker`: Docker-hosted llama.cpp containers  
- `ollama`: Local Ollama installations

### Configuration Structure
```yaml
evaluation_models:
  primary:           # Primary evaluation model
    provider: "..."
    config: {...}
  fallback:          # Fallback models (tried in order)
    - provider: "..."
      config: {...}

generation_models:   # For question generation (future use)
  challenger: {...}
  solver: {...}

timeouts:           # Request timeouts
  evaluation: 30
  generation: 60

retry:              # Retry settings
  max_attempts: 3
  backoff_factor: 2.0
```

## 🐛 Troubleshooting

### Docker Model Runner Issues
- Ensure Docker container is running and accessible
- Check port mapping: `-p 12434:8080`
- Verify API endpoint responds: `curl http://localhost:12434/engines/llama.cpp/v1/models`

### Ollama Issues
- Check if Ollama is running: `ollama list`
- Pull model if missing: `ollama pull llama3.2`
- Verify service: `curl http://localhost:11434/api/tags`

### Configuration Issues
- Check YAML syntax
- Verify file paths
- Ensure environment variables are set (for `${VAR_NAME}` substitution)

## 🔄 Migration from Original

1. **Replace evaluation script**:
   ```bash
   # Old
   python evaluation/results_recheck.py
   
   # New  
   python evaluation/results_recheck_extended.py
   ```

2. **Configuration-based setup**:
   - Create `config/llm_config.yaml`
   - Set your preferred providers
   - Remove hardcoded API keys from code

3. **Backwards compatibility**:
   - Original scripts still work
   - OpenAI provider still supported
   - New scripts fallback to defaults if config missing

## 📈 Performance Notes

- **Ollama**: Slower than cloud APIs but no network dependency
- **Docker Runner**: Performance depends on container resources
- **Fallback system**: Adds resilience but may increase latency

## 🤝 Contributing

To add support for new providers:

1. Create new client class inheriting from `BaseLLMClient`
2. Add provider to `ModelFactory.create_client()`
3. Update configuration schema
4. Add tests and documentation

Example provider structure:
```python
from llm_clients.base_client import BaseLLMClient

class MyCustomClient(BaseLLMClient):
    def generate_response(self, messages, **kwargs):
        # Implementation here
        pass
    
    def is_available(self):
        # Health check here  
        pass
```