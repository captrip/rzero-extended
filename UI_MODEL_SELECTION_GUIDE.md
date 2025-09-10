# UI Model Selection Guide

The R-Zero training system now includes flexible model selection in the UI for both challenger and solver models.

## Features Added

### 🤖 Challenger Model Selection
The challenger model generates training questions. You can now choose from:

1. **Preset Models**
   - Local Llama 3.2 (Default)
   - Ollama Llama 3.2
   - Ollama Qwen 2.5

2. **External APIs**
   - **OpenAI GPT**: GPT-4o, GPT-4 Turbo, GPT-3.5, O1 models
   - **Anthropic Claude**: Claude 3.5 Sonnet, Claude 3 Opus, etc.
   - **Custom API**: Any OpenAI-compatible endpoint

3. **Custom Local**
   - Your own local model server
   - Custom base URL and model name

### 🎯 Solver Model Selection
The solver model is what gets trained. Options include:

1. **Preset Models**
   - Local Llama 3.2 (Default)
   - Ollama Llama 3.2
   - DistilGPT-2 (Fast)
   - FLAN-T5 Small

2. **Custom**
   - Custom OpenAI-compatible endpoint
   - Your own model server

## How to Use

### 1. Start the Training API
```bash
cd ui_training_system
python real_training_api.py
```

### 2. Open the Web UI
Open `ui_training_system/real_training_ui.html` in your browser.

### 3. Configure Models

#### Challenger Model Configuration:
1. Click "Configure" to open training settings
2. In the "Challenger Model" section:
   - **For GPT-4**: Choose "External API" → "OpenAI GPT" → Select model → Enter API key
   - **For Claude**: Choose "External API" → "Anthropic Claude" → Select model → Enter API key
   - **For Custom API**: Choose "External API" → "Custom API" → Enter base URL, model name, and API key
   - **For Local**: Choose "Custom" → Enter base URL and model name

3. Click "Test Connection" to verify the model works

#### Solver Model Configuration:
1. In the "Solver Model" section:
   - **For Local Models**: Choose "Preset" → Select from available options
   - **For Custom**: Choose "Custom" → Enter base URL and model name

2. Click "Test Connection" to verify

### 4. Start Training
Once both models are configured and tested, click "Start Real Training".

## Example Configurations

### Using GPT-4o as Challenger
```
Challenger Model:
- Mode: External API
- Provider: OpenAI GPT  
- Model: GPT-4o (Recommended)
- API Key: sk-your-openai-key

Solver Model:
- Mode: Preset
- Model: Local Llama 3.2 (Default)
```

### Using Claude as Challenger
```
Challenger Model:
- Mode: External API
- Provider: Anthropic Claude
- Model: Claude 3.5 Sonnet (Recommended)
- API Key: sk-ant-your-anthropic-key

Solver Model:
- Mode: Custom
- Base URL: http://localhost:11434/v1
- Model Name: llama3.2
```

### Using Custom API as Challenger
```
Challenger Model:
- Mode: External API
- Provider: Custom API
- Base URL: https://api.yourservice.com/v1
- Model Name: your-model-name
- API Key: your-api-key

Solver Model:
- Mode: Preset
- Model: DistilGPT-2 (Fast)
```

## Benefits

### Better Training Quality
- **GPT-4/Claude**: Generate high-quality, challenging questions
- **Domain Expertise**: Advanced models understand complex mathematical concepts
- **Variety**: Different models provide diverse question styles

### Flexibility
- **Cost Control**: Use local models when budget is a concern
- **Performance**: Choose faster models for quick iterations
- **Customization**: Connect to your own fine-tuned models

### Easy Testing
- Connection testing ensures models work before training starts
- Real-time status indicators show connection health
- Error messages help diagnose configuration issues

## Troubleshooting

### Connection Test Fails
1. **Check API Keys**: Make sure your API keys are correct and have sufficient credits
2. **Check URLs**: Verify base URLs are accessible (try in browser)
3. **Check Models**: Ensure model names are spelled correctly
4. **Check Network**: Verify internet connection for external APIs

### Training Fails to Start
1. **Test Both Models**: Make sure both challenger and solver tests pass
2. **Check Logs**: Look at the training logs for specific error messages
3. **Verify Resources**: Ensure sufficient GPU memory for training

### Model Not Responding
1. **Check Server Status**: For local models, ensure servers are running
2. **Check Rate Limits**: External APIs may have rate limiting
3. **Try Different Model**: Some models may be temporarily unavailable

## Advanced Usage

### Environment Variables
Set these for convenience:
```bash
set OPENAI_API_KEY=your-openai-key
set ANTHROPIC_API_KEY=your-anthropic-key
```

### Multiple Configurations
The system saves your last configuration, but you can quickly switch between setups for different experiments.

### Custom Endpoints
Any service that supports OpenAI-compatible chat completions can be used as either challenger or solver.

## What's Different from Before

**Before**: Fixed mock evaluation with simple hardcoded questions
**Now**: Flexible challenger models that generate real, challenging questions

**Before**: Single solver model configuration  
**Now**: Easy switching between different solver models and endpoints

**Before**: No connection testing
**Now**: Built-in connection testing with status indicators

This enhanced system allows for much better training quality while maintaining ease of use through the web interface.