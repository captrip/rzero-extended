# R-Zero Training System with UI

A comprehensive web-based interface for R-Zero training with real-time monitoring and LangSmith integration.

## Features

- 🎯 **Web-based Training Interface**: Configure and start training sessions through an intuitive web UI
- 📊 **Real-time Progress Tracking**: Live updates on training progress, loss, and metrics
- 📈 **LangSmith Integration**: Comprehensive experiment tracking (replaces WandB)
- 🔄 **Interactive Training Control**: Pause, resume, and stop training sessions
- 📁 **Model Management**: View and manage trained models with metadata
- 📋 **Training History**: Complete history of training sessions and iterations
- 🎨 **Modern UI**: Dark theme with responsive design using Material-UI

## Architecture

```
ui_training_system/
├── backend/
│   ├── api/                     # FastAPI REST API and WebSocket server
│   ├── training/                # Training management with UI integration
│   ├── langsmith_integration/   # LangSmith tracking (replaces WandB)
│   └── monitoring/              # Real-time monitoring utilities
├── frontend/
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── services/          # API services
│   │   └── types/             # TypeScript types
│   └── package.json
└── start_ui_training.py        # Main launcher script
```

## Quick Start

### Prerequisites

1. **Python 3.8+** with required packages
2. **Node.js 16+** and npm (for frontend)
3. **CUDA-compatible GPU** (recommended)
4. **LangSmith API Key** (optional, for experiment tracking)

### Installation

1. **Install Python dependencies**:
```bash
pip install -r backend/requirements.txt
```

2. **Install frontend dependencies**:
```bash
cd frontend
npm install
cd ..
```

3. **Set up LangSmith** (optional):
```bash
export LANGSMITH_API_KEY=your_api_key_here
```

### Running the System

**Start everything with one command**:
```bash
python start_ui_training.py
```

This will:
- Start the backend API server (port 8000)
- Start the frontend development server (port 3000)
- Open your browser to the training interface

**Backend only** (for API development):
```bash
python start_ui_training.py --backend-only
```

**Check system setup**:
```bash
python start_ui_training.py --check-setup
```

## Usage

### 1. Configure Training

1. Open the web interface (usually http://localhost:3000)
2. Go to the "Configuration" tab
3. Set your training parameters:
   - **Base Model**: HuggingFace model name (e.g., `meta-llama/Llama-3.2-1B`)
   - **Experiment Name**: Unique identifier for this training run
   - **Training Parameters**: Iterations, questions per iteration, learning rates, etc.
   - **LangSmith**: Enable/disable experiment tracking

### 2. Start Training

1. Click "Start Training" from the configuration page
2. Training will begin automatically
3. Monitor progress in real-time on the Dashboard

### 3. Monitor Progress

The Dashboard shows:
- **Current Status**: Running, paused, completed, or error
- **Progress**: Overall and phase-specific progress bars
- **Metrics**: Loss, learning rate, ETA
- **Real-time Charts**: Training loss and iteration progress
- **Model Paths**: Latest trained model locations

### 4. Control Training

- **Pause**: Temporarily halt training (can be resumed)
- **Resume**: Continue paused training
- **Stop**: Permanently halt training

### 5. View Results

- **History Tab**: View all past training sessions
- **Models Tab**: Browse and inspect trained models
- **LangSmith**: View detailed experiment tracking (if enabled)

## API Endpoints

### Training Control
- `POST /training/start` - Start new training session
- `POST /training/pause` - Pause current training
- `POST /training/resume` - Resume paused training
- `POST /training/stop` - Stop current training
- `GET /training/status` - Get current status
- `GET /training/history` - Get training history

### Data Management
- `GET /experiments` - List all experiments
- `GET /models` - List trained models
- `GET /health` - Health check

### WebSocket
- `WS /ws` - Real-time training updates

## LangSmith Integration

The system replaces WandB with LangSmith for experiment tracking:

### Features
- **Experiment Tracking**: Complete training run metadata
- **Real-time Metrics**: Loss, learning rate, training steps
- **Model Checkpoints**: Automatic logging of model saves
- **Question Generation**: Track generated questions and evaluation
- **Error Logging**: Capture and track training errors

### Setup
1. Get your LangSmith API key from https://smith.langchain.com
2. Set the environment variable:
   ```bash
   export LANGSMITH_API_KEY=your_key_here
   ```
3. Enable in training configuration
4. View experiments at https://smith.langchain.com

## Configuration Options

### Training Parameters
- `num_iterations`: Number of R-Zero training iterations (default: 5)
- `questions_per_iteration`: Questions generated per iteration (default: 1000)
- `max_steps_questioner`: Training steps for questioner model (default: 6)
- `max_steps_solver`: Training steps for solver model (default: 20)
- `learning_rate`: Learning rate for training (default: 1e-6)
- `batch_size`: Training batch size (default: 4)

### System Settings
- `storage_path`: Where to store models and data (default: "./storage")
- `enable_langsmith`: Enable LangSmith tracking (default: true)
- `langsmith_project`: LangSmith project name (default: "r-zero-training")

## Development

### Backend Development

```bash
# Start backend in development mode
cd backend/api
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

```bash
# Start frontend development server
cd frontend
npm run dev
```

### Adding New Features

1. **Backend**: Add endpoints in `backend/api/main.py`
2. **Training Logic**: Extend `backend/training/ui_training_manager.py`
3. **Frontend**: Add components in `frontend/src/components/`
4. **LangSmith**: Extend tracking in `backend/langsmith_integration/`

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports with `--backend-port` and `--frontend-port`
2. **GPU memory**: Reduce `batch_size` in training configuration
3. **LangSmith errors**: Check API key and network connection
4. **Frontend build issues**: Delete `node_modules` and run `npm install`

### Debug Mode

```bash
# Enable verbose logging
export PYTHONPATH=. 
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python start_ui_training.py
```

### Logs

- Backend logs: Console output from uvicorn server
- Frontend logs: Browser developer console
- Training logs: Displayed in UI and LangSmith

## Examples

### Quick Test Training

```bash
# Run a quick 2-iteration test
python start_ui_training.py --example
```

### Custom Configuration

```python
from backend.training.ui_training_manager import UITrainingManager, UITrainingConfig

config = UITrainingConfig(
    base_model="microsoft/DialoGPT-medium",
    experiment_name="custom-test",
    num_iterations=3,
    questions_per_iteration=50,
    enable_langsmith=True
)

manager = UITrainingManager(config)
await manager.run_full_training_pipeline()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is part of the R-Zero training system. See main repository for license information.

---

**Happy Training! 🚀**

For issues and questions, please refer to the main R-Zero repository.