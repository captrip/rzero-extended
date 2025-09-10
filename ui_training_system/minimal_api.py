#!/usr/bin/env python
"""
Minimal API for R-Zero UI Model Selection Testing
Focuses on the model selection functionality without heavy training dependencies
"""

import json
import time
import os
import sys
from typing import Dict, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ['LANGSMITH_API_KEY'] = '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'
os.environ['STORAGE_PATH'] = 'storage'


# Pydantic models for API
class ChallengerConfig(BaseModel):
    mode: str  # 'preset', 'external', 'custom'
    preset: Optional[str] = None
    provider: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class SolverConfig(BaseModel):
    mode: str  # 'preset', 'custom'
    preset: Optional[str] = None
    provider: Optional[str] = None
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class TrainingConfigRequest(BaseModel):
    base_model: str
    challenger_config: Optional[ChallengerConfig] = None
    solver_config: Optional[SolverConfig] = None
    experiment_name: str


# FastAPI app
app = FastAPI(title="R-Zero Minimal API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "R-Zero Minimal API for Model Selection Testing", 
        "version": "1.0.0",
        "features": [
            "Model selection testing",
            "Challenger/Solver configuration",
            "LangSmith integration ready"
        ]
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "langsmith_configured": bool(os.getenv('LANGSMITH_API_KEY')),
        "storage_path": os.getenv('STORAGE_PATH', 'storage')
    }


@app.post("/api/test-challenger")
async def test_challenger(config: ChallengerConfig):
    """Test challenger model connection"""
    try:
        print(f"Testing challenger: {config.model_dump()}")
        
        # Simulate different test scenarios
        if config.mode == 'preset':
            if config.preset in ['local_llama', 'ollama_llama', 'ollama_qwen']:
                # Simulate successful local connection
                return {"success": True, "message": f"Connected to {config.preset}"}
            else:
                return {"success": False, "error": f"Unknown preset: {config.preset}"}
        
        elif config.mode == 'external':
            if config.provider == 'openai':
                if not config.api_key or len(config.api_key) < 10:
                    return {"success": False, "error": "Valid OpenAI API key required"}
                # Simulate API key validation
                if config.api_key.startswith('sk-'):
                    return {"success": True, "message": f"Connected to OpenAI {config.model_name}"}
                else:
                    return {"success": False, "error": "Invalid OpenAI API key format"}
            
            elif config.provider == 'anthropic':
                if not config.api_key or len(config.api_key) < 10:
                    return {"success": False, "error": "Valid Anthropic API key required"}
                if config.api_key.startswith('sk-ant-'):
                    return {"success": True, "message": f"Connected to Anthropic {config.model_name}"}
                else:
                    return {"success": False, "error": "Invalid Anthropic API key format"}
            
            elif config.provider == 'custom':
                if not config.base_url or not config.model_name:
                    return {"success": False, "error": "Base URL and model name required"}
                return {"success": True, "message": f"Connected to custom API {config.model_name}"}
        
        elif config.mode == 'custom':
            if not config.base_url or not config.model_name:
                return {"success": False, "error": "Base URL and model name required"}
            return {"success": True, "message": f"Connected to {config.model_name}"}
        
        return {"success": False, "error": "Invalid configuration"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/test-solver")
async def test_solver(config: SolverConfig):
    """Test solver model connection"""
    try:
        print(f"Testing solver: {config.model_dump()}")
        
        if config.mode == 'preset':
            preset_models = ['local_llama', 'ollama_llama', 'distilgpt2', 'flan_t5_small']
            if config.preset in preset_models:
                return {"success": True, "message": f"Connected to {config.preset}"}
            else:
                return {"success": False, "error": f"Unknown preset: {config.preset}"}
        
        elif config.mode == 'custom':
            if not config.base_url or not config.model_name:
                return {"success": False, "error": "Base URL and model name required"}
            return {"success": True, "message": f"Connected to {config.model_name}"}
        
        return {"success": False, "error": "Invalid configuration"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/training/start")
async def start_training(config_request: TrainingConfigRequest):
    """Mock training start - just for UI testing"""
    try:
        print(f"Mock training start: {config_request.model_dump()}")
        
        # Validate configurations
        if not config_request.challenger_config:
            raise HTTPException(status_code=400, detail="Challenger configuration required")
        
        if not config_request.solver_config:
            raise HTTPException(status_code=400, detail="Solver configuration required")
        
        # Mock successful training start
        experiment_id = f"exp_{int(time.time())}"
        
        return {
            "success": True,
            "message": f"Mock training started with experiment: {experiment_id}",
            "experiment_id": experiment_id,
            "config": config_request.model_dump()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/training/status")
async def get_training_status():
    """Mock training status"""
    return {
        "experiment_id": None,
        "status": "idle",
        "current_iteration": 0,
        "total_iterations": 0,
        "current_phase": "idle",
        "progress": 0.0,
        "message": "No training in progress (mock mode)"
    }


@app.get("/available_models")
async def get_available_models():
    """Get list of available models"""
    return {
        "recommended_models": [
            {
                "name": "microsoft/DialoGPT-small",
                "description": "Small conversational model (117M parameters)",
                "size": "117M",
                "recommended_batch_size": 4,
                "memory_requirement": "2GB"
            },
            {
                "name": "distilgpt2",
                "description": "Distilled GPT-2 (82M parameters)",
                "size": "82M",
                "recommended_batch_size": 8,
                "memory_requirement": "1GB"
            },
            {
                "name": "google/flan-t5-small",
                "description": "FLAN-T5 Small (80M parameters)",
                "size": "80M",
                "recommended_batch_size": 8,
                "memory_requirement": "1GB"
            }
        ]
    }


def main():
    print("=" * 60)
    print("R-Zero Minimal API for Model Selection Testing")
    print("=" * 60)
    print("Starting minimal API on 127.0.0.1:8001")
    print()
    print("Features:")
    print("- Challenger model selection testing")
    print("- Solver model selection testing") 
    print("- OpenAI/Anthropic/Custom API configuration")
    print("- Mock training functionality")
    print()
    print("API Documentation: http://127.0.0.1:8001/docs")
    print("=" * 60)
    
    uvicorn.run(
        "minimal_api:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()