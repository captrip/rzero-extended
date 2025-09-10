#!/usr/bin/env python3
"""
FastAPI Backend for R-Zero Training UI
Provides REST API and WebSocket endpoints for training management
"""

import asyncio
import json
import time
from typing import Dict, List, Optional
from pathlib import Path
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
from training.ui_training_manager import UITrainingManager, UITrainingConfig, TrainingStatus


# Pydantic models for API
class TrainingConfigRequest(BaseModel):
    base_model: str
    experiment_name: str
    storage_path: str = "./storage"
    huggingface_name: str = "test-user"
    num_iterations: int = 5
    questions_per_iteration: int = 1000
    max_steps_questioner: int = 6
    max_steps_solver: int = 20
    learning_rate: float = 1e-6
    batch_size: int = 4
    save_steps: int = 2
    enable_langsmith: bool = True
    langsmith_project: str = "r-zero-training"


class ExperimentInfo(BaseModel):
    experiment_id: str
    experiment_name: str
    status: str
    start_time: float
    config: Dict


class TrainingResponse(BaseModel):
    success: bool
    message: str
    experiment_id: Optional[str] = None


# FastAPI app
app = FastAPI(title="R-Zero Training API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global training manager
training_manager: Optional[UITrainingManager] = None
active_websockets: List[WebSocket] = []


class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        print(f"WebSocket connected. Total connections: {len(self.connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)
        print(f"WebSocket disconnected. Total connections: {len(self.connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.connections:
            return
        
        disconnected = []
        for connection in self.connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


websocket_manager = WebSocketManager()


def status_callback(status: TrainingStatus):
    """Callback function for training status updates"""
    message = {
        "type": "training_status",
        "data": {
            "experiment_id": status.experiment_id,
            "status": status.status,
            "current_iteration": status.current_iteration,
            "total_iterations": status.total_iterations,
            "current_phase": status.current_phase,
            "progress_percentage": status.progress_percentage,
            "current_step": status.current_step,
            "total_steps": status.total_steps,
            "loss": status.loss,
            "learning_rate": status.learning_rate,
            "eta_seconds": status.eta_seconds,
            "error_message": status.error_message,
            "model_paths": status.model_paths
        }
    }
    
    # Schedule broadcast (since this might be called from another thread)
    asyncio.create_task(websocket_manager.broadcast(message))


# API Routes

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "R-Zero Training API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/training/start", response_model=TrainingResponse)
async def start_training(config_request: TrainingConfigRequest):
    """Start a new training experiment"""
    global training_manager
    
    try:
        # Check if training is already running
        if training_manager and training_manager.is_training:
            raise HTTPException(status_code=400, detail="Training is already running")
        
        # Create training configuration
        config = UITrainingConfig(
            base_model=config_request.base_model,
            experiment_name=config_request.experiment_name,
            storage_path=config_request.storage_path,
            huggingface_name=config_request.huggingface_name,
            num_iterations=config_request.num_iterations,
            questions_per_iteration=config_request.questions_per_iteration,
            max_steps_questioner=config_request.max_steps_questioner,
            max_steps_solver=config_request.max_steps_solver,
            learning_rate=config_request.learning_rate,
            batch_size=config_request.batch_size,
            save_steps=config_request.save_steps,
            enable_langsmith=config_request.enable_langsmith,
            langsmith_project=config_request.langsmith_project
        )
        
        # Create training manager
        training_manager = UITrainingManager(config, status_callback)
        
        # Start training in background
        asyncio.create_task(training_manager.run_full_training_pipeline())
        
        return TrainingResponse(
            success=True,
            message="Training started successfully",
            experiment_id=training_manager.current_status.experiment_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/pause")
async def pause_training():
    """Pause current training"""
    global training_manager
    
    if not training_manager or not training_manager.is_training:
        raise HTTPException(status_code=400, detail="No training is currently running")
    
    training_manager.pause_training()
    return {"success": True, "message": "Training paused"}


@app.post("/training/resume")
async def resume_training():
    """Resume paused training"""
    global training_manager
    
    if not training_manager:
        raise HTTPException(status_code=400, detail="No training session found")
    
    training_manager.resume_training()
    return {"success": True, "message": "Training resumed"}


@app.post("/training/stop")
async def stop_training():
    """Stop current training"""
    global training_manager
    
    if not training_manager:
        raise HTTPException(status_code=400, detail="No training session found")
    
    training_manager.stop_training()
    return {"success": True, "message": "Training stop requested"}


@app.get("/training/status")
async def get_training_status():
    """Get current training status"""
    global training_manager
    
    if not training_manager:
        return {
            "experiment_id": None,
            "status": "idle",
            "current_iteration": 0,
            "total_iterations": 0,
            "current_phase": "idle",
            "progress_percentage": 0.0,
            "current_step": 0,
            "total_steps": 0,
            "loss": None,
            "learning_rate": None,
            "eta_seconds": None,
            "error_message": None,
            "model_paths": None
        }
    
    status = training_manager.get_status()
    return {
        "experiment_id": status.experiment_id,
        "status": status.status,
        "current_iteration": status.current_iteration,
        "total_iterations": status.total_iterations,
        "current_phase": status.current_phase,
        "progress_percentage": status.progress_percentage,
        "current_step": status.current_step,
        "total_steps": status.total_steps,
        "loss": status.loss,
        "learning_rate": status.learning_rate,
        "eta_seconds": status.eta_seconds,
        "error_message": status.error_message,
        "model_paths": status.model_paths
    }


@app.get("/training/history")
async def get_training_history():
    """Get training history"""
    global training_manager
    
    if not training_manager:
        return {"history": []}
    
    history = training_manager.get_training_history()
    return {"history": history}


@app.get("/experiments")
async def list_experiments():
    """List all training experiments"""
    # In a real implementation, this would query a database
    # For now, return the current experiment if exists
    experiments = []
    
    if training_manager:
        experiments.append({
            "experiment_id": training_manager.current_status.experiment_id,
            "experiment_name": training_manager.config.experiment_name,
            "status": training_manager.current_status.status,
            "start_time": time.time(),  # Would be actual start time
            "config": training_manager.config.__dict__
        })
    
    return {"experiments": experiments}


@app.get("/models")
async def list_models():
    """List available trained models"""
    storage_path = Path("./storage/models")
    models = []
    
    if storage_path.exists():
        for model_dir in storage_path.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "training_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        
                        models.append({
                            "name": model_dir.name,
                            "path": str(model_dir),
                            "metadata": metadata
                        })
                    except:
                        models.append({
                            "name": model_dir.name,
                            "path": str(model_dir),
                            "metadata": {}
                        })
    
    return {"models": models}


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket)
    
    try:
        # Send initial status
        if training_manager:
            status = training_manager.get_status()
            initial_message = {
                "type": "training_status",
                "data": {
                    "experiment_id": status.experiment_id,
                    "status": status.status,
                    "current_iteration": status.current_iteration,
                    "total_iterations": status.total_iterations,
                    "current_phase": status.current_phase,
                    "progress_percentage": status.progress_percentage,
                    "current_step": status.current_step,
                    "total_steps": status.total_steps,
                    "loss": status.loss,
                    "learning_rate": status.learning_rate,
                    "eta_seconds": status.eta_seconds,
                    "error_message": status.error_message,
                    "model_paths": status.model_paths
                }
            }
            await websocket.send_json(initial_message)
        
        # Keep connection alive
        while True:
            try:
                # Wait for client messages (ping/pong, etc.)
                data = await websocket.receive_text()
                # Echo back or handle client messages if needed
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
    
    except WebSocketDisconnect:
        pass
    finally:
        websocket_manager.disconnect(websocket)


# Static files (for serving frontend if needed)
frontend_path = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


def main():
    """Run the FastAPI server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="R-Zero Training API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"Starting R-Zero Training API on {args.host}:{args.port}")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()