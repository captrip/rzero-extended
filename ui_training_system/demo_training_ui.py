#!/usr/bin/env python3
"""
R-Zero Training UI Demo
Simplified version for demonstration without external dependencies
"""

import asyncio
import json
import time
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import uuid

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Installing FastAPI and dependencies...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]", "websockets"])
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn


@dataclass
class TrainingStatus:
    """Current training status for UI updates"""
    experiment_id: str
    status: str  # "idle", "running", "paused", "completed", "error"
    current_iteration: int
    total_iterations: int
    current_phase: str  # "questioner", "solver", "evaluation"
    progress_percentage: float
    current_step: int
    total_steps: int
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    eta_seconds: Optional[float] = None
    error_message: Optional[str] = None
    model_paths: Optional[Dict[str, str]] = None


@dataclass
class DemoTrainingConfig:
    """Demo training configuration"""
    base_model: str
    experiment_name: str
    storage_path: str = "./storage"
    num_iterations: int = 5
    questions_per_iteration: int = 100
    max_steps_questioner: int = 10
    max_steps_solver: int = 20
    learning_rate: float = 1e-6
    batch_size: int = 4


class TrainingConfigRequest(BaseModel):
    base_model: str
    experiment_name: str
    storage_path: str = "./storage"
    num_iterations: int = 5
    questions_per_iteration: int = 100
    max_steps_questioner: int = 10
    max_steps_solver: int = 20
    learning_rate: float = 1e-6
    batch_size: int = 4


class DemoTrainingManager:
    """Demo training manager that simulates training"""
    
    def __init__(self, config: DemoTrainingConfig, status_callback):
        self.config = config
        self.status_callback = status_callback
        
        # Initialize training state
        self.current_status = TrainingStatus(
            experiment_id=str(uuid.uuid4()),
            status="idle",
            current_iteration=0,
            total_iterations=config.num_iterations,
            current_phase="idle",
            progress_percentage=0.0,
            current_step=0,
            total_steps=0
        )
        
        # Training control
        self.is_training = False
        self.should_pause = False
        self.should_stop = False
        
        # Training history
        self.training_history = []
    
    def update_status(self, **kwargs):
        """Update training status and notify UI"""
        for key, value in kwargs.items():
            if hasattr(self.current_status, key):
                setattr(self.current_status, key, value)
        
        # Call status callback
        if self.status_callback:
            try:
                self.status_callback(self.current_status)
            except Exception as e:
                print(f"Status callback error: {e}")
    
    async def simulate_training_step(self, phase: str, step: int, total_steps: int):
        """Simulate a training step with realistic timing"""
        # Simulate some processing time
        await asyncio.sleep(0.5)  # 500ms per step
        
        # Simulate loss decrease
        loss = max(0.1, 3.0 - (step / total_steps) * 2.5 + (0.1 * (1 - step/total_steps)))
        
        self.update_status(
            current_phase=phase,
            current_step=step,
            total_steps=total_steps,
            loss=loss,
            learning_rate=self.config.learning_rate,
            progress_percentage=(step / total_steps) * 100
        )
    
    async def train_questioner(self, iteration: int):
        """Simulate questioner training"""
        print(f"Training Questioner Iteration {iteration}")
        
        self.update_status(
            current_iteration=iteration,
            current_phase="training_questioner",
            current_step=0,
            total_steps=self.config.max_steps_questioner
        )
        
        for step in range(1, self.config.max_steps_questioner + 1):
            if self.should_stop:
                return None
                
            while self.should_pause and not self.should_stop:
                await asyncio.sleep(1)
                
            await self.simulate_training_step("training_questioner", step, self.config.max_steps_questioner)
        
        # Simulate model save
        model_path = f"./storage/models/questioner_v{iteration}"
        return model_path
    
    async def generate_questions(self, iteration: int):
        """Simulate question generation"""
        print(f"Generating questions for iteration {iteration}")
        
        self.update_status(current_phase="question_generation")
        await asyncio.sleep(1)  # Simulate generation time
        
        # Return mock questions
        return [{"question": f"What is {i} + {i+1}?", "answer": str(i + i + 1)} 
                for i in range(self.config.questions_per_iteration)]
    
    async def train_solver(self, iteration: int, questioner_path: str):
        """Simulate solver training"""
        print(f"Training Solver Iteration {iteration}")
        
        # Generate questions first
        questions = await self.generate_questions(iteration)
        
        # Simulate evaluation
        self.update_status(current_phase="evaluation")
        await asyncio.sleep(1)
        
        # Train solver
        self.update_status(
            current_iteration=iteration,
            current_phase="training_solver",
            current_step=0,
            total_steps=self.config.max_steps_solver
        )
        
        for step in range(1, self.config.max_steps_solver + 1):
            if self.should_stop:
                return None
                
            while self.should_pause and not self.should_stop:
                await asyncio.sleep(1)
                
            await self.simulate_training_step("training_solver", step, self.config.max_steps_solver)
        
        # Simulate model save
        model_path = f"./storage/models/solver_v{iteration}"
        return model_path
    
    async def run_full_training_pipeline(self):
        """Run the complete training pipeline simulation"""
        print("=== Starting Demo R-Zero Training ===")
        self.update_status(status="running", progress_percentage=0.0)
        
        self.is_training = True
        current_questioner_path = self.config.base_model
        current_solver_path = self.config.base_model
        
        try:
            for iteration in range(1, self.config.num_iterations + 1):
                if self.should_stop:
                    print("Training stopped by user")
                    break
                
                iteration_start_time = time.time()
                print(f"\n=== ITERATION {iteration}/{self.config.num_iterations} ===")
                
                try:
                    # Train questioner
                    questioner_path = await self.train_questioner(iteration)
                    if questioner_path is None:  # Stopped
                        break
                    
                    # Train solver
                    solver_path = await self.train_solver(iteration, questioner_path)
                    if solver_path is None:  # Stopped
                        break
                    
                    # Update paths
                    current_questioner_path = questioner_path
                    current_solver_path = solver_path
                    
                    # Record iteration results
                    iteration_time = time.time() - iteration_start_time
                    iteration_result = {
                        "iteration": iteration,
                        "questioner_path": questioner_path,
                        "solver_path": solver_path,
                        "training_time_seconds": iteration_time,
                        "timestamp": time.time()
                    }
                    
                    self.training_history.append(iteration_result)
                    
                    # Update overall progress
                    overall_progress = (iteration / self.config.num_iterations) * 100
                    self.update_status(
                        progress_percentage=overall_progress,
                        model_paths={
                            "questioner": questioner_path,
                            "solver": solver_path
                        }
                    )
                    
                    print(f"Iteration {iteration} completed in {iteration_time:.1f}s")
                    
                except Exception as e:
                    print(f"ERROR in iteration {iteration}: {e}")
                    self.update_status(status="error", error_message=str(e))
                    continue
            
            # Training completed
            self.update_status(status="completed", progress_percentage=100.0)
            print("=== Demo Training Completed ===")
            
        except Exception as e:
            print(f"Training pipeline error: {e}")
            self.update_status(status="error", error_message=str(e))
        
        finally:
            self.is_training = False
    
    def pause_training(self):
        """Pause training"""
        self.should_pause = True
        print("Training paused")
    
    def resume_training(self):
        """Resume training"""
        self.should_pause = False
        print("Training resumed")
    
    def stop_training(self):
        """Stop training"""
        self.should_stop = True
        print("Training stopped")
    
    def get_status(self):
        """Get current training status"""
        return self.current_status
    
    def get_training_history(self):
        """Get training history"""
        return self.training_history


# FastAPI app
app = FastAPI(title="R-Zero Training Demo API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global training manager
training_manager: Optional[DemoTrainingManager] = None
active_websockets: List[WebSocket] = []


def status_callback(status: TrainingStatus):
    """Callback function for training status updates"""
    message = {
        "type": "training_status",
        "data": asdict(status)
    }
    
    # Broadcast to all connected websockets
    asyncio.create_task(broadcast_message(message))


async def broadcast_message(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    if not active_websockets:
        return
    
    disconnected = []
    for websocket in active_websockets:
        try:
            await websocket.send_json(message)
        except:
            disconnected.append(websocket)
    
    # Remove disconnected clients
    for ws in disconnected:
        if ws in active_websockets:
            active_websockets.remove(ws)


# API Routes

@app.get("/")
async def root():
    return {"message": "R-Zero Training Demo API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/training/start")
async def start_training(config_request: TrainingConfigRequest):
    global training_manager
    
    try:
        if training_manager and training_manager.is_training:
            return {"success": False, "message": "Training is already running"}
        
        # Create demo configuration
        config = DemoTrainingConfig(
            base_model=config_request.base_model,
            experiment_name=config_request.experiment_name,
            storage_path=config_request.storage_path,
            num_iterations=config_request.num_iterations,
            questions_per_iteration=config_request.questions_per_iteration,
            max_steps_questioner=config_request.max_steps_questioner,
            max_steps_solver=config_request.max_steps_solver,
            learning_rate=config_request.learning_rate,
            batch_size=config_request.batch_size
        )
        
        # Create training manager
        training_manager = DemoTrainingManager(config, status_callback)
        
        # Start training in background
        asyncio.create_task(training_manager.run_full_training_pipeline())
        
        return {
            "success": True,
            "message": "Demo training started successfully",
            "experiment_id": training_manager.current_status.experiment_id
        }
        
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/training/pause")
async def pause_training():
    global training_manager
    
    if not training_manager or not training_manager.is_training:
        return {"success": False, "message": "No training is currently running"}
    
    training_manager.pause_training()
    return {"success": True, "message": "Training paused"}


@app.post("/training/resume")
async def resume_training():
    global training_manager
    
    if not training_manager:
        return {"success": False, "message": "No training session found"}
    
    training_manager.resume_training()
    return {"success": True, "message": "Training resumed"}


@app.post("/training/stop")
async def stop_training():
    global training_manager
    
    if not training_manager:
        return {"success": False, "message": "No training session found"}
    
    training_manager.stop_training()
    return {"success": True, "message": "Training stop requested"}


@app.get("/training/status")
async def get_training_status():
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
    
    return asdict(training_manager.get_status())


@app.get("/training/history")
async def get_training_history():
    global training_manager
    
    if not training_manager:
        return {"history": []}
    
    return {"history": training_manager.get_training_history()}


@app.get("/experiments")
async def list_experiments():
    experiments = []
    
    if training_manager:
        experiments.append({
            "experiment_id": training_manager.current_status.experiment_id,
            "experiment_name": training_manager.config.experiment_name,
            "status": training_manager.current_status.status,
            "start_time": time.time(),
            "config": asdict(training_manager.config)
        })
    
    return {"experiments": experiments}


@app.get("/models")
async def list_models():
    # Mock model data for demo
    models = [
        {
            "name": "questioner_v1",
            "path": "./storage/models/questioner_v1",
            "metadata": {
                "iteration": 1,
                "role": "questioner",
                "training_steps": 10,
                "timestamp": time.time() - 3600
            }
        },
        {
            "name": "solver_v1", 
            "path": "./storage/models/solver_v1",
            "metadata": {
                "iteration": 1,
                "role": "solver",
                "training_steps": 20,
                "timestamp": time.time() - 3500
            }
        }
    ]
    
    return {"models": models}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    
    try:
        # Send initial status
        if training_manager:
            status = training_manager.get_status()
            initial_message = {
                "type": "training_status",
                "data": asdict(status)
            }
            await websocket.send_json(initial_message)
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
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
        if websocket in active_websockets:
            active_websockets.remove(websocket)


def main():
    print("R-Zero Training Demo System")
    print("=" * 50)
    print("Starting demo backend server...")
    print("Backend API will be available at: http://127.0.0.1:8000")
    print("API Documentation: http://127.0.0.1:8000/docs")
    print()
    print("This is a simulation that demonstrates the training UI.")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    uvicorn.run(
        "demo_training_ui:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()