#!/usr/bin/env python3
"""
LangSmith Integration for R-Zero Training System
Replaces WandB with LangSmith for experiment tracking and monitoring
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import asyncio
import uuid

from langsmith import Client
from langsmith.schemas import Run, Example
from langsmith.utils import tracing_context


@dataclass
class TrainingMetrics:
    """Training metrics for tracking"""
    iteration: int
    role: str  # "questioner" or "solver"
    loss: float
    learning_rate: float
    step: int
    timestamp: float
    model_path: Optional[str] = None
    questions_generated: Optional[int] = None
    evaluation_score: Optional[float] = None
    training_time_seconds: Optional[float] = None


@dataclass
class ExperimentConfig:
    """Configuration for LangSmith experiment"""
    experiment_name: str
    base_model: str
    num_iterations: int
    questions_per_iteration: int
    learning_rate: float
    batch_size: int
    max_steps_questioner: int
    max_steps_solver: int


class LangSmithTrainingTracker:
    """LangSmith integration for tracking R-Zero training experiments"""
    
    def __init__(self, api_key: Optional[str] = None, project_name: str = "r-zero-training"):
        """Initialize LangSmith client"""
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY")
        if not self.api_key:
            raise ValueError("LangSmith API key must be provided or set in LANGSMITH_API_KEY environment variable")
            
        self.client = Client(api_key=self.api_key)
        self.project_name = project_name
        self.experiment_id = None
        self.current_run_id = None
        self.metrics_buffer = []
        
        # Create project if it doesn't exist
        self._ensure_project_exists()
        
    def _ensure_project_exists(self):
        """Ensure the project exists in LangSmith"""
        try:
            # Try to get project info
            projects = list(self.client.list_projects())
            project_names = [p.name for p in projects]
            
            if self.project_name not in project_names:
                # Create project
                project = self.client.create_project(
                    project_name=self.project_name,
                    description="R-Zero Training Experiment Tracking"
                )
                print(f"Created LangSmith project: {self.project_name}")
            else:
                print(f"Using existing LangSmith project: {self.project_name}")
                
        except Exception as e:
            print(f"Warning: Could not verify/create project: {e}")
    
    def start_experiment(self, config: ExperimentConfig) -> str:
        """Start a new training experiment"""
        self.experiment_id = str(uuid.uuid4())
        
        # Create experiment run
        experiment_metadata = {
            "experiment_type": "r-zero-training",
            "config": asdict(config),
            "start_time": time.time(),
            "experiment_id": self.experiment_id
        }
        
        try:
            # Create initial run for the entire experiment
            self.current_run_id = self.client.create_run(
                name=f"R-Zero Training: {config.experiment_name}",
                project_name=self.project_name,
                run_type="chain",
                inputs={"config": asdict(config)},
                extra=experiment_metadata
            ).id
            
            print(f"Started LangSmith experiment: {config.experiment_name}")
            print(f"Experiment ID: {self.experiment_id}")
            print(f"LangSmith Run ID: {self.current_run_id}")
            
        except Exception as e:
            print(f"Warning: Could not create LangSmith run: {e}")
            self.current_run_id = None
            
        return self.experiment_id
    
    def log_iteration_start(self, iteration: int, role: str) -> str:
        """Log the start of a training iteration"""
        iteration_run_id = str(uuid.uuid4())
        
        try:
            # Create child run for this iteration
            self.client.create_run(
                name=f"Iteration {iteration} - {role.title()}",
                project_name=self.project_name,
                parent_run_id=self.current_run_id,
                run_type="chain",
                inputs={
                    "iteration": iteration,
                    "role": role,
                    "start_time": time.time()
                },
                extra={
                    "iteration": iteration,
                    "role": role,
                    "phase": "training"
                }
            )
            
        except Exception as e:
            print(f"Warning: Could not log iteration start: {e}")
            
        return iteration_run_id
    
    def log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics"""
        self.metrics_buffer.append(metrics)
        
        try:
            # Log metrics to LangSmith
            if self.current_run_id:
                self.client.update_run(
                    run_id=self.current_run_id,
                    outputs={
                        "latest_metrics": asdict(metrics),
                        "total_logged_metrics": len(self.metrics_buffer)
                    },
                    extra={
                        f"metrics_{metrics.iteration}_{metrics.role}": asdict(metrics)
                    }
                )
                
        except Exception as e:
            print(f"Warning: Could not log metrics to LangSmith: {e}")
    
    def log_model_checkpoint(self, iteration: int, role: str, model_path: str, metadata: Dict[str, Any]):
        """Log model checkpoint information"""
        checkpoint_data = {
            "iteration": iteration,
            "role": role,
            "model_path": model_path,
            "metadata": metadata,
            "timestamp": time.time()
        }
        
        try:
            if self.current_run_id:
                self.client.update_run(
                    run_id=self.current_run_id,
                    extra={
                        f"checkpoint_{iteration}_{role}": checkpoint_data
                    }
                )
                
                print(f"Logged checkpoint for {role} iteration {iteration}")
                
        except Exception as e:
            print(f"Warning: Could not log checkpoint: {e}")
    
    def log_question_generation(self, iteration: int, questions: List[Dict[str, Any]], generation_time: float):
        """Log question generation results"""
        generation_data = {
            "iteration": iteration,
            "num_questions": len(questions),
            "generation_time_seconds": generation_time,
            "sample_questions": questions[:5],  # Log first 5 as samples
            "timestamp": time.time()
        }
        
        try:
            if self.current_run_id:
                self.client.update_run(
                    run_id=self.current_run_id,
                    extra={
                        f"question_generation_{iteration}": generation_data
                    }
                )
                
                print(f"Logged question generation for iteration {iteration}: {len(questions)} questions")
                
        except Exception as e:
            print(f"Warning: Could not log question generation: {e}")
    
    def log_evaluation_results(self, iteration: int, evaluation_results: Dict[str, Any]):
        """Log evaluation results"""
        eval_data = {
            "iteration": iteration,
            "results": evaluation_results,
            "timestamp": time.time()
        }
        
        try:
            if self.current_run_id:
                self.client.update_run(
                    run_id=self.current_run_id,
                    extra={
                        f"evaluation_{iteration}": eval_data
                    }
                )
                
                print(f"Logged evaluation results for iteration {iteration}")
                
        except Exception as e:
            print(f"Warning: Could not log evaluation results: {e}")
    
    def log_error(self, error_message: str, iteration: Optional[int] = None, context: Optional[Dict] = None):
        """Log error information"""
        error_data = {
            "error_message": error_message,
            "iteration": iteration,
            "context": context or {},
            "timestamp": time.time()
        }
        
        try:
            if self.current_run_id:
                self.client.update_run(
                    run_id=self.current_run_id,
                    extra={
                        f"error_{int(time.time())}": error_data
                    }
                )
                
        except Exception as e:
            print(f"Warning: Could not log error: {e}")
    
    def finish_experiment(self, final_summary: Dict[str, Any]):
        """Finish the experiment and log final summary"""
        try:
            if self.current_run_id:
                self.client.update_run(
                    run_id=self.current_run_id,
                    outputs={
                        "final_summary": final_summary,
                        "total_metrics_logged": len(self.metrics_buffer),
                        "end_time": time.time()
                    },
                    end_time=time.time()
                )
                
                print(f"Experiment finished. LangSmith run: {self.current_run_id}")
                print(f"View at: https://smith.langchain.com/projects/{self.project_name}")
                
        except Exception as e:
            print(f"Warning: Could not finish experiment: {e}")
    
    def get_experiment_url(self) -> Optional[str]:
        """Get the URL to view the experiment in LangSmith"""
        if self.current_run_id:
            return f"https://smith.langchain.com/runs/{self.current_run_id}?project={self.project_name}"
        return None
    
    def export_metrics(self, output_file: str):
        """Export all logged metrics to a file"""
        export_data = {
            "experiment_id": self.experiment_id,
            "project_name": self.project_name,
            "run_id": self.current_run_id,
            "metrics": [asdict(m) for m in self.metrics_buffer],
            "export_timestamp": time.time()
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"Metrics exported to: {output_file}")


# Example usage and testing
def test_langsmith_integration():
    """Test the LangSmith integration"""
    # This requires a valid LangSmith API key
    try:
        tracker = LangSmithTrainingTracker(project_name="r-zero-test")
        
        config = ExperimentConfig(
            experiment_name="test-experiment",
            base_model="meta-llama/Llama-3.2-1B",
            num_iterations=2,
            questions_per_iteration=100,
            learning_rate=1e-6,
            batch_size=4,
            max_steps_questioner=6,
            max_steps_solver=20
        )
        
        experiment_id = tracker.start_experiment(config)
        
        # Log some test metrics
        metrics = TrainingMetrics(
            iteration=1,
            role="questioner",
            loss=2.5,
            learning_rate=1e-6,
            step=100,
            timestamp=time.time(),
            training_time_seconds=120.0
        )
        
        tracker.log_metrics(metrics)
        tracker.log_question_generation(1, [{"question": "Test question?"}], 30.0)
        
        final_summary = {"status": "completed", "total_time": 240.0}
        tracker.finish_experiment(final_summary)
        
        print(f"Test completed. Experiment URL: {tracker.get_experiment_url()}")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_langsmith_integration()