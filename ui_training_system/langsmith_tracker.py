#!/usr/bin/env python3
"""
LangSmith Integration for R-Zero Training Tracking
Replaces WandB with LangSmith for experiment tracking and visualization
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

try:
    from langsmith import Client
    from langchain.callbacks.tracers.langchain import LangChainTracer
    LANGSMITH_AVAILABLE = True
except ImportError:
    print("LangSmith not available. Install with: pip install langsmith langchain")
    LANGSMITH_AVAILABLE = False


class LangSmithTracker:
    """LangSmith integration for R-Zero training tracking"""
    
    def __init__(self, project_name: str = "r-zero-training", experiment_name: str = None):
        self.project_name = project_name
        self.experiment_name = experiment_name or f"rzero-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Initialize LangSmith client
        self.client = None
        self.session_id = None
        
        if LANGSMITH_AVAILABLE:
            try:
                self.client = Client()
                self.session_id = self.client.create_session(
                    session_name=self.experiment_name,
                    metadata={
                        "project": self.project_name,
                        "framework": "r-zero",
                        "start_time": datetime.now().isoformat()
                    }
                )
                print(f"LangSmith tracking initialized: {self.experiment_name}")
            except Exception as e:
                print(f"LangSmith initialization failed: {e}")
                print("Falling back to local logging")
                self.client = None
        
        # Local logging fallback
        self.local_logs = []
        self.metrics_history = []
        
    def log_experiment_start(self, config: Dict[str, Any]):
        """Log experiment configuration and start"""
        log_data = {
            "event": "experiment_start",
            "timestamp": time.time(),
            "config": config,
            "experiment_name": self.experiment_name
        }
        
        if self.client:
            try:
                self.client.create_run(
                    name="experiment_config",
                    run_type="chain",
                    session_name=self.experiment_name,
                    inputs={"config": config},
                    outputs={"status": "started"}
                )
            except Exception as e:
                print(f"LangSmith logging failed: {e}")
        
        self.local_logs.append(log_data)
        print(f"Experiment started: {self.experiment_name}")
        
    def log_iteration_start(self, iteration: int, phase: str, model_info: Dict[str, Any]):
        """Log start of training iteration"""
        log_data = {
            "event": "iteration_start",
            "timestamp": time.time(),
            "iteration": iteration,
            "phase": phase,  # "questioner" or "solver"
            "model_info": model_info
        }
        
        if self.client:
            try:
                self.client.create_run(
                    name=f"iteration_{iteration}_{phase}",
                    run_type="chain",
                    session_name=self.experiment_name,
                    inputs={
                        "iteration": iteration,
                        "phase": phase,
                        "model_info": model_info
                    }
                )
            except Exception as e:
                print(f"LangSmith logging failed: {e}")
        
        self.local_logs.append(log_data)
        print(f"Iteration {iteration} {phase} started")
    
    def log_training_metrics(self, iteration: int, phase: str, metrics: Dict[str, Any]):
        """Log training metrics"""
        timestamp = time.time()
        
        metric_data = {
            "event": "training_metrics",
            "timestamp": timestamp,
            "iteration": iteration,
            "phase": phase,
            "metrics": metrics
        }
        
        if self.client:
            try:
                self.client.create_run(
                    name=f"metrics_{iteration}_{phase}",
                    run_type="llm",
                    session_name=self.experiment_name,
                    inputs={"iteration": iteration, "phase": phase},
                    outputs={"metrics": metrics}
                )
            except Exception as e:
                print(f"LangSmith metrics logging failed: {e}")
        
        self.local_logs.append(metric_data)
        self.metrics_history.append(metric_data)
        
        # Print key metrics
        print(f"Iteration {iteration} {phase} metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")
    
    def log_benchmark_results(self, iteration: int, phase: str, benchmark_results: Dict[str, Any]):
        """Log golden dataset benchmark results"""
        log_data = {
            "event": "benchmark_results",
            "timestamp": time.time(),
            "iteration": iteration,
            "phase": phase,
            "benchmark_results": benchmark_results
        }
        
        if self.client:
            try:
                self.client.create_run(
                    name=f"benchmark_{iteration}_{phase}",
                    run_type="tool",
                    session_name=self.experiment_name,
                    inputs={
                        "iteration": iteration,
                        "phase": phase,
                        "datasets": list(benchmark_results.keys())
                    },
                    outputs={"results": benchmark_results}
                )
            except Exception as e:
                print(f"LangSmith benchmark logging failed: {e}")
        
        self.local_logs.append(log_data)
        
        # Print benchmark summary
        print(f"Iteration {iteration} {phase} benchmark results:")
        for dataset, result in benchmark_results.items():
            if isinstance(result, dict) and 'accuracy' in result:
                accuracy = result['accuracy']
                correct = result.get('correct_answers', 'N/A')
                total = result.get('total_questions', 'N/A')
                print(f"  {dataset}: {accuracy:.2%} ({correct}/{total})")
    
    def log_model_saved(self, iteration: int, phase: str, model_path: str, metadata: Dict[str, Any]):
        """Log model saving event"""
        log_data = {
            "event": "model_saved",
            "timestamp": time.time(),
            "iteration": iteration,
            "phase": phase,
            "model_path": model_path,
            "metadata": metadata
        }
        
        if self.client:
            try:
                self.client.create_run(
                    name=f"model_save_{iteration}_{phase}",
                    run_type="chain",
                    session_name=self.experiment_name,
                    inputs={
                        "iteration": iteration,
                        "phase": phase,
                        "model_path": model_path
                    },
                    outputs={"metadata": metadata}
                )
            except Exception as e:
                print(f"LangSmith model logging failed: {e}")
        
        self.local_logs.append(log_data)
        print(f"Model saved: {model_path}")
    
    def log_error(self, iteration: int, phase: str, error: str, context: Dict[str, Any]):
        """Log training error"""
        log_data = {
            "event": "error",
            "timestamp": time.time(),
            "iteration": iteration,
            "phase": phase,
            "error": error,
            "context": context
        }
        
        if self.client:
            try:
                self.client.create_run(
                    name=f"error_{iteration}_{phase}",
                    run_type="chain",
                    session_name=self.experiment_name,
                    inputs={"iteration": iteration, "phase": phase, "context": context},
                    outputs={"error": error}
                )
            except Exception as e:
                print(f"LangSmith error logging failed: {e}")
        
        self.local_logs.append(log_data)
        print(f"ERROR in iteration {iteration} {phase}: {error}")
    
    def log_experiment_complete(self, summary: Dict[str, Any]):
        """Log experiment completion"""
        log_data = {
            "event": "experiment_complete",
            "timestamp": time.time(),
            "summary": summary,
            "total_logs": len(self.local_logs)
        }
        
        if self.client:
            try:
                self.client.create_run(
                    name="experiment_summary",
                    run_type="chain",
                    session_name=self.experiment_name,
                    inputs={"experiment_name": self.experiment_name},
                    outputs={"summary": summary}
                )
            except Exception as e:
                print(f"LangSmith completion logging failed: {e}")
        
        self.local_logs.append(log_data)
        
        # Save local logs
        self.save_local_logs()
        
        print(f"Experiment completed: {self.experiment_name}")
        print(f"Total events logged: {len(self.local_logs)}")
        
        if self.client:
            print(f"View results in LangSmith: https://smith.langchain.com/")
        else:
            print(f"Local logs saved to: storage/langsmith_logs/")
    
    def save_local_logs(self):
        """Save logs locally as backup"""
        logs_dir = Path("storage/langsmith_logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all logs
        all_logs_file = logs_dir / f"{self.experiment_name}_all_logs.json"
        with open(all_logs_file, 'w') as f:
            json.dump(self.local_logs, f, indent=2, default=str)
        
        # Save metrics history
        metrics_file = logs_dir / f"{self.experiment_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        
        print(f"Local logs saved to: {logs_dir}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics"""
        if not self.metrics_history:
            return {}
        
        # Extract key metrics
        iterations = []
        accuracies = []
        losses = []
        
        for metric in self.metrics_history:
            if 'metrics' in metric:
                metrics = metric['metrics']
                iteration = metric['iteration']
                
                iterations.append(iteration)
                
                if 'accuracy' in metrics:
                    accuracies.append(metrics['accuracy'])
                if 'loss' in metrics:
                    losses.append(metrics['loss'])
        
        summary = {
            "total_iterations": len(set(iterations)),
            "total_metrics_logged": len(self.metrics_history)
        }
        
        if accuracies:
            summary["accuracy"] = {
                "min": min(accuracies),
                "max": max(accuracies),
                "final": accuracies[-1] if accuracies else None
            }
        
        if losses:
            summary["loss"] = {
                "min": min(losses),
                "max": max(losses), 
                "final": losses[-1] if losses else None
            }
        
        return summary


def setup_langsmith_environment():
    """Setup LangSmith environment variables if not set"""
    
    # Check if LangSmith API key is set
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("LANGSMITH SETUP REQUIRED:")
        print("1. Get your LangSmith API key from: https://smith.langchain.com/")
        print("2. Set environment variables:")
        print("   export LANGCHAIN_API_KEY='your-api-key'")
        print("   export LANGCHAIN_TRACING_V2=true")
        print("   export LANGCHAIN_PROJECT='r-zero-training'")
        print("")
        print("For Windows:")
        print("   set LANGCHAIN_API_KEY=your-api-key")
        print("   set LANGCHAIN_TRACING_V2=true") 
        print("   set LANGCHAIN_PROJECT=r-zero-training")
        return False
    
    return True


if __name__ == "__main__":
    # Test LangSmith tracker
    if not setup_langsmith_environment():
        print("Please set up LangSmith environment variables first")
    else:
        tracker = LangSmithTracker("test-project", "test-experiment")
        
        # Test logging
        tracker.log_experiment_start({"model": "test", "iterations": 1})
        tracker.log_iteration_start(1, "questioner", {"base_model": "test"})
        tracker.log_training_metrics(1, "questioner", {"loss": 0.5, "accuracy": 0.8})
        tracker.log_experiment_complete({"status": "test_complete"})
        
        print("LangSmith tracker test completed")