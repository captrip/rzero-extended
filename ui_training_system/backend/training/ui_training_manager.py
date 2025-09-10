#!/usr/bin/env python3
"""
UI Training Manager
Manages R-Zero training with real-time updates and LangSmith integration
"""

import os
import sys
import json
import time
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from queue import Queue, Empty
import uuid

import torch
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from backend.langsmith_integration.langsmith_client import (
    LangSmithTrainingTracker, 
    TrainingMetrics, 
    ExperimentConfig
)
from llm_clients.config_manager import LLMConfigManager
from question_generate.question_generate_platform_independent import generate_questions_simple
from question_evaluate.evaluate_platform_independent import evaluate_questions


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
class UITrainingConfig:
    """Extended training configuration with UI-specific settings"""
    base_model: str
    experiment_name: str
    storage_path: str
    huggingface_name: str
    num_iterations: int = 5
    questions_per_iteration: int = 1000
    max_steps_questioner: int = 6
    max_steps_solver: int = 20
    learning_rate: float = 1e-6
    batch_size: int = 4
    save_steps: int = 2
    enable_langsmith: bool = True
    langsmith_project: str = "r-zero-training"
    real_time_updates: bool = True
    auto_save_interval: int = 60  # seconds


class UITrainingManager:
    """Training manager with real-time UI updates and LangSmith integration"""
    
    def __init__(self, config: UITrainingConfig, status_callback: Optional[Callable] = None):
        self.config = config
        self.status_callback = status_callback
        self.status_queue = Queue()
        
        # Initialize paths
        self.storage_path = Path(config.storage_path)
        self.models_path = self.storage_path / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize LangSmith tracker
        self.langsmith_tracker = None
        if config.enable_langsmith:
            try:
                self.langsmith_tracker = LangSmithTrainingTracker(
                    project_name=config.langsmith_project
                )
            except Exception as e:
                self.log(f"Warning: Could not initialize LangSmith tracker: {e}")
        
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
        
        # LLM client for inference
        self.llm_config = LLMConfigManager()
        
        # Training history
        self.training_history = []
        
        # Background update thread
        self.update_thread = None
        self.update_thread_running = False
        
    def log(self, message: str):
        """Log message with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        # Add to status queue for UI updates
        self.status_queue.put({"type": "log", "message": log_message})
        
    def update_status(self, **kwargs):
        """Update training status and notify UI"""
        for key, value in kwargs.items():
            if hasattr(self.current_status, key):
                setattr(self.current_status, key, value)
        
        # Add to status queue
        self.status_queue.put({"type": "status", "status": asdict(self.current_status)})
        
        # Call status callback if provided
        if self.status_callback:
            try:
                self.status_callback(self.current_status)
            except Exception as e:
                print(f"Status callback error: {e}")
    
    def start_background_updates(self):
        """Start background thread for status updates"""
        if not self.update_thread_running:
            self.update_thread_running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
    
    def stop_background_updates(self):
        """Stop background thread for status updates"""
        self.update_thread_running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
    
    def _update_loop(self):
        """Background update loop"""
        while self.update_thread_running:
            try:
                # Process status updates
                while True:
                    try:
                        update = self.status_queue.get_nowait()
                        # Here you would send updates to frontend via WebSocket
                        # For now, we just process the queue
                    except Empty:
                        break
                
                time.sleep(0.1)  # 100ms update interval
                
            except Exception as e:
                print(f"Update loop error: {e}")
    
    def load_base_model(self, model_path: str):
        """Load base model and tokenizer with progress updates"""
        self.log(f"Loading base model: {model_path}")
        self.update_status(current_phase="loading_model", progress_percentage=5.0)
        
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.update_status(progress_percentage=25.0)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.update_status(progress_percentage=50.0)
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        
        self.update_status(progress_percentage=100.0)
        self.log("Model loaded successfully")
        
        return model, tokenizer, config
    
    def create_training_dataset(self, questions_data: List[Dict], role: str = "questioner"):
        """Create dataset for training from questions/answers"""
        self.log(f"Creating training dataset for {role} with {len(questions_data)} samples")
        
        if role == "questioner":
            texts = [item.get("question", "") for item in questions_data]
        else:
            texts = []
            for item in questions_data:
                question = item.get("question", "")
                answer = item.get("answer", "")
                text = f"Question: {question}\nAnswer: {answer}"
                texts.append(text)
        
        return Dataset.from_dict({"text": texts})
    
    def train_model_with_progress(
        self, 
        model, 
        tokenizer, 
        dataset: Dataset, 
        output_dir: Path,
        max_steps: int,
        role: str
    ):
        """Train model with real-time progress updates"""
        self.log(f"Starting {role} training with {len(dataset)} samples")
        self.update_status(
            current_phase=f"training_{role}",
            total_steps=max_steps,
            current_step=0
        )
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=2048,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Custom training arguments with callbacks
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=1,
            max_steps=max_steps,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            logging_steps=1,
            save_steps=self.config.save_steps,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            bf16=True,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Custom trainer with progress callbacks
        class ProgressTrainer(Trainer):
            def __init__(self, ui_manager, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.ui_manager = ui_manager
                
            def on_step_end(self, args, state, control, **kwargs):
                # Update progress
                logs = kwargs.get('logs', {})
                loss = logs.get('train_loss', 0.0)
                lr = logs.get('learning_rate', 0.0)
                
                progress = (state.global_step / args.max_steps) * 100
                
                self.ui_manager.update_status(
                    current_step=state.global_step,
                    loss=loss,
                    learning_rate=lr,
                    progress_percentage=progress
                )
                
                # Log to LangSmith
                if self.ui_manager.langsmith_tracker:
                    metrics = TrainingMetrics(
                        iteration=self.ui_manager.current_status.current_iteration,
                        role=role,
                        loss=loss,
                        learning_rate=lr,
                        step=state.global_step,
                        timestamp=time.time()
                    )
                    self.ui_manager.langsmith_tracker.log_metrics(metrics)
                
                return super().on_step_end(args, state, control, **kwargs)
        
        trainer = ProgressTrainer(
            ui_manager=self,
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Save final checkpoint
        final_checkpoint = output_dir / f"global_step_{max_steps}"
        trainer.save_model(str(final_checkpoint / "actor"))
        
        self.update_status(current_step=max_steps, progress_percentage=100.0)
        return final_checkpoint / "actor"
    
    def train_questioner(self, iteration: int, base_model_path: str) -> Path:
        """Train questioner model with UI updates"""
        save_name = f"llama32_questioner_v{iteration}"
        output_dir = self.models_path / save_name
        
        self.log(f"=== Training Questioner V{iteration} ===")
        self.update_status(
            current_iteration=iteration,
            current_phase="questioner",
            progress_percentage=0.0
        )
        
        # Log iteration start to LangSmith
        if self.langsmith_tracker:
            self.langsmith_tracker.log_iteration_start(iteration, "questioner")
        
        try:
            # Load model
            model, tokenizer, config = self.load_base_model(base_model_path)
            
            # Create synthetic dataset (in real implementation, use reward-based data)
            synthetic_questions = [
                {"question": f"What is {i} + {i+1}?"} for i in range(1, 101)
            ]
            
            dataset = self.create_training_dataset(synthetic_questions, "questioner")
            
            # Train model with progress updates
            checkpoint_path = self.train_model_with_progress(
                model, tokenizer, dataset, output_dir,
                self.config.max_steps_questioner, "questioner"
            )
            
            # Save with metadata
            metadata = {
                "iteration": iteration,
                "role": "questioner",
                "base_model": base_model_path,
                "training_steps": self.config.max_steps_questioner,
                "timestamp": time.time()
            }
            
            final_path = self.save_model_with_metadata(model, tokenizer, checkpoint_path, metadata)
            
            # Log checkpoint to LangSmith
            if self.langsmith_tracker:
                self.langsmith_tracker.log_model_checkpoint(
                    iteration, "questioner", str(final_path), metadata
                )
            
            # Clean up GPU memory
            del model
            torch.cuda.empty_cache()
            
            self.log(f"Questioner V{iteration} training completed")
            return final_path
            
        except Exception as e:
            self.log(f"Error training questioner V{iteration}: {e}")
            self.update_status(status="error", error_message=str(e))
            if self.langsmith_tracker:
                self.langsmith_tracker.log_error(str(e), iteration, {"role": "questioner"})
            raise
    
    def train_solver(self, iteration: int, solver_model_path: str, questioner_model_path: str) -> Path:
        """Train solver model with UI updates"""
        save_name = f"llama32_solver_v{iteration}"
        output_dir = self.models_path / save_name
        
        self.log(f"=== Training Solver V{iteration} ===")
        self.update_status(
            current_iteration=iteration,
            current_phase="solver",
            progress_percentage=0.0
        )
        
        # Log iteration start to LangSmith
        if self.langsmith_tracker:
            self.langsmith_tracker.log_iteration_start(iteration, "solver")
        
        try:
            # Step 1: Generate questions
            self.update_status(current_phase="question_generation")
            self.log("Generating questions...")
            
            questions_file = self.storage_path / f"generated_question/training_{save_name}.json"
            questions_file.parent.mkdir(parents=True, exist_ok=True)
            
            start_time = time.time()
            try:
                generated_questions = generate_questions_simple(
                    model_path=str(questioner_model_path),
                    num_questions=self.config.questions_per_iteration,
                    output_file=str(questions_file)
                )
                generation_time = time.time() - start_time
                
                self.log(f"Generated {len(generated_questions)} questions in {generation_time:.1f}s")
                
                # Log to LangSmith
                if self.langsmith_tracker:
                    self.langsmith_tracker.log_question_generation(
                        iteration, generated_questions, generation_time
                    )
                    
            except Exception as e:
                self.log(f"Question generation failed: {e}, using fallback questions")
                generated_questions = [
                    {"question": f"Calculate {i} * {j}", "answer": str(i*j)}
                    for i in range(2, 12) for j in range(2, 12)
                ]
            
            # Step 2: Evaluate questions
            self.update_status(current_phase="evaluation")
            self.log("Evaluating questions...")
            
            try:
                evaluation_results = evaluate_questions(
                    model_path=str(solver_model_path),
                    questions_file=str(questions_file),
                    output_file=str(questions_file.with_suffix('_results.json'))
                )
                self.log(f"Evaluation completed: {evaluation_results.get('statistics', {})}")
                
                # Log to LangSmith
                if self.langsmith_tracker:
                    self.langsmith_tracker.log_evaluation_results(iteration, evaluation_results)
                    
            except Exception as e:
                self.log(f"Evaluation failed: {e}")
                evaluation_results = {}
            
            # Step 3: Train solver
            self.update_status(current_phase="training_solver")
            self.log("Training solver...")
            
            model, tokenizer, config = self.load_base_model(solver_model_path)
            dataset = self.create_training_dataset(generated_questions, "solver")
            
            checkpoint_path = self.train_model_with_progress(
                model, tokenizer, dataset, output_dir,
                self.config.max_steps_solver, "solver"
            )
            
            # Save with metadata
            metadata = {
                "iteration": iteration,
                "role": "solver",
                "base_model": solver_model_path,
                "questioner_model": questioner_model_path,
                "training_steps": self.config.max_steps_solver,
                "questions_generated": len(generated_questions),
                "evaluation_results": evaluation_results,
                "timestamp": time.time()
            }
            
            final_path = self.save_model_with_metadata(model, tokenizer, checkpoint_path, metadata)
            
            # Log checkpoint to LangSmith
            if self.langsmith_tracker:
                self.langsmith_tracker.log_model_checkpoint(
                    iteration, "solver", str(final_path), metadata
                )
            
            # Clean up GPU memory
            del model
            torch.cuda.empty_cache()
            
            self.log(f"Solver V{iteration} training completed")
            return final_path
            
        except Exception as e:
            self.log(f"Error training solver V{iteration}: {e}")
            self.update_status(status="error", error_message=str(e))
            if self.langsmith_tracker:
                self.langsmith_tracker.log_error(str(e), iteration, {"role": "solver"})
            raise
    
    def save_model_with_metadata(self, model, tokenizer, save_path: Path, metadata: Dict):
        """Save model with training metadata"""
        save_path.mkdir(parents=True, exist_ok=True)
        hf_path = save_path / "huggingface"
        
        # Save model and tokenizer
        model.save_pretrained(hf_path)
        tokenizer.save_pretrained(hf_path)
        
        # Save metadata
        metadata_path = save_path / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.log(f"Model saved to {hf_path}")
        return hf_path
    
    async def run_full_training_pipeline(self):
        """Run the complete R-Zero training pipeline asynchronously"""
        self.log("=== Starting R-Zero UI Training Pipeline ===")
        self.update_status(status="running", progress_percentage=0.0)
        
        # Start LangSmith experiment
        if self.langsmith_tracker:
            langsmith_config = ExperimentConfig(
                experiment_name=self.config.experiment_name,
                base_model=self.config.base_model,
                num_iterations=self.config.num_iterations,
                questions_per_iteration=self.config.questions_per_iteration,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                max_steps_questioner=self.config.max_steps_questioner,
                max_steps_solver=self.config.max_steps_solver
            )
            self.langsmith_tracker.start_experiment(langsmith_config)
        
        # Start background updates
        self.start_background_updates()
        
        try:
            # Initialize with base model
            current_questioner_path = self.config.base_model
            current_solver_path = self.config.base_model
            
            self.is_training = True
            
            for iteration in range(1, self.config.num_iterations + 1):
                if self.should_stop:
                    self.log("Training stopped by user")
                    break
                
                if self.should_pause:
                    self.log("Training paused by user")
                    self.update_status(status="paused")
                    while self.should_pause and not self.should_stop:
                        await asyncio.sleep(1)
                    if self.should_stop:
                        break
                    self.log("Training resumed")
                    self.update_status(status="running")
                
                iteration_start_time = time.time()
                
                self.log(f"\n{'='*60}")
                self.log(f"ITERATION {iteration}/{self.config.num_iterations}")
                self.log(f"{'='*60}")
                
                try:
                    # Train questioner
                    questioner_path = self.train_questioner(iteration, current_questioner_path)
                    
                    # Train solver
                    solver_path = self.train_solver(iteration, current_solver_path, str(questioner_path))
                    
                    # Update paths for next iteration
                    current_questioner_path = str(questioner_path)
                    current_solver_path = str(solver_path)
                    
                    # Record iteration results
                    iteration_time = time.time() - iteration_start_time
                    iteration_result = {
                        "iteration": iteration,
                        "questioner_path": str(questioner_path),
                        "solver_path": str(solver_path),
                        "training_time_seconds": iteration_time,
                        "timestamp": time.time()
                    }
                    
                    self.training_history.append(iteration_result)
                    
                    # Update overall progress
                    overall_progress = (iteration / self.config.num_iterations) * 100
                    self.update_status(
                        progress_percentage=overall_progress,
                        model_paths={
                            "questioner": str(questioner_path),
                            "solver": str(solver_path)
                        }
                    )
                    
                    self.log(f"Iteration {iteration} completed in {iteration_time:.1f}s")
                    
                except Exception as e:
                    self.log(f"ERROR in iteration {iteration}: {e}")
                    self.update_status(status="error", error_message=str(e))
                    if self.langsmith_tracker:
                        self.langsmith_tracker.log_error(str(e), iteration)
                    continue
            
            # Training completed
            self.update_status(status="completed", progress_percentage=100.0)
            self.log("=== R-Zero Training Pipeline Completed ===")
            
            # Save training summary
            final_summary = self.save_training_summary()
            
            # Finish LangSmith experiment
            if self.langsmith_tracker:
                self.langsmith_tracker.finish_experiment(final_summary)
                experiment_url = self.langsmith_tracker.get_experiment_url()
                if experiment_url:
                    self.log(f"View experiment at: {experiment_url}")
            
        except Exception as e:
            self.log(f"Training pipeline error: {e}")
            self.update_status(status="error", error_message=str(e))
            if self.langsmith_tracker:
                self.langsmith_tracker.log_error(str(e))
        
        finally:
            self.is_training = False
            self.stop_background_updates()
    
    def save_training_summary(self):
        """Save training summary and results"""
        summary = {
            "config": asdict(self.config),
            "training_history": self.training_history,
            "total_training_time": sum(r.get("training_time_seconds", 0) for r in self.training_history),
            "completed_iterations": len(self.training_history),
            "final_models": {
                "questioner": self.training_history[-1]["questioner_path"] if self.training_history else None,
                "solver": self.training_history[-1]["solver_path"] if self.training_history else None
            } if self.training_history else {},
            "experiment_id": self.current_status.experiment_id
        }
        
        summary_file = self.storage_path / f"{self.config.experiment_name}_training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log(f"Training summary saved: {summary_file}")
        return summary
    
    # Control methods for UI
    def pause_training(self):
        """Pause training"""
        self.should_pause = True
        self.log("Pause requested")
    
    def resume_training(self):
        """Resume training"""
        self.should_pause = False
        self.log("Resume requested")
    
    def stop_training(self):
        """Stop training"""
        self.should_stop = True
        self.log("Stop requested")
    
    def get_status(self) -> TrainingStatus:
        """Get current training status"""
        return self.current_status
    
    def get_training_history(self) -> List[Dict]:
        """Get training history"""
        return self.training_history