#!/usr/bin/env python3
"""
Platform-Independent R-Zero Training System
Replaces vLLM-dependent training pipeline with OpenAI-compatible API approach
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
import numpy as np

from llm_clients.config_manager import LLMConfigManager
from question_generate.question_generate_platform_independent import generate_questions_simple
from question_evaluate.evaluate_platform_independent import evaluate_questions


@dataclass
class TrainingConfig:
    """Configuration for R-Zero training pipeline"""
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


class PlatformIndependentTrainer:
    """Platform-independent R-Zero trainer that doesn't depend on vLLM"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.storage_path = Path(config.storage_path)
        self.models_path = self.storage_path / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM client for inference
        self.llm_config = LLMConfigManager()
        
        # Track training history
        self.training_history = []
        
    def log(self, message: str):
        """Log training progress"""
        print(f"[R-Zero Training] {message}")
        
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
        
    def load_base_model(self, model_path: str):
        """Load base model and tokenizer"""
        self.log(f"Loading base model: {model_path}")
        
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
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
        
        return model, tokenizer, config
        
    def create_training_dataset(self, questions_data: List[Dict], role: str = "questioner"):
        """Create dataset for training from questions/answers"""
        if role == "questioner":
            # For questioner training, use questions as targets
            texts = [item.get("question", "") for item in questions_data]
        else:
            # For solver training, use question-answer pairs
            texts = []
            for item in questions_data:
                question = item.get("question", "")
                answer = item.get("answer", "")
                text = f"Question: {question}\nAnswer: {answer}"
                texts.append(text)
                
        return Dataset.from_dict({"text": texts})
        
    def train_model(
        self, 
        model, 
        tokenizer, 
        dataset: Dataset, 
        output_dir: Path,
        max_steps: int,
        role: str
    ):
        """Train model using HuggingFace Trainer"""
        self.log(f"Starting {role} training with {len(dataset)} samples")
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=2048,
                return_tensors="pt"
            )
            
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
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
        
        trainer = Trainer(
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
        
        return final_checkpoint / "actor"
        
    def train_questioner(self, iteration: int, base_model_path: str) -> Path:
        """Train questioner model for given iteration"""
        save_name = f"llama32_questioner_v{iteration}"
        output_dir = self.models_path / save_name
        
        self.log(f"=== Training Questioner V{iteration} ===")
        
        # Load model
        model, tokenizer, config = self.load_base_model(base_model_path)
        
        # Create synthetic dataset for questioner training
        # In real implementation, this would use reward-based training data
        synthetic_questions = [
            {"question": f"What is {i} + {i+1}?"} for i in range(1, 101)
        ]
        
        dataset = self.create_training_dataset(synthetic_questions, "questioner")
        
        # Train model
        checkpoint_path = self.train_model(
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
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()
        
        return final_path
        
    def train_solver(self, iteration: int, solver_model_path: str, questioner_model_path: str) -> Path:
        """Train solver model for given iteration"""
        save_name = f"llama32_solver_v{iteration}"
        output_dir = self.models_path / save_name
        
        self.log(f"=== Training Solver V{iteration} ===")
        
        # Step 1: Generate questions using questioner
        self.log("Generating questions...")
        questions_file = self.storage_path / f"generated_question/training_{save_name}.json"
        questions_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use questioner to generate questions
            generated_questions = generate_questions_simple(
                model_path=str(questioner_model_path),
                num_questions=self.config.questions_per_iteration,
                output_file=str(questions_file)
            )
            self.log(f"Generated {len(generated_questions)} questions")
        except Exception as e:
            self.log(f"Question generation failed: {e}, using fallback questions")
            # Fallback to simple questions
            generated_questions = [
                {"question": f"Calculate {i} * {j}", "answer": str(i*j)} 
                for i in range(2, 12) for j in range(2, 12)
            ]
            
        # Step 2: Evaluate questions using current solver
        self.log("Evaluating questions...")
        try:
            evaluation_results = evaluate_questions(
                model_path=str(solver_model_path),
                questions_file=str(questions_file),
                output_file=str(questions_file.with_suffix('_results.json'))
            )
            self.log(f"Evaluation completed: {evaluation_results.get('statistics', {})}")
        except Exception as e:
            self.log(f"Evaluation failed: {e}, proceeding with generated questions")
            
        # Step 3: Train solver on the generated questions
        self.log("Training solver...")
        model, tokenizer, config = self.load_base_model(solver_model_path)
        
        dataset = self.create_training_dataset(generated_questions, "solver")
        
        checkpoint_path = self.train_model(
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
            "timestamp": time.time()
        }
        
        final_path = self.save_model_with_metadata(model, tokenizer, checkpoint_path, metadata)
        
        # Clean up GPU memory
        del model  
        torch.cuda.empty_cache()
        
        return final_path
        
    def run_full_training_pipeline(self):
        """Run the complete R-Zero training pipeline"""
        self.log("=== Starting R-Zero Platform-Independent Training ===")
        self.log(f"Base model: {self.config.base_model}")
        self.log(f"Iterations: {self.config.num_iterations}")
        self.log(f"Storage path: {self.config.storage_path}")
        
        # Initialize with base model
        current_questioner_path = self.config.base_model
        current_solver_path = self.config.base_model
        
        for iteration in range(1, self.config.num_iterations + 1):
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
                
                self.log(f"Iteration {iteration} completed in {iteration_time:.1f}s")
                self.log(f"Questioner saved: {questioner_path}")
                self.log(f"Solver saved: {solver_path}")
                
            except Exception as e:
                self.log(f"ERROR in iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        # Save training summary
        self.save_training_summary()
        self.log("=== R-Zero Training Pipeline Completed ===")
        
    def save_training_summary(self):
        """Save training summary and results"""
        summary = {
            "config": {
                "base_model": self.config.base_model,
                "experiment_name": self.config.experiment_name,
                "num_iterations": self.config.num_iterations,
                "questions_per_iteration": self.config.questions_per_iteration
            },
            "training_history": self.training_history,
            "total_training_time": sum(r.get("training_time_seconds", 0) for r in self.training_history),
            "completed_iterations": len(self.training_history),
            "final_models": {
                "questioner": self.training_history[-1]["questioner_path"] if self.training_history else None,
                "solver": self.training_history[-1]["solver_path"] if self.training_history else None
            } if self.training_history else {}
        }
        
        summary_file = self.storage_path / f"{self.config.experiment_name}_training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.log(f"Training summary saved: {summary_file}")
        return summary


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Platform-Independent R-Zero Training")
    parser.add_argument("--base_model", required=True, help="Base model path")
    parser.add_argument("--experiment_name", required=True, help="Experiment name") 
    parser.add_argument("--storage_path", default="./storage", help="Storage path")
    parser.add_argument("--huggingface_name", default="test-user", help="HuggingFace username")
    parser.add_argument("--num_iterations", type=int, default=5, help="Number of training iterations")
    parser.add_argument("--questions_per_iteration", type=int, default=100, help="Questions per iteration")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        base_model=args.base_model,
        experiment_name=args.experiment_name,
        storage_path=args.storage_path,
        huggingface_name=args.huggingface_name,
        num_iterations=args.num_iterations,
        questions_per_iteration=args.questions_per_iteration
    )
    
    trainer = PlatformIndependentTrainer(config)
    trainer.run_full_training_pipeline()


if __name__ == "__main__":
    main()