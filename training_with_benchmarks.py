#!/usr/bin/env python3
"""
Enhanced R-Zero Training with Golden Dataset Benchmarking
Integrates benchmark evaluation at each training iteration
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training_platform_independent import TrainingConfig, PlatformIndependentTrainer
from evaluation.golden_dataset_benchmarker import GoldenDatasetBenchmarker


class BenchmarkedTrainer(PlatformIndependentTrainer):
    """Enhanced trainer with integrated benchmarking"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.benchmarker = GoldenDatasetBenchmarker(str(self.storage_path))
        
        # Initialize golden datasets
        self.benchmarker.save_golden_datasets()
        
        # Track benchmarking results
        self.benchmark_results = {}
    
    def run_full_training_pipeline_with_benchmarks(self):
        """Run complete R-Zero training pipeline with benchmarking at each iteration"""
        self.log("=== Starting R-Zero Training with Golden Dataset Benchmarking ===")
        self.log(f"Base model: {self.config.base_model}")
        self.log(f"Iterations: {self.config.num_iterations}")
        self.log(f"Storage path: {self.config.storage_path}")
        
        # Benchmark base model first
        self.log("\n📊 Benchmarking Base Model...")
        try:
            base_benchmark = self.benchmarker.benchmark_model(
                model_path=self.config.base_model,
                model_name="base_model",
                iteration=0,
                role="base"
            )
            self.benchmark_results[0] = {"base": base_benchmark}
            
            # Print base model performance
            avg_accuracy = sum(r.accuracy for r in base_benchmark) / len(base_benchmark) if base_benchmark else 0
            self.log(f"Base model average accuracy: {avg_accuracy:.2%}")
            
        except Exception as e:
            self.log(f"⚠️  Base model benchmarking failed: {e}")
            self.benchmark_results[0] = {"base": []}
        
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
                
                # Benchmark the newly trained models
                self.log(f"\n📊 Benchmarking Iteration {iteration} Models...")
                
                iteration_benchmarks = {}
                
                # Benchmark questioner
                try:
                    questioner_benchmark = self.benchmarker.benchmark_model(
                        model_path=str(questioner_path),
                        model_name=f"questioner_v{iteration}",
                        iteration=iteration,
                        role="questioner"
                    )
                    iteration_benchmarks["questioner"] = questioner_benchmark
                    
                    q_avg_accuracy = sum(r.accuracy for r in questioner_benchmark) / len(questioner_benchmark)
                    self.log(f"Questioner v{iteration} average accuracy: {q_avg_accuracy:.2%}")
                    
                except Exception as e:
                    self.log(f"⚠️  Questioner v{iteration} benchmarking failed: {e}")
                    iteration_benchmarks["questioner"] = []
                
                # Benchmark solver
                try:
                    solver_benchmark = self.benchmarker.benchmark_model(
                        model_path=str(solver_path),
                        model_name=f"solver_v{iteration}",
                        iteration=iteration,
                        role="solver"
                    )
                    iteration_benchmarks["solver"] = solver_benchmark
                    
                    s_avg_accuracy = sum(r.accuracy for r in solver_benchmark) / len(solver_benchmark)
                    self.log(f"Solver v{iteration} average accuracy: {s_avg_accuracy:.2%}")
                    
                    # Show improvement from base model
                    if self.benchmark_results[0]["base"]:
                        base_avg = sum(r.accuracy for r in self.benchmark_results[0]["base"]) / len(self.benchmark_results[0]["base"])
                        improvement = s_avg_accuracy - base_avg
                        self.log(f"Solver v{iteration} improvement over base: {improvement:+.2%}")
                    
                except Exception as e:
                    self.log(f"⚠️  Solver v{iteration} benchmarking failed: {e}")
                    iteration_benchmarks["solver"] = []
                
                self.benchmark_results[iteration] = iteration_benchmarks
                
                # Update paths for next iteration
                current_questioner_path = str(questioner_path)
                current_solver_path = str(solver_path)
                
                # Record iteration results with benchmarks
                iteration_time = time.time() - iteration_start_time
                iteration_result = {
                    "iteration": iteration,
                    "questioner_path": str(questioner_path),
                    "solver_path": str(solver_path),
                    "training_time_seconds": iteration_time,
                    "timestamp": time.time(),
                    "benchmark_results": iteration_benchmarks
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
        
        # Generate final comprehensive benchmark report
        self.log("\n📈 Generating Final Performance Report...")
        self.benchmarker._generate_comprehensive_report_from_results(self.benchmark_results)
        
        # Save enhanced training summary with benchmarks
        self.save_enhanced_training_summary()
        self.log("=== R-Zero Training with Benchmarking Completed ===")
    
    def save_enhanced_training_summary(self):
        """Save training summary enhanced with benchmark results"""
        # Calculate performance improvements
        performance_improvements = {}
        
        if self.benchmark_results.get(0, {}).get("base"):
            base_accuracy = sum(r.accuracy for r in self.benchmark_results[0]["base"]) / len(self.benchmark_results[0]["base"])
            
            for iteration in range(1, self.config.num_iterations + 1):
                if iteration in self.benchmark_results:
                    iter_results = self.benchmark_results[iteration]
                    
                    for role in ["questioner", "solver"]:
                        if role in iter_results and iter_results[role]:
                            role_accuracy = sum(r.accuracy for r in iter_results[role]) / len(iter_results[role])
                            improvement = role_accuracy - base_accuracy
                            performance_improvements[f"{role}_v{iteration}"] = {
                                "accuracy": role_accuracy,
                                "improvement_over_base": improvement,
                                "improvement_percentage": improvement / base_accuracy if base_accuracy > 0 else 0
                            }
        
        # Enhanced summary with benchmarks
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
            } if self.training_history else {},
            "benchmark_results": self.benchmark_results,
            "performance_improvements": performance_improvements,
            "golden_datasets": list(self.benchmarker.golden_datasets.keys()),
            "benchmarking_summary": {
                "total_evaluations": sum(
                    len(roles.get("questioner", [])) + len(roles.get("solver", [])) + len(roles.get("base", []))
                    for roles in self.benchmark_results.values()
                ),
                "datasets_tested": len(self.benchmarker.golden_datasets),
                "best_performing_iteration": self._find_best_iteration()
            }
        }
        
        summary_file = self.storage_path / f"{self.config.experiment_name}_enhanced_training_summary.json"
        with open(summary_file, 'w') as f:
            import json
            json.dump(summary, f, indent=2, default=str)
            
        self.log(f"Enhanced training summary saved: {summary_file}")
        
        # Print performance summary
        self._print_performance_summary(performance_improvements)
        
        return summary
    
    def _find_best_iteration(self) -> dict:
        """Find the best performing iteration"""
        best_results = {"iteration": 0, "role": "base", "accuracy": 0.0}
        
        for iteration, roles in self.benchmark_results.items():
            for role, results in roles.items():
                if results:
                    avg_accuracy = sum(r.accuracy for r in results) / len(results)
                    if avg_accuracy > best_results["accuracy"]:
                        best_results = {
                            "iteration": iteration,
                            "role": role,
                            "accuracy": avg_accuracy
                        }
        
        return best_results
    
    def _print_performance_summary(self, improvements: dict):
        """Print a summary of performance improvements"""
        self.log("\n📊 PERFORMANCE IMPROVEMENT SUMMARY")
        self.log("=" * 50)
        
        if not improvements:
            self.log("No performance improvements calculated (missing base model benchmark)")
            return
        
        # Find best performers
        best_questioner = max(
            [(k, v) for k, v in improvements.items() if "questioner" in k],
            key=lambda x: x[1]["accuracy"],
            default=(None, None)
        )
        
        best_solver = max(
            [(k, v) for k, v in improvements.items() if "solver" in k],
            key=lambda x: x[1]["accuracy"],
            default=(None, None)
        )
        
        if best_questioner[0]:
            self.log(f"🏆 Best Questioner: {best_questioner[0]}")
            self.log(f"   Accuracy: {best_questioner[1]['accuracy']:.2%}")
            self.log(f"   Improvement: {best_questioner[1]['improvement_over_base']:+.2%}")
        
        if best_solver[0]:
            self.log(f"🏆 Best Solver: {best_solver[0]}")
            self.log(f"   Accuracy: {best_solver[1]['accuracy']:.2%}")
            self.log(f"   Improvement: {best_solver[1]['improvement_over_base']:+.2%}")
        
        # Show progression
        self.log("\n📈 Training Progression:")
        for iteration in range(1, self.config.num_iterations + 1):
            solver_key = f"solver_v{iteration}"
            if solver_key in improvements:
                acc = improvements[solver_key]["accuracy"]
                imp = improvements[solver_key]["improvement_over_base"]
                self.log(f"   Iteration {iteration}: {acc:.2%} ({imp:+.2%})")


def main():
    """Main training function with benchmarks"""
    import argparse
    
    parser = argparse.ArgumentParser(description="R-Zero Training with Golden Dataset Benchmarking")
    parser.add_argument("--base_model", default="ai/llama3.2", help="Base model path")
    parser.add_argument("--experiment_name", default="benchmarked_training", help="Experiment name")
    parser.add_argument("--storage_path", default="storage", help="Storage path")
    parser.add_argument("--iterations", type=int, default=3, help="Number of training iterations")
    parser.add_argument("--questions_per_iteration", type=int, default=50, help="Questions per iteration")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["STORAGE_PATH"] = str(Path(args.storage_path).absolute())
    os.environ["HUGGINGFACENAME"] = "test-user"
    
    config = TrainingConfig(
        base_model=args.base_model,
        experiment_name=args.experiment_name,
        storage_path=args.storage_path,
        huggingface_name="test-user",
        num_iterations=args.iterations,
        questions_per_iteration=args.questions_per_iteration,
        max_steps_questioner=3,  # Reduced for faster testing
        max_steps_solver=10,     # Reduced for faster testing
        learning_rate=5e-6,
        batch_size=2
    )
    
    # Create enhanced trainer
    trainer = BenchmarkedTrainer(config)
    
    # Run training with benchmarks
    trainer.run_full_training_pipeline_with_benchmarks()


if __name__ == "__main__":
    main()