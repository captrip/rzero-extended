#!/usr/bin/env python3
"""
Golden Dataset Benchmarker for R-Zero Training Evaluation
Evaluates model performance across training iterations using standard mathematical datasets
"""

import os
import sys
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_clients.config_manager import LLMConfigManager
from question_evaluate.evaluate_platform_independent import evaluate_questions


@dataclass
class BenchmarkResult:
    """Results from benchmarking a model on golden datasets"""
    model_name: str
    model_path: str
    iteration: Optional[int]
    role: str  # 'questioner', 'solver', or 'base'
    dataset_name: str
    total_questions: int
    correct_answers: int
    accuracy: float
    average_response_time: float
    timestamp: float
    detailed_results: Dict


class GoldenDatasetBenchmarker:
    """Benchmarks R-Zero models against golden mathematical datasets"""
    
    def __init__(self, storage_path: str = "storage"):
        self.storage_path = Path(storage_path)
        self.benchmarks_path = self.storage_path / "benchmarks"
        self.benchmarks_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize golden datasets
        self.golden_datasets = self._initialize_golden_datasets()
        
        # Track all benchmark results
        self.benchmark_history = []
        
    def _initialize_golden_datasets(self) -> Dict[str, List[Dict]]:
        """Initialize golden mathematical datasets for evaluation"""
        
        # Dataset 1: Basic Arithmetic (Elementary)
        arithmetic_basic = [
            {"question": "What is 15 + 27?", "answer": "42"},
            {"question": "Calculate 144 ÷ 12", "answer": "12"},
            {"question": "What is 8 × 9?", "answer": "72"},
            {"question": "What is 100 - 37?", "answer": "63"},
            {"question": "Calculate 7 × 6", "answer": "42"},
            {"question": "What is 81 ÷ 9?", "answer": "9"},
            {"question": "What is 25 + 35?", "answer": "60"},
            {"question": "Calculate 12 × 5", "answer": "60"},
            {"question": "What is 90 - 45?", "answer": "45"},
            {"question": "What is 64 ÷ 8?", "answer": "8"},
        ]
        
        # Dataset 2: Algebra (Intermediate)
        algebra_intermediate = [
            {"question": "If 3x + 7 = 22, what is x?", "answer": "5"},
            {"question": "Solve for y: 2y - 5 = 11", "answer": "8"},
            {"question": "If 4a + 3 = 19, what is a?", "answer": "4"},
            {"question": "What is x if 5x - 8 = 17?", "answer": "5"},
            {"question": "Solve: 3(x + 2) = 21", "answer": "5"},
            {"question": "If 2x + 3x = 25, what is x?", "answer": "5"},
            {"question": "Solve for n: 4n - 7 = 21", "answer": "7"},
            {"question": "What is y if 6y + 4 = 28?", "answer": "4"},
            {"question": "If 3x - 9 = 12, what is x?", "answer": "7"},
            {"question": "Solve: 2(x - 3) = 14", "answer": "10"},
        ]
        
        # Dataset 3: Geometry & Advanced (Advanced)
        geometry_advanced = [
            {"question": "What is the square root of 64?", "answer": "8"},
            {"question": "What is the area of a circle with radius 3? (use π ≈ 3.14)", "answer": "28.26"},
            {"question": "What is the square root of 144?", "answer": "12"},
            {"question": "What is 2³ + 3²?", "answer": "17"},
            {"question": "What is the perimeter of a rectangle with length 8 and width 5?", "answer": "26"},
            {"question": "What is the square root of 100?", "answer": "10"},
            {"question": "What is 4² × 3?", "answer": "48"},
            {"question": "What is the area of a triangle with base 6 and height 4?", "answer": "12"},
            {"question": "What is the square root of 36?", "answer": "6"},
            {"question": "What is the volume of a cube with side length 3?", "answer": "27"},
        ]
        
        # Dataset 4: Word Problems (Complex reasoning)
        word_problems = [
            {"question": "Sarah has 15 apples. She gives away 7 and buys 12 more. How many apples does she have?", "answer": "20"},
            {"question": "A train travels 60 miles in 2 hours. What is its average speed in miles per hour?", "answer": "30"},
            {"question": "If 5 pencils cost $2.50, how much does 1 pencil cost?", "answer": "0.5"},
            {"question": "A rectangle has a perimeter of 20 and width of 3. What is its length?", "answer": "7"},
            {"question": "Tom has twice as many marbles as Jerry. If Jerry has 12 marbles, how many does Tom have?", "answer": "24"},
            {"question": "A pizza is cut into 8 equal slices. If you eat 3 slices, what fraction of the pizza is left?", "answer": "5/8"},
            {"question": "There are 24 students in a class. If 1/3 are boys, how many girls are there?", "answer": "16"},
            {"question": "A car uses 1 gallon of gas to travel 25 miles. How many gallons are needed for 100 miles?", "answer": "4"},
            {"question": "If you save $5 per week, how much will you have after 8 weeks?", "answer": "40"},
            {"question": "A store has 36 items. If they sell 2/3 of them, how many items are left?", "answer": "12"},
        ]
        
        return {
            "arithmetic_basic": arithmetic_basic,
            "algebra_intermediate": algebra_intermediate, 
            "geometry_advanced": geometry_advanced,
            "word_problems": word_problems
        }
    
    def save_golden_datasets(self):
        """Save golden datasets to files for evaluation"""
        datasets_path = self.storage_path / "golden_datasets"
        datasets_path.mkdir(exist_ok=True)
        
        for dataset_name, questions in self.golden_datasets.items():
            dataset_file = datasets_path / f"{dataset_name}.json"
            with open(dataset_file, 'w') as f:
                json.dump(questions, f, indent=2)
        
        print(f"Golden datasets saved to {datasets_path}")
        return datasets_path
    
    def benchmark_model(
        self, 
        model_path: str, 
        model_name: str, 
        iteration: Optional[int] = None,
        role: str = "unknown"
    ) -> List[BenchmarkResult]:
        """Benchmark a model against all golden datasets"""
        print(f"\n=== Benchmarking Model: {model_name} ===")
        print(f"Model Path: {model_path}")
        print(f"Iteration: {iteration}, Role: {role}")
        
        results = []
        
        for dataset_name, questions in self.golden_datasets.items():
            print(f"\nTesting on {dataset_name} ({len(questions)} questions)...")
            
            # Save dataset temporarily for evaluation
            temp_dataset_file = self.benchmarks_path / f"temp_{dataset_name}.json"
            with open(temp_dataset_file, 'w') as f:
                json.dump(questions, f, indent=2)
            
            # Evaluate model on this dataset
            output_file = self.benchmarks_path / f"{model_name}_{dataset_name}_results.json"
            
            start_time = time.time()
            try:
                eval_results = evaluate_questions(
                    model_path=model_path,
                    questions_file=str(temp_dataset_file),
                    output_file=str(output_file)
                )
                
                evaluation_time = time.time() - start_time
                
                # Extract results
                stats = eval_results.get('statistics', {})
                total_questions = stats.get('total_questions', len(questions))
                correct_answers = stats.get('total_correct', 0)
                accuracy = stats.get('overall_accuracy', 0.0)
                avg_response_time = evaluation_time / max(total_questions, 1)
                
                # Create benchmark result
                benchmark_result = BenchmarkResult(
                    model_name=model_name,
                    model_path=model_path,
                    iteration=iteration,
                    role=role,
                    dataset_name=dataset_name,
                    total_questions=total_questions,
                    correct_answers=correct_answers,
                    accuracy=accuracy,
                    average_response_time=avg_response_time,
                    timestamp=time.time(),
                    detailed_results=eval_results
                )
                
                results.append(benchmark_result)
                
                print(f"  ✓ {dataset_name}: {correct_answers}/{total_questions} ({accuracy:.2%})")
                
            except Exception as e:
                print(f"  ✗ {dataset_name}: Evaluation failed - {e}")
                
                # Create failed result
                benchmark_result = BenchmarkResult(
                    model_name=model_name,
                    model_path=model_path,
                    iteration=iteration,
                    role=role,
                    dataset_name=dataset_name,
                    total_questions=len(questions),
                    correct_answers=0,
                    accuracy=0.0,
                    average_response_time=0.0,
                    timestamp=time.time(),
                    detailed_results={"error": str(e)}
                )
                results.append(benchmark_result)
            
            # Clean up temp file
            if temp_dataset_file.exists():
                temp_dataset_file.unlink()
        
        # Add results to history
        self.benchmark_history.extend(results)
        
        # Save results
        self._save_benchmark_results(results)
        
        return results
    
    def benchmark_training_iteration(self, iteration: int, storage_path: str) -> Dict[str, List[BenchmarkResult]]:
        """Benchmark both questioner and solver models from a training iteration"""
        print(f"\n🔍 Benchmarking Training Iteration {iteration}")
        
        iteration_results = {
            "questioner": [],
            "solver": []
        }
        
        models_path = Path(storage_path) / "models"
        
        # Benchmark questioner
        questioner_path = models_path / f"llama32_questioner_v{iteration}" / "huggingface"
        if questioner_path.exists():
            iteration_results["questioner"] = self.benchmark_model(
                model_path=str(questioner_path),
                model_name=f"llama32_questioner_v{iteration}",
                iteration=iteration,
                role="questioner"
            )
        else:
            print(f"⚠️  Questioner model v{iteration} not found at {questioner_path}")
        
        # Benchmark solver
        solver_path = models_path / f"llama32_solver_v{iteration}" / "huggingface"
        if solver_path.exists():
            iteration_results["solver"] = self.benchmark_model(
                model_path=str(solver_path),
                model_name=f"llama32_solver_v{iteration}",
                iteration=iteration,
                role="solver"
            )
        else:
            print(f"⚠️  Solver model v{iteration} not found at {solver_path}")
        
        return iteration_results
    
    def benchmark_all_iterations(self, storage_path: str, max_iterations: int = 5) -> Dict[int, Dict]:
        """Benchmark all available training iterations"""
        print(f"\n🎯 Benchmarking All Training Iterations (1-{max_iterations})")
        
        all_results = {}
        
        # First benchmark the base model
        try:
            print("\n📊 Benchmarking Base Model (ai/llama3.2)...")
            base_results = self.benchmark_model(
                model_path="ai/llama3.2",
                model_name="base_llama32",
                iteration=0,
                role="base"
            )
            all_results[0] = {"base": base_results}
        except Exception as e:
            print(f"⚠️  Base model benchmarking failed: {e}")
            all_results[0] = {"base": []}
        
        # Benchmark each training iteration
        for iteration in range(1, max_iterations + 1):
            try:
                iteration_results = self.benchmark_training_iteration(iteration, storage_path)
                all_results[iteration] = iteration_results
            except Exception as e:
                print(f"⚠️  Iteration {iteration} benchmarking failed: {e}")
                all_results[iteration] = {"questioner": [], "solver": []}
        
        # Generate comprehensive report
        self._generate_comprehensive_report(all_results)
        
        return all_results
    
    def _save_benchmark_results(self, results: List[BenchmarkResult]):
        """Save benchmark results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.benchmarks_path / f"benchmark_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                "model_name": result.model_name,
                "model_path": result.model_path,
                "iteration": result.iteration,
                "role": result.role,
                "dataset_name": result.dataset_name,
                "total_questions": result.total_questions,
                "correct_answers": result.correct_answers,
                "accuracy": result.accuracy,
                "average_response_time": result.average_response_time,
                "timestamp": result.timestamp,
                "detailed_results": result.detailed_results
            })
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"📁 Benchmark results saved to {results_file}")
    
    def _generate_comprehensive_report(self, all_results: Dict[int, Dict]):
        """Generate a comprehensive performance report with visualizations"""
        print(f"\n📈 Generating Comprehensive Performance Report...")
        
        # Prepare data for analysis
        report_data = []
        
        for iteration, roles in all_results.items():
            for role, results_list in roles.items():
                for result in results_list:
                    if isinstance(result, BenchmarkResult):
                        report_data.append({
                            'iteration': iteration,
                            'role': role,
                            'model_name': result.model_name,
                            'dataset': result.dataset_name,
                            'accuracy': result.accuracy,
                            'correct_answers': result.correct_answers,
                            'total_questions': result.total_questions,
                            'response_time': result.average_response_time
                        })
        
        if not report_data:
            print("⚠️  No benchmark data available for report generation")
            return
        
        # Create DataFrame for analysis
        df = pd.DataFrame(report_data)
        
        # Generate performance plots
        self._create_performance_plots(df)
        
        # Generate detailed report
        self._create_detailed_report(df)
        
        # Generate summary statistics
        self._create_summary_statistics(df)
    
    def _create_performance_plots(self, df: pd.DataFrame):
        """Create performance visualization plots"""
        try:
            # Set up the plotting style
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('R-Zero Training Performance Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Accuracy by Iteration (All Datasets)
            ax1 = axes[0, 0]
            for dataset in df['dataset'].unique():
                dataset_data = df[df['dataset'] == dataset]
                iterations = dataset_data['iteration'].values
                accuracies = dataset_data['accuracy'].values
                ax1.plot(iterations, accuracies, marker='o', label=dataset, linewidth=2)
            
            ax1.set_title('Accuracy Improvement by Iteration')
            ax1.set_xlabel('Training Iteration')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Plot 2: Accuracy by Role
            ax2 = axes[0, 1]
            role_accuracy = df.groupby(['iteration', 'role'])['accuracy'].mean().reset_index()
            for role in role_accuracy['role'].unique():
                role_data = role_accuracy[role_accuracy['role'] == role]
                ax2.plot(role_data['iteration'], role_data['accuracy'], 
                        marker='s', label=f'{role} model', linewidth=2)
            
            ax2.set_title('Average Accuracy by Model Role')
            ax2.set_xlabel('Training Iteration')
            ax2.set_ylabel('Average Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            
            # Plot 3: Dataset Difficulty Analysis
            ax3 = axes[1, 0]
            dataset_avg = df.groupby('dataset')['accuracy'].mean().sort_values(ascending=True)
            bars = ax3.barh(range(len(dataset_avg)), dataset_avg.values, color='skyblue')
            ax3.set_yticks(range(len(dataset_avg)))
            ax3.set_yticklabels(dataset_avg.index)
            ax3.set_title('Dataset Difficulty Ranking (Lower = Harder)')
            ax3.set_xlabel('Average Accuracy')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.2%}', ha='left', va='center')
            
            # Plot 4: Performance Improvement Summary
            ax4 = axes[1, 1]
            if len(df['iteration'].unique()) > 1:
                # Calculate improvement from base to final
                base_perf = df[df['iteration'] == 0]['accuracy'].mean() if 0 in df['iteration'].unique() else 0
                final_perf = df[df['iteration'] == df['iteration'].max()]['accuracy'].mean()
                improvement = final_perf - base_perf
                
                categories = ['Base Model', 'Final Model']
                values = [base_perf, final_perf]
                colors = ['lightcoral', 'lightgreen']
                
                bars = ax4.bar(categories, values, color=colors)
                ax4.set_title(f'Overall Performance Improvement\n(+{improvement:.2%})')
                ax4.set_ylabel('Average Accuracy')
                ax4.set_ylim(0, 1)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.2%}', ha='center', va='bottom', fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'Insufficient data\nfor improvement analysis', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Performance Improvement')
            
            plt.tight_layout()
            
            # Save the plot
            plot_file = self.benchmarks_path / "performance_analysis.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Performance plots saved to {plot_file}")
            
        except Exception as e:
            print(f"⚠️  Plot generation failed: {e}")
    
    def _create_detailed_report(self, df: pd.DataFrame):
        """Create detailed text report"""
        report_file = self.benchmarks_path / "detailed_performance_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# R-Zero Training Performance Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            
            if len(df) > 0:
                total_iterations = df['iteration'].nunique()
                total_datasets = df['dataset'].nunique()
                avg_accuracy = df['accuracy'].mean()
                
                f.write(f"- **Training Iterations Evaluated:** {total_iterations}\n")
                f.write(f"- **Datasets Tested:** {total_datasets}\n")
                f.write(f"- **Overall Average Accuracy:** {avg_accuracy:.2%}\n")
                
                if 0 in df['iteration'].unique() and df['iteration'].max() > 0:
                    base_acc = df[df['iteration'] == 0]['accuracy'].mean()
                    final_acc = df[df['iteration'] == df['iteration'].max()]['accuracy'].mean()
                    improvement = final_acc - base_acc
                    f.write(f"- **Performance Improvement:** {improvement:+.2%}\n")
                
                f.write("\n")
            
            # Detailed Results by Iteration
            f.write("## Detailed Results by Iteration\n\n")
            
            for iteration in sorted(df['iteration'].unique()):
                iter_data = df[df['iteration'] == iteration]
                f.write(f"### Iteration {iteration}\n\n")
                
                for role in iter_data['role'].unique():
                    role_data = iter_data[iter_data['role'] == role]
                    f.write(f"**{role.title()} Model:**\n\n")
                    
                    for _, row in role_data.iterrows():
                        f.write(f"- {row['dataset']}: {row['correct_answers']}/{row['total_questions']} ({row['accuracy']:.2%})\n")
                    
                    avg_acc = role_data['accuracy'].mean()
                    f.write(f"- **Average Accuracy:** {avg_acc:.2%}\n\n")
            
            # Dataset Analysis
            f.write("## Dataset Difficulty Analysis\n\n")
            dataset_stats = df.groupby('dataset').agg({
                'accuracy': ['mean', 'std', 'count'],
                'correct_answers': 'sum',
                'total_questions': 'sum'
            }).round(4)
            
            f.write("| Dataset | Avg Accuracy | Std Dev | Tests | Total Correct | Total Questions |\n")
            f.write("|---------|-------------|---------|-------|---------------|----------------|\n")
            
            for dataset in dataset_stats.index:
                stats = dataset_stats.loc[dataset]
                f.write(f"| {dataset} | {stats[('accuracy', 'mean')]:.2%} | ±{stats[('accuracy', 'std')]:.3f} | {stats[('accuracy', 'count')]} | {stats[('correct_answers', 'sum')]} | {stats[('total_questions', 'sum')]} |\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if len(df[df['iteration'] > 0]) > 0:
                best_iteration = df[df['iteration'] > 0].groupby('iteration')['accuracy'].mean().idxmax()
                best_accuracy = df[df['iteration'] == best_iteration]['accuracy'].mean()
                
                f.write(f"1. **Best Performing Iteration:** Iteration {best_iteration} with {best_accuracy:.2%} average accuracy\n")
                
                # Find most challenging dataset
                worst_dataset = df.groupby('dataset')['accuracy'].mean().idxmin()
                worst_accuracy = df[df['dataset'] == worst_dataset]['accuracy'].mean()
                f.write(f"2. **Most Challenging Dataset:** {worst_dataset} ({worst_accuracy:.2%} accuracy) - Consider additional training on similar problems\n")
                
                # Performance trend
                if len(df['iteration'].unique()) > 2:
                    iteration_trend = df.groupby('iteration')['accuracy'].mean()
                    if iteration_trend.iloc[-1] > iteration_trend.iloc[0]:
                        f.write(f"3. **Positive Training Trend:** Models show consistent improvement across iterations\n")
                    else:
                        f.write(f"3. **Training Plateau:** Consider adjusting hyperparameters or training methodology\n")
        
        print(f"📋 Detailed report saved to {report_file}")
    
    def _create_summary_statistics(self, df: pd.DataFrame):
        """Create summary statistics file"""
        stats_file = self.benchmarks_path / "performance_statistics.json"
        
        stats = {
            "overview": {
                "total_evaluations": len(df),
                "unique_iterations": df['iteration'].nunique(),
                "unique_datasets": df['dataset'].nunique(),
                "unique_roles": df['role'].nunique(),
                "evaluation_timestamp": datetime.now().isoformat()
            },
            "performance_metrics": {
                "overall_average_accuracy": float(df['accuracy'].mean()),
                "overall_std_accuracy": float(df['accuracy'].std()),
                "best_single_performance": float(df['accuracy'].max()),
                "worst_single_performance": float(df['accuracy'].min())
            },
            "by_iteration": {},
            "by_dataset": {},
            "by_role": {}
        }
        
        # Statistics by iteration
        for iteration in sorted(df['iteration'].unique()):
            iter_data = df[df['iteration'] == iteration]
            stats["by_iteration"][str(iteration)] = {
                "average_accuracy": float(iter_data['accuracy'].mean()),
                "std_accuracy": float(iter_data['accuracy'].std()),
                "total_correct": int(iter_data['correct_answers'].sum()),
                "total_questions": int(iter_data['total_questions'].sum()),
                "datasets_tested": iter_data['dataset'].nunique()
            }
        
        # Statistics by dataset
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            stats["by_dataset"][dataset] = {
                "average_accuracy": float(dataset_data['accuracy'].mean()),
                "std_accuracy": float(dataset_data['accuracy'].std()),
                "total_evaluations": len(dataset_data),
                "difficulty_rank": int(df.groupby('dataset')['accuracy'].mean().rank(ascending=True)[dataset])
            }
        
        # Statistics by role
        for role in df['role'].unique():
            role_data = df[df['role'] == role]
            stats["by_role"][role] = {
                "average_accuracy": float(role_data['accuracy'].mean()),
                "std_accuracy": float(role_data['accuracy'].std()),
                "total_evaluations": len(role_data)
            }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📊 Performance statistics saved to {stats_file}")


def main():
    """Main benchmarking function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Golden Dataset Benchmarker for R-Zero Training")
    parser.add_argument("--storage_path", default="storage", help="Storage path for models")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations to benchmark")
    parser.add_argument("--model_path", help="Benchmark a specific model")
    parser.add_argument("--model_name", help="Name for specific model")
    
    args = parser.parse_args()
    
    # Initialize benchmarker
    benchmarker = GoldenDatasetBenchmarker(args.storage_path)
    
    # Save golden datasets
    benchmarker.save_golden_datasets()
    
    if args.model_path and args.model_name:
        # Benchmark specific model
        print(f"Benchmarking specific model: {args.model_name}")
        benchmarker.benchmark_model(args.model_path, args.model_name)
    else:
        # Benchmark all iterations
        print("Benchmarking all training iterations...")
        benchmarker.benchmark_all_iterations(args.storage_path, args.iterations)
    
    print("\n🎉 Benchmarking completed! Check the storage/benchmarks/ directory for results.")


if __name__ == "__main__":
    main()