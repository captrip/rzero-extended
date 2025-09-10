# Enhanced Evaluation System for R-Zero

This enhanced evaluation system addresses the limitations of the original mock evaluation by providing:

1. **External Challenger Models**: Support for powerful models like GPT-4, Claude, etc.
2. **Custom Dataset Handling**: Load and evaluate on your own datasets
3. **Improved Question Generation**: Generate challenging, domain-specific questions
4. **Better Evaluation Metrics**: Difficulty scoring, reasoning quality assessment

## Quick Start

### 1. Setup a Challenger Model

```bash
# Interactive setup tool
python scripts/challenger_setup.py
```

This will help you configure external models like:
- **OpenAI GPT-4/GPT-3.5**: For high-quality question generation
- **Anthropic Claude**: For diverse reasoning challenges  
- **Ollama Models**: For local deployment
- **Custom APIs**: Any OpenAI-compatible endpoint

### 2. Test the Enhanced System

```bash
# Run comprehensive test suite
python test_enhanced_evaluation.py
```

### 3. Use Enhanced Evaluation

```python
from evaluation.enhanced_evaluator import EnhancedEvaluator

# Initialize evaluator
evaluator = EnhancedEvaluator()

# Run comprehensive evaluation
results = evaluator.run_comprehensive_evaluation(
    custom_dataset_path="path/to/your/dataset.json",  # Optional
    num_generated_questions=20,
    num_samples_per_question=5,
    domain="mathematics", 
    difficulty="hard"
)

# Save results
evaluator.save_results(results, "my_evaluation_results.json")
```

## Key Features

### 🤖 External Challenger Models

Instead of using mock questions, the system can now use powerful external models to generate challenging questions:

```yaml
# config/llm_config.yaml
generation_models:
  challenger:
    provider: "openai"
    config:
      model_name: "gpt-4o"
      api_key: "${OPENAI_API_KEY}"
```

### 📊 Custom Dataset Support

Load datasets in multiple formats:

```python
# JSON format
[
  {"question": "What is 2+2?", "answer": "4"},
  {"question": "Solve x^2 = 16", "answer": "x = ±4"}
]

# CSV format (question,answer columns)
# JSONL format (one JSON object per line)
```

### 📈 Enhanced Metrics

The system now provides:
- **Overall Accuracy**: Traditional correctness measure
- **Difficulty Score**: How challenging questions are for the solver
- **Reasoning Quality**: Assessment of step-by-step reasoning

### 🎯 Domain-Specific Generation

Generate questions for specific domains:

```python
# Mathematics
questions = evaluator.generate_questions_with_challenger(
    domain="mathematics",
    difficulty="hard",
    num_questions=10
)

# Other domains: "physics", "chemistry", "programming", etc.
```

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  
export STORAGE_PATH="./storage"
```

### Model Configuration

The `config/llm_config.yaml` file controls which models are used:

- **Challenger**: Generates questions (can be external like GPT-4)
- **Solver**: Model being trained/evaluated (usually local)
- **Evaluator**: Checks answer correctness (can be external)

## Example Usage

### Basic Evaluation

```bash
# Generate 10 questions and evaluate with 3 samples each
python evaluation/enhanced_evaluator.py \
    --generated 10 \
    --samples 3 \
    --domain mathematics \
    --difficulty medium
```

### With Custom Dataset

```bash
# Use your own dataset + generate additional questions
python evaluation/enhanced_evaluator.py \
    --dataset ./my_questions.json \
    --generated 5 \
    --samples 3 \
    --output my_evaluation_results.json
```

### Advanced Configuration

```python
from evaluation.enhanced_evaluator import EnhancedEvaluator

evaluator = EnhancedEvaluator()

# Set specific challenger
evaluator.challenger_manager.set_challenger("primary")  # or "claude", "gpt4", etc.

# Run evaluation
results = evaluator.run_comprehensive_evaluation(
    custom_dataset_path="datasets/gsm8k_subset.json",
    num_generated_questions=50,
    num_samples_per_question=10,
    domain="mathematical_reasoning",
    difficulty="challenging"
)
```

## Comparison with Original System

| Feature | Original | Enhanced |
|---------|----------|-----------|
| Question Source | Mock/hardcoded | External models (GPT-4, Claude) |
| Dataset Support | Limited | JSON, JSONL, CSV |
| Question Quality | Basic | Challenging, domain-specific |
| Evaluation Depth | Accuracy only | Accuracy + Difficulty + Reasoning |
| Model Flexibility | Fixed local models | Any OpenAI-compatible API |
| Custom Domains | No | Yes (math, physics, programming, etc.) |

## Benefits for Training

1. **Better Learning Signal**: High-quality questions from advanced models help the solver learn faster
2. **Diverse Challenges**: Different challenger models provide varied question styles
3. **Real-world Alignment**: Custom datasets let you evaluate on your specific use case
4. **Progressive Difficulty**: Questions can be made more challenging as the solver improves

## Troubleshooting

### Challenger Connection Issues
```bash
# Test your configuration
python scripts/challenger_setup.py
# Choose option 6 to test generation
```

### Dataset Loading Problems
```python
# Verify dataset format
from evaluation.enhanced_evaluator import EnhancedEvaluator
evaluator = EnhancedEvaluator()
dataset = evaluator.load_custom_dataset("your_file.json")
print(f"Loaded {len(dataset)} questions")
```

### Solver Connection Issues
- Ensure your local model server is running
- Check the `solver` configuration in `llm_config.yaml`
- Test with: `python question_evaluate/evaluate_platform_independent.py`

## Next Steps

1. **Setup your challenger model** using the interactive tool
2. **Prepare your custom dataset** (optional but recommended)  
3. **Run enhanced evaluation** to see the improved quality
4. **Integrate with training** to use better questions for learning

The enhanced system maintains compatibility with the existing R-Zero framework while providing significantly better evaluation capabilities.