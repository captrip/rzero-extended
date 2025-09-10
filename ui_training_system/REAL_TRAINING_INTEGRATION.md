# Converting Simulation to Real R-Zero Training

## Current State: Demo vs Reality

### Demo System (What We Have)
```python
# In demo_training_ui.py
async def simulate_training_step(self, phase: str, step: int, total_steps: int):
    await asyncio.sleep(0.5)  # Just wait
    loss = max(0.1, 3.0 - (step / total_steps) * 2.5)  # Mock loss
    self.update_status(loss=loss)  # Update UI
```

### Real Training (What's Needed)
```python
# Real R-Zero with GRPO
async def real_training_step(self, model, batch, step: int):
    # Actual gradient computation
    loss = compute_grpo_loss(model, batch)
    loss.backward()
    optimizer.step()
    
    # Real metrics
    self.update_status(loss=loss.item())
```

## Integration Points

### 1. Replace Simulation with Real Training Manager

**File:** `ui_training_system/backend/training/ui_training_manager.py`

Replace the `DemoTrainingManager` with the real `UITrainingManager` that:

```python
class RealUITrainingManager(UITrainingManager):
    def __init__(self, config, status_callback):
        super().__init__(config, status_callback)
        
        # Real model loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    async def train_questioner(self, iteration: int, base_model_path: str):
        # Load real model
        model, tokenizer, config = self.load_base_model(base_model_path)
        
        # Real question generation training with GRPO
        for step in range(self.config.max_steps_questioner):
            batch = self.get_training_batch()  # Real data
            
            # GRPO loss computation
            loss = self.compute_grpo_loss(model, batch, "questioner")
            loss.backward()
            self.optimizer.step()
            
            # Real-time UI updates
            self.update_status(
                current_step=step,
                loss=loss.item(),
                learning_rate=self.optimizer.param_groups[0]['lr']
            )
            
            await asyncio.sleep(0.1)  # Allow UI updates
```

### 2. Real Question Generation

**Current (Mock):**
```python
questions = [{"question": f"What is {i} + {i+1}?"} for i in range(100)]
```

**Real Implementation:**
```python
# Use the existing question_generate_platform_independent.py
from question_generate.question_generate_platform_independent import generate_questions_simple

async def generate_questions_real(self, questioner_model_path: str, num_questions: int):
    self.update_status(current_phase="question_generation")
    
    # Real question generation using trained questioner
    questions = await asyncio.get_event_loop().run_in_executor(
        None,  # Use default executor
        lambda: generate_questions_simple(
            model_path=questioner_model_path,
            num_questions=num_questions,
            output_file=f"./storage/questions_iter_{self.current_iteration}.json"
        )
    )
    
    return questions
```

### 3. Real Model Training with GRPO

**Key Components Needed:**

```python
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer
)

class GRPOTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        
    def compute_grpo_loss(self, questions, answers, rewards):
        """Compute Generalized Reward Policy Optimization loss"""
        # Encode questions and answers
        inputs = self.tokenizer(questions, return_tensors="pt", padding=True)
        targets = self.tokenizer(answers, return_tensors="pt", padding=True)
        
        # Forward pass
        outputs = self.model(**inputs, labels=targets.input_ids)
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        
        # GRPO loss with rewards
        policy_loss = -torch.mean(log_probs * rewards.unsqueeze(-1))
        
        return policy_loss
```

### 4. Real Evaluation System

**Current (Mock):**
```python
evaluation_results = {"accuracy": 0.85, "score": 0.9}
```

**Real Implementation:**
```python
# Use existing evaluate_platform_independent.py
from question_evaluate.evaluate_platform_independent import evaluate_questions

async def evaluate_questions_real(self, solver_model_path: str, questions_file: str):
    self.update_status(current_phase="evaluation")
    
    # Real evaluation using trained solver
    evaluation_results = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: evaluate_questions(
            model_path=solver_model_path,
            questions_file=questions_file,
            output_file=questions_file.replace('.json', '_results.json')
        )
    )
    
    # Extract meaningful metrics
    accuracy = evaluation_results.get('statistics', {}).get('accuracy', 0.0)
    self.update_status(evaluation_score=accuracy)
    
    return evaluation_results
```

## File Replacements Needed

### 1. Replace Demo Backend
```bash
# Current demo file
ui_training_system/demo_training_ui.py

# Replace with real backend
ui_training_system/backend/api/main.py  # (Already created)
ui_training_system/backend/training/ui_training_manager.py  # (Already created)
```

### 2. Integration Script
```python
# ui_training_system/start_real_training.py
def main():
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Training will be slow.")
    
    # Start real backend with actual training
    backend_process = start_real_backend_server()
    frontend_process = start_frontend_server()
    
    print("Real R-Zero Training System Started!")
    print("- GPU Training: Enabled" if torch.cuda.is_available() else "- GPU Training: Disabled")
    print("- LangSmith Tracking: Enabled" if os.getenv("LANGSMITH_API_KEY") else "- LangSmith Tracking: Disabled")
```

## Required Dependencies for Real Training

```bash
# Additional packages needed for real training
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
pip install wandb  # or keep langsmith
pip install deepspeed  # for large model training
pip install flash-attn  # for memory efficiency
```

## Configuration Changes

### Real Training Config
```python
@dataclass
class RealTrainingConfig(UITrainingConfig):
    # GPU settings
    device: str = "auto"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # GRPO specific
    reward_model_path: str = None
    kl_coefficient: float = 0.1
    clip_range: float = 0.2
    
    # Performance
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    compile_model: bool = True  # PyTorch 2.0 compilation
```

## Performance Optimizations

### Memory Management
```python
# In real training manager
def optimize_memory(self):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        
    # Enable gradient checkpointing for large models
    if hasattr(self.model, 'gradient_checkpointing_enable'):
        self.model.gradient_checkpointing_enable()
```

### Distributed Training
```python
# For multiple GPUs
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed_training(self):
    if torch.cuda.device_count() > 1:
        self.model = DDP(self.model)
        print(f"Using {torch.cuda.device_count()} GPUs")
```

## Testing Real vs Demo

```python
# Test script to verify real training
def test_real_vs_demo():
    demo_config = DemoTrainingConfig(...)
    real_config = RealTrainingConfig(...)
    
    print("Demo Loss (should be deterministic):", run_demo_step())
    print("Real Loss (should vary with actual gradients):", run_real_step())
```

## Migration Path

1. **Phase 1**: Keep demo running, implement real backend in parallel
2. **Phase 2**: Add toggle in UI: "Demo Mode" vs "Real Training Mode"
3. **Phase 3**: Replace demo with real system once tested
4. **Phase 4**: Add advanced features (distributed training, custom rewards)

The UI framework is already production-ready - we just need to swap the simulation backend with real training logic!