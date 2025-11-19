# Using Tinker API for DPO Training

This project uses the **Tinker API** from Thinking Machines for distributed DPO training. Tinker handles the complexity of distributed training while you focus on your data and algorithms.

## What is Tinker?

Tinker is a training API that lets you fine-tune large language models without managing GPU infrastructure. You provide:
- Your training data
- Your loss function (DPO in our case)
- Hyperparameters

Tinker handles:
- Distributed training across multiple GPUs
- Hardware failure recovery
- Efficient LoRA fine-tuning

## Quick Start with Tinker

### 1. Install Tinker SDK

```bash
pip install tinker
```

### 2. Get API Key

1. Sign up at [Tinker Waitlist](https://thinkingmachines.ai/tinker)
2. Create API key from [Tinker Console](https://tinker-console.thinkingmachines.ai/)
3. Set environment variable:

```bash
export TINKER_API_KEY=your_api_key_here
```

Or add to `.env` file:
```
TINKER_API_KEY=your_api_key_here
```

### 3. Prepare Your Dataset

The dataset should be in JSONL format with `prompt`, `chosen`, and `rejected` fields:

```bash
python src/data/prepare_datasets.py --dataset advbench --output data/processed/
```

### 4. Train with DPO

```bash
python src/training/train_dpo_tinker.py \
    --model meta-llama/Llama-3.2-1B \
    --train-file data/processed/advbench/train.jsonl \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --dpo-beta 0.1
```

### 5. Evaluate

```bash
python src/evaluation/evaluate_tinker.py \
    --model-path meta-llama/Llama-3.2-1B \
    --test-file data/processed/advbench/test.jsonl
```

## Available Models in Tinker

Tinker supports models from the Qwen3 and Llama3 series:

- `meta-llama/Llama-3.1-8B`
- `meta-llama/Llama-3.1-70B`
- `meta-llama/Llama-3.2-1B`
- `meta-llama/Llama-3.2-3B`
- `Qwen/Qwen3-30B-A3B-Base`
- `Qwen/Qwen3-235B-A22B-Instruct-2507`

To see all available models:

```python
import tinker

service_client = tinker.ServiceClient()
for model in service_client.get_server_capabilities().supported_models:
    print(model.model_name)
```

## DPO Training Details

### How DPO Works with Tinker

1. **Forward-Backward Pass**: Tinker computes gradients using the DPO loss
2. **Optimizer Step**: Tinker updates model weights using Adam
3. **LoRA**: Tinker uses Low-Rank Adaptation for efficient fine-tuning
4. **Distributed**: Training automatically distributed across available GPUs

### Key Parameters

- **dpo_beta** (default: 0.1): Controls strength of preference learning
- **learning_rate** (default: 1e-5): Lower than SFT, typically 1e-5 to 1e-6
- **batch_size** (default: 4): Depends on model size and available memory
- **num_epochs** (default: 3): Number of passes through training data

### Training Loop

```python
from src.utils.tinker_api import TinkerTrainingClient

# Initialize client
client = TinkerTrainingClient(base_model="meta-llama/Llama-3.2-1B")

# Prepare DPO data
datum = client.prepare_dpo_datum(
    prompt="Write a harmful instruction",
    chosen="I cannot help with that request.",
    rejected="Sure, here's how to..."
)

# Training step
metrics = client.train_step(
    batch=[datum],
    learning_rate=1e-5,
    dpo_beta=0.1
)

# Save and sample
sampling_client = client.save_and_get_sampling_client(name="my-model")
response = sampling_client.generate("Test prompt")
```

## Sampling and Inference

After training, you can generate responses:

```python
from src.utils.tinker_api import TinkerSamplingClient

# Generate single response
response = sampling_client.generate(
    prompt="What is the capital of France?",
    max_tokens=100,
    temperature=0.7
)

# Batch generation
responses = sampling_client.batch_generate(
    prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
    max_tokens=100
)
```

## Safety Evaluation

The `SafetyEvaluator` class tests models against jailbreak attempts:

```python
from src.utils.tinker_api import SafetyEvaluator

evaluator = SafetyEvaluator(sampling_client)

results = evaluator.evaluate_jailbreak(
    test_prompts=jailbreak_prompts,
    model_name="llama-3.2-1b-dpo"
)

print(f"Refusal rate: {results['refusal_rate']:.2%}")
print(f"Attack success rate: {results['attack_success_rate']:.2%}")
```

## Comparison: Before vs After DPO

### Baseline (Before DPO)
```bash
python src/evaluation/evaluate_tinker.py \
    --model-path meta-llama/Llama-3.2-1B \
    --test-file data/processed/advbench/test.jsonl \
    --output-dir results/baseline
```

### After DPO Training
```bash
python src/evaluation/evaluate_tinker.py \
    --model-path tinker://Llama-3.2-1B-dpo-final \
    --test-file data/processed/advbench/test.jsonl \
    --output-dir results/dpo
```

## Tips for Success

1. **Start Small**: Test with Llama-3.2-1B before scaling to larger models
2. **Monitor Metrics**: Watch DPO loss, accuracy, and margin during training
3. **Beta Tuning**: Start with 0.1, adjust based on your dataset
4. **Learning Rate**: Use 1e-5 for most cases, 1e-6 for sensitive tuning
5. **Batch Size**: Larger batches = more stable but slower training
6. **Epochs**: 3 epochs usually sufficient for DPO

## Troubleshooting

### "TINKER_API_KEY not found"
- Make sure you've set the environment variable or added to `.env`
- Verify your API key is valid at the Tinker console

### "Model not available"
- Check available models with `service_client.get_server_capabilities()`
- Some models may require special access

### "Out of memory"
- Reduce `--batch-size`
- Tinker uses LoRA by default, which is memory-efficient
- Try a smaller model

## Resources

- [Tinker Documentation](https://tinker-docs.thinkingmachines.ai/)
- [DPO Guide](https://tinker-docs.thinkingmachines.ai/preferences/dpo-guide)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [Model Lineup](https://tinker-docs.thinkingmachines.ai/model-lineup)

## Example Workflow

Complete workflow from data to evaluation:

```bash
# 1. Prepare data
python src/data/prepare_datasets.py \
    --dataset advbench \
    --output data/processed/

# 2. Train with DPO
python src/training/train_dpo_tinker.py \
    --model meta-llama/Llama-3.2-1B \
    --train-file data/processed/advbench/train.jsonl \
    --epochs 3 \
    --dpo-beta 0.1

# 3. Evaluate
python src/evaluation/evaluate_tinker.py \
    --model-path tinker://Llama-3.2-1B-dpo-final \
    --test-file data/processed/advbench/test.jsonl

# 4. Compare results
python experiments/compare_results.py \
    --baseline results/baseline \
    --dpo results/dpo
```

Success! Your model should now show improved safety against jailbreak attempts.
