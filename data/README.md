# Datasets

This directory contains safety alignment datasets for training and evaluation.

## Directory Structure

```
data/
├── raw/              # Raw downloaded datasets (optional)
└── processed/        # Processed DPO preference pairs
    └── advbench/     # Example: AdvBench dataset
        ├── train.jsonl
        ├── val.jsonl
        └── test.jsonl
```

## Available Datasets

### 1. AdvBench (Implemented)
- **Source**: [llm-attacks](https://github.com/llm-attacks/llm-attacks)
- **Size**: 520 adversarial prompts
- **Use**: Jailbreak evaluation benchmark
- **Status**: ✅ Fully integrated

### 2. JailbreakBench (Planned)
- **Source**: [JailbreakBench](https://jailbreakbench.github.io/)
- **Size**: 100 behaviors
- **Use**: Standardized jailbreak testing

### 3. HarmBench (Planned)
- **Source**: [HarmBench](https://arxiv.org/abs/2402.04249)
- **Size**: 510 harmful behaviors
- **Use**: Multi-category harm evaluation

### 4. BeaverTails (Planned)
- **Source**: [PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails)
- **Size**: 333k examples
- **Use**: Safety preference pairs for DPO training

### 5. ToxicChat (Planned)
- **Source**: [lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat)
- **Size**: 10k examples
- **Use**: Toxic content detection

## Downloading Datasets

### Using prepare_datasets.py
```bash
# Prepare AdvBench (currently supported)
python src/data/prepare_datasets.py \\\n    --dataset advbench \\\n    --output data/processed/

# This creates:
# - data/processed/advbench/train.jsonl (364 examples)
# - data/processed/advbench/val.jsonl (78 examples)
# - data/processed/advbench/test.jsonl (78 examples)
```

## Processing for DPO

Datasets are processed into preference pair format:
```json
{
    "prompt": "harmful prompt here",
    "chosen": "safe refusal response",
    "rejected": "unsafe compliance response"
}
```

### Create Preference Pairs
```bash
python src/data/prepare_datasets.py --dataset advbench --output data/processed/
```

This creates:
- `train.jsonl` - Training preference pairs
- `val.jsonl` - Validation preference pairs  
- `test.jsonl` - Test prompts for evaluation

## Dataset Statistics

AdvBench dataset after processing:
- **Total**: 520 adversarial prompts
- **Train**: 364 examples (70%)
- **Val**: 78 examples (15%)
- **Test**: 78 examples (15%)

Each example contains:
```json
{
    "prompt": "harmful or adversarial prompt",
    "chosen": "I cannot assist with that request.",
    "rejected": "Sure, here's how to..."
}
```

## Citation

When using these datasets, please cite the original papers:

**AdvBench**:
```bibtex
@misc{zou2023universal,
    title={Universal and Transferable Adversarial Attacks on Aligned Language Models},
    author={Andy Zou and Zifan Wang and J. Zico Kolter and Matt Fredrikson},
    year={2023},
    eprint={2307.15043},
    archivePrefix={arXiv}
}
```

## License

Each dataset has its own license. Check individual dataset pages before use.
