# Post-Training Alignment Against Prompt Injection

This repository implements post-training alignment techniques to improve small language models' robustness against jailbreaking and prompt injection attacks without requiring full re-pretraining.

**Now using [Tinker API](https://tinker-docs.thinkingmachines.ai/) for distributed DPO training!**

## Project Overview

**Course**: DSC-291 - Safety for GenAI  
**Focus**: Using Direct Preference Optimization (DPO) to align small language models for safer responses  
**Training Platform**: Tinker - A training API for distributed LLM fine-tuning

### Key Components

1. **Dataset Collection**: Adversarial datasets for jailbreaking and prompt injection
2. **Model Selection**: Small Language Models (1B-70B parameters) via Tinker
3. **Training Pipeline**: DPO-based alignment using Tinker's distributed training
4. **Evaluation**: Comprehensive metrics for safety and performance
5. **Experimentation**: Tinker API integration for scaled experiments

## Project Structure

```
├── data/                   # Dataset storage and preprocessing
│   ├── raw/               # Raw datasets
│   ├── processed/         # Processed preference pairs
│   └── README.md          # Dataset documentation
├── src/                   # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── training/          # DPO training with Tinker
│   ├── evaluation/        # Evaluation metrics and scripts
│   └── utils/             # Tinker API wrapper
├── configs/               # Configuration files
│   ├── datasets/          # Dataset configs
│   └── models/            # Model configs
├── notebooks/             # Jupyter notebooks for analysis
├── experiments/           # Experiment scripts
├── results/              # Training outputs and results
├── README.md             # This file
└── TINKER_GUIDE.md       # Complete Tinker API usage guide
```

## Datasets

### Primary Datasets
- **AdvBench**: Adversarial prompts for safety evaluation
- **JailbreakBench**: Comprehensive jailbreak attempts
- **ToxicChat**: Toxic and harmful conversations
- **HarmBench**: Multi-category harmful behaviors
- **WildJailbreak**: Real-world jailbreak attempts

### Dataset Preparation
See `data/README.md` for detailed instructions on downloading and processing datasets.

## Models

### Candidate Small Language Models
1. **Llama-3.2-1B** (Meta)
2. **Phi-3-mini** (Microsoft)
3. **Gemma-2B** (Google)
4. **Qwen-2-1.5B** (Alibaba)
5. **StableLM-2-1.6B** (Stability AI)

## Setup

### Prerequisites
- Python 3.9+
- Tinker API key (get from [Tinker Dashboard](https://tinker-docs.thinkingmachines.ai/))
- HuggingFace token (for model access)

### Installation

```bash
# Create a conda environment
conda create --name dsc291proj python=3.14
conda activate dsc291proj

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:
```
TINKER_API_KEY=your_tinker_api_key_here
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key_here  # Optional
```

For detailed setup and usage, see **`TINKER_GUIDE.md`**.

## Usage

### 1. Data Preparation
```bash
python src/data/prepare_datasets.py --dataset advbench --output data/processed/
```

### 2. Training with DPO (Tinker)
```bash
python src/training/train_dpo_tinker.py \
    --model meta-llama/Llama-3.2-1B \
    --train-file data/processed/advbench/train.jsonl \
    --epochs 1 \
    --batch-size 2 \
    --learning-rate 1e-5 \
    --dpo-beta 0.1
```

### 3. Evaluation
```bash
python src/evaluation/evaluate_tinker.py \
    --model-path tinker://llama-3.2-1b-dpo-final \
    --test-file data/processed/advbench/test.jsonl
```

### 4. Run Experiments
See `TINKER_GUIDE.md` for detailed Tinker API usage and examples.

## Evaluation Metrics

1. **Attack Success Rate (ASR)**: Percentage of successful jailbreak attempts
2. **Refusal Rate**: Percentage of harmful prompts refused
3. **Perplexity**: Model's language modeling capability
4. **BLEU/ROUGE**: Response quality on benign prompts
5. **Safety Score**: Custom composite safety metric

## Experiment Tracking

We use Weights & Biases for experiment tracking. View runs at:
```
https://wandb.ai/your-username/dsc291-safety-alignment
```

## Results

Results will be stored in `results/` directory with the following structure:
```
results/
├── {model_name}_{timestamp}/
│   ├── checkpoints/
│   ├── logs/
│   ├── metrics.json
│   └── config.yaml
```

## Notebooks

- `notebooks/01_dataset_exploration.ipynb`: Dataset analysis
- `notebooks/02_baseline_evaluation.ipynb`: Pre-training baselines

## Documentation

- **`TINKER_GUIDE.md`**: Complete guide for using Tinker API with DPO training
- **`data/README.md`**: Dataset information and usage

## Contributing

This is a course project for DSC-291 - Safety for GenAI at UCSD.

## References

1. Rafailov et al. (2023) - Direct Preference Optimization
2. Zou et al. (2023) - Universal and Transferable Adversarial Attacks on Aligned Language Models
3. Mazeika et al. (2024) - HarmBench: A Standardized Evaluation Framework

## License

Academic use only. See LICENSE file for details.
