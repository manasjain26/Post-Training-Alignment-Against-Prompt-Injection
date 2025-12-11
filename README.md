# Post-Training Alignment Against Prompt Injection

This repository contains code for defending against prompt injection attacks using Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) on small language models (3-4B parameters). This work demonstrates that post-training alignment can significantly improve robustness without re-pretraining the model from scratch.

## Overview

We evaluate four training strategies on two small instruction-tuned models:
- **Llama-3.2-3B Base** (meta-llama/Llama-3.2-3B)
- **Qwen3-4B-Instruct-2507** (Qwen/Qwen3-4B-Instruct-2507)

Our best configuration (SFT+DPO) reduces Attack Success Rate (ASR) from 9.0% to 2.7% on Qwen-4B while improving benign helpfulness from 38.0% to 72.6%.

For in-depth details about the methodology, experimental setup, and results analysis, please refer to the full project report: `DSC291_Final_Project_Report.pdf`.

## Training Strategies

1. **Baseline**: Zero-shot behavior with no additional fine-tuning
2. **SFT-only**: LoRA-based supervised fine-tuning on safety refusals and benign responses
3. **DPO-only**: Direct preference optimization on preference pairs
4. **SFT+DPO**: Two-stage pipeline (SFT on 50% of data, then DPO on remaining 50%)

## Repository Structure

### Training Scripts
- `train_sft_tinker.py` - Single-stage SFT training
- `train_sequential_sft_tinker.py` - Sequential SFT training
- `train_dpo_tinker.py` - Single-stage DPO training
- `train_sequential_dpo_tinker.py` - Sequential DPO training (includes warm-start from SFT)

### Evaluation Scripts
- `evaluate.py` - Main evaluation script (ASR, helpfulness metrics)
- `evaluate_sft.py` - SFT/DPO-specific evaluation
- `evaluate_baseline.py` - Baseline model evaluation
- `openai_judge.py` - OpenAI-based judge implementation

### Data Processing
- `download_data.py` - Downloads WildJailbreak, AdvBench, JailbreakBench, and Alpaca datasets
- `prepare_training_data.py` - Formats data for training (creates preference pairs for DPO)

### Results
- `results/` - Evaluation results directory
- `results_openai/` - OpenAI judge results
- `figures/` - Plots and visualizations

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download and prepare datasets:
   ```bash
   python download_data.py
   python prepare_training_data.py
   ```

## Training Data

**Training Pool** (20,000 examples total):
- **WildJailbreak** (10k adversarial): Harmful prompts with safe refusals
- **Alpaca** (10k benign): Normal instructions with helpful responses

**Evaluation Benchmarks**:
- **AdvBench** (520 adversarial prompts)
- **JailbreakBench** (100 adversarial prompts)
- **Alpaca test split** (326 benign prompts)

## Training

### Supervised Fine-Tuning (SFT)

```bash
python train_sft_tinker.py --num_epochs 2
```

Hyperparameters:
- Batch size: 64
- Learning rate: 1e-4
- Epochs: 1 for Qwen-4B, 2 for Llama-3B

### Direct Preference Optimization (DPO)

```bash
python train_dpo_tinker.py --num_epochs 3
```

Hyperparameters:
- Batch size: 64
- Learning rate: 5e-6
- Beta: 0.1
- Epochs: 1 for Qwen-4B, 3 for Llama-3B

### Two-Stage SFT+DPO Pipeline

```bash
python train_sequential_sft_tinker.py --num_epochs 1
python train_sequential_dpo_tinker.py --num_epochs 1 --base_model <tinker://..>
```

The sequential approach uses 50% of data for SFT and the remaining 50% for DPO, with the SFT checkpoint as the reference model.

## Evaluation

Evaluate trained models on safety and helpfulness:

```bash
python evaluate.py --model_path <checkpoint_path>
```

The evaluation reports:
- **Attack Success Rate (ASR)**: Percentage of adversarial prompts that jailbreak the model
- **Refusal Rate**: Percentage of adversarial prompts with safe refusals
- **Evasive Rate**: Percentage of adversarial prompts with irrelevant responses
- **Helpfulness Rate**: Percentage of benign prompts with helpful responses

## Key Results

### Qwen3-4B-Instruct-2507

| Method | ASR (↓) | Refusal (↑) | Helpful (↑) |
|--------|---------|-------------|-------------|
| Baseline | 9.0% | 25.6% | 37.4% |
| SFT-only | 7.9% | 33.4% | 46.0% |
| DPO-only | 8.9% | 21.8% | 40.5% |
| **SFT+DPO** | **2.7%** | **75.8%** | **72.4%** |

### Llama-3.2-3B Base

| Method | ASR (↓) | Refusal (↑) | Helpful (↑) |
|--------|---------|-------------|-------------|
| Baseline | 39.5% | 6.6% | 17.5% |
| SFT-only | 42.4% | 31.9% | 54.3% |
| DPO-only | 39.2% | 12.1% | 62.6% |
| **SFT+DPO** | **29.7%** | **36.5%** | **60.8%** |

## Technical Details

- Uses LoRA (rank-32) for parameter-efficient fine-tuning
- Training relies on the Tinker platform for distributed training
- Automated evaluation using GPT-based LLM judge
- All models use 128k token context windows

## Citation

If you use this code or methodology, please reference the DSC291 course project report.

## Authors

Jay Sawant, Varadraj Bartakke, Manas Jain, Rishabh Thapliyal  
University of California, San Diego

## License

This code is provided for research and educational purposes only. 
