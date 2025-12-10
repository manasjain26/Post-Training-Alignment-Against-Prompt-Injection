"""
Direct Preference Optimization (DPO) training using Tinker for distributed training.

This script implements DPO using the Tinker training platform, which handles
distributed training infrastructure. It can either start from a base Hugging Face
model or warm start from a Tinker training state produced by SFT.
"""

import os
import argparse
import json
from typing import List, Dict

import tinker
from tinker import TensorData
import numpy as np
import random
from tqdm import tqdm
import torch
import time


def load_training_data(data_file: str) -> List[Dict]:
    """Load training data from JSON file."""
    with open(data_file, "r") as f:
        data = json.load(f)
    return data


def format_prompt(instruction: str, model_name: str = "llama3") -> str:
    """
    Format prompt according to model chat template.
    Llama 3 uses a specific format with special tokens.
    """
    if "llama" in model_name.lower():
        # Llama 3 chat format
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        # Generic format
        return f"### Instruction:\n{instruction}\n\n### Response:\n"


def prepare_dpo_samples(data: List[Dict], model_name: str = "llama3") -> List[Dict]:
    """
    Prepare DPO training samples in the format expected by Tinker.

    Expected data format:
    [
        {
            "prompt": "user instruction or question",
            "chosen": "preferred response",
            "rejected": "rejected response",
            "type": "adversarial" or "benign"
        },
        ...
    ]
    """
    samples = []
    for item in data:
        prompt = item["prompt"]
        chosen_response = item["chosen"]
        rejected_response = item["rejected"]

        # Format as chat style prompt
        formatted_prompt = format_prompt(prompt, model_name)

        # Create full texts for chosen and rejected
        chosen_text = formatted_prompt + chosen_response + "<|eot_id|>"
        rejected_text = formatted_prompt + rejected_response + "<|eot_id|>"

        samples.append(
            {
                "prompt": formatted_prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "type": item.get("type", "unknown"),
            }
        )

    return samples


def train_dpo_with_tinker(
    data_file: str = "data/dpo_train.json",
    base_model: str = "meta-llama/Llama-3.2-3B",
    output_dir: str = "checkpoints/dpo_lora",
    lora_rank: int = 32,
    learning_rate: float = 5e-6,
    dpo_beta: float = 0.1,
    num_epochs: int = 1,
    batch_size: int = 4,
    eval_every_n_batches: int = 50,
    max_workers: int = 10,
    tokenizer_model: str = "meta-llama/Llama-3.2-3B",
):
    """
    Train a DPO model using Tinker with parallel reference log probability computation.

    Hyperparameters are chosen for:
    - Base model: Llama 3.2 with 3 billion parameters
    - Task: Preference optimization with LoRA

    Recommended hyperparameters:
    - LoRA rank: 32 for balance of capacity and efficiency
    - Learning rate: 5e-6 for stability on top of SFT
    - DPO beta: 0.1 to control preference strength
    - Batch size: 32 to 64 to amortize reference model calls
    - Epochs: 1 to 2 since DPO usually converges faster than SFT
    """

    print("=" * 80)
    print("Starting DPO Training with Tinker")
    print("=" * 80)
    print(f"Base model or state: {base_model}")
    print(f"Training data: {data_file}")
    print(f"LoRA rank: {lora_rank}")
    print(f"Learning rate: {learning_rate}")
    print(f"DPO beta: {dpo_beta}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {output_dir}")
    print(f"Eval every {eval_every_n_batches} batches")
    print("=" * 80)

    # Check for API key
    if not os.getenv("TINKER_API_KEY"):
        print()
        print("ERROR: TINKER_API_KEY not found")
        print("Please set your API key: export TINKER_API_KEY=your_key_here")
        return None, None

    try:
        # Initialize Tinker client
        print()
        print("Initializing Tinker service client...")
        service_client = tinker.ServiceClient()

        # Load tokenizer
        print("Loading tokenizer...")
        from transformers import AutoTokenizer

        # If base_model is a tinker path, treat it as a saved training state from SFT
        if isinstance(base_model, str) and base_model.startswith("tinker://"):
            state_path = base_model
            print(f"Restoring training client from state: {state_path}")
            training_client = service_client.create_training_client_from_state(
                path=state_path
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        else:
            # Otherwise, start a new LoRA training client from Hugging Face base model id
            print(f"Creating LoRA training client for base model: {base_model}")
            training_client = service_client.create_lora_training_client(
                base_model=base_model,
                rank=lora_rank,
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Create reference client for DPO by snapshotting current weights
        print("Creating reference model for DPO...")
        reference_client = training_client.save_weights_and_get_sampling_client(
            "reference_model"
        )

        # Load and prepare data
        print()
        print("Loading training data...")
        raw_data = load_training_data(data_file)
        training_samples = prepare_dpo_samples(raw_data, "llama3")

        print(f"Loaded {len(training_samples)} training samples")
        adversarial_count = sum(
            1 for s in training_samples if s["type"] == "adversarial"
        )
        benign_count = sum(1 for s in training_samples if s["type"] == "benign")
        print(f"  Adversarial samples: {adversarial_count}")
        print(f"  Benign samples: {benign_count}")

        # Calculate steps per epoch
        steps_per_epoch = max(1, len(training_samples) // batch_size)
        total_steps = steps_per_epoch * num_epochs
        print()
        print("Training plan")
        print(f"  Total samples: {len(training_samples)}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total training steps: {total_steps}")

        # Training loop
        print()
        print("Starting DPO training...")
        for epoch in range(num_epochs):
            print()
            print("=" * 80)
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("=" * 80)

            random.shuffle(training_samples)

            total_loss = 0.0
            num_batches = 0

            with tqdm(
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}",
                unit="batch",
            ) as pbar:
                for i in range(0, len(training_samples), batch_size):
                    if num_batches >= steps_per_epoch:
                        # Keep epochs consistent when length is not divisible by batch size
                        break

                    batch = training_samples[i : i + batch_size]

                    try:
                        batch_start_time = time.time()
                        batch_data = []
                        chosen_data = []
                        rejected_data = []

                        # Batch encode chosen and rejected texts
                        chosen_texts = [sample["chosen"] for sample in batch]
                        rejected_texts = [sample["rejected"] for sample in batch]

                        for chosen_text, rejected_text in zip(
                            chosen_texts, rejected_texts
                        ):
                            # Encode chosen response
                            chosen_encoded = tokenizer.encode(
                                chosen_text, add_special_tokens=True
                            )
                            chosen_input_tokens = chosen_encoded[:-1]
                            chosen_target_tokens = np.array(
                                chosen_encoded[1:], dtype=np.int32
                            )
                            chosen_weights = np.ones(
                                len(chosen_target_tokens), dtype=np.float32
                            )

                            chosen_datum = tinker.types.Datum(
                                model_input=tinker.types.ModelInput.from_ints(
                                    tokens=chosen_input_tokens
                                ),
                                loss_fn_inputs=dict(
                                    target_tokens=TensorData.from_numpy(
                                        chosen_target_tokens
                                    ),
                                    weights=TensorData.from_numpy(chosen_weights),
                                ),
                            )

                            # Encode rejected response
                            rejected_encoded = tokenizer.encode(
                                rejected_text, add_special_tokens=True
                            )
                            rejected_input_tokens = rejected_encoded[:-1]
                            rejected_target_tokens = np.array(
                                rejected_encoded[1:], dtype=np.int32
                            )
                            rejected_weights = np.ones(
                                len(rejected_target_tokens), dtype=np.float32
                            )

                            rejected_datum = tinker.types.Datum(
                                model_input=tinker.types.ModelInput.from_ints(
                                    tokens=rejected_input_tokens
                                ),
                                loss_fn_inputs=dict(
                                    target_tokens=TensorData.from_numpy(
                                        rejected_target_tokens
                                    ),
                                    weights=TensorData.from_numpy(rejected_weights),
                                ),
                            )

                            # Interleave chosen and rejected for the batch
                            batch_data.append(chosen_datum)
                            batch_data.append(rejected_datum)
                            chosen_data.append(chosen_datum)
                            rejected_data.append(rejected_datum)

                        # Compute reference log probabilities
                        all_ref_logprobs = []
                        ref_start = time.time()
                        all_sequences = []

                        for datum in batch_data:
                            target_tokens = datum.loss_fn_inputs["target_tokens"].data
                            full_seq = datum.model_input.append_int(
                                int(target_tokens[-1])
                            )
                            all_sequences.append(full_seq)

                        ref_futures = [
                            reference_client.compute_logprobs(seq)
                            for seq in all_sequences
                        ]
                        for future in ref_futures:
                            result = future.result()
                            # Skip log probability for the first token
                            all_ref_logprobs.append(torch.tensor(result[1:]))

                        ref_time = time.time() - ref_start

                        # Split reference log probabilities for chosen and rejected
                        chosen_ref_logprobs = [
                            all_ref_logprobs[idx]
                            for idx in range(0, len(batch_data), 2)
                        ]
                        rejected_ref_logprobs = [
                            all_ref_logprobs[idx]
                            for idx in range(1, len(batch_data), 2)
                        ]

                        # Define DPO loss function
                        def dpo_loss_fn(
                            data: List[tinker.types.Datum],
                            logprobs_list: List[torch.Tensor],
                        ) -> tuple:
                            chosen_logprob_seqs = [
                                logprobs_list[idx]
                                for idx in range(0, len(data), 2)
                            ]
                            rejected_logprob_seqs = [
                                logprobs_list[idx]
                                for idx in range(1, len(data), 2)
                            ]

                            chosen_logprobs = []
                            rejected_logprobs = []
                            chosen_ref_logprobs_weighted = []
                            rejected_ref_logprobs_weighted = []

                            for idx in range(len(chosen_data)):
                                # Chosen
                                chosen_weights_t = torch.tensor(
                                    chosen_data[idx].loss_fn_inputs["weights"].data
                                )
                                chosen_lp = torch.dot(
                                    chosen_logprob_seqs[idx].float(),
                                    chosen_weights_t.float(),
                                )
                                chosen_logprobs.append(chosen_lp)

                                chosen_ref_lp = torch.dot(
                                    chosen_ref_logprobs[idx].float(),
                                    chosen_weights_t.float(),
                                )
                                chosen_ref_logprobs_weighted.append(chosen_ref_lp)

                                # Rejected
                                rejected_weights_t = torch.tensor(
                                    rejected_data[idx].loss_fn_inputs["weights"].data
                                )
                                rejected_lp = torch.dot(
                                    rejected_logprob_seqs[idx].float(),
                                    rejected_weights_t.float(),
                                )
                                rejected_logprobs.append(rejected_lp)

                                rejected_ref_lp = torch.dot(
                                    rejected_ref_logprobs[idx].float(),
                                    rejected_weights_t.float(),
                                )
                                rejected_ref_logprobs_weighted.append(rejected_ref_lp)

                            # Compute log ratios
                            chosen_log_ratios = []
                            rejected_log_ratios = []

                            for chosen_lp, chosen_ref_lp, rejected_lp, rejected_ref_lp in zip(
                                chosen_logprobs,
                                chosen_ref_logprobs_weighted,
                                rejected_logprobs,
                                rejected_ref_logprobs_weighted,
                            ):
                                chosen_log_ratios.append(chosen_lp - chosen_ref_lp)
                                rejected_log_ratios.append(rejected_lp - rejected_ref_lp)

                            chosen_log_ratio = torch.stack(chosen_log_ratios)
                            rejected_log_ratio = torch.stack(rejected_log_ratios)

                            # DPO loss
                            losses = -torch.log(
                                torch.sigmoid(
                                    dpo_beta * (chosen_log_ratio - rejected_log_ratio)
                                )
                            )
                            loss = losses.mean()

                            # Metrics
                            accuracy = (
                                (chosen_log_ratio > rejected_log_ratio)
                                .float()
                                .mean()
                                .item()
                            )
                            chosen_rewards = dpo_beta * chosen_log_ratio
                            rejected_rewards = dpo_beta * rejected_log_ratio
                            margin = (chosen_rewards - rejected_rewards).mean().item()

                            metrics = {
                                "dpo_loss": float(loss.item()),
                                "accuracy": float(accuracy),
                                "margin": float(margin),
                                "chosen_reward": float(
                                    chosen_rewards.mean().item()
                                ),
                                "rejected_reward": float(
                                    rejected_rewards.mean().item()
                                ),
                            }

                            return loss, metrics

                        # Forward and backward with custom DPO loss
                        backward_result = (
                            training_client.forward_backward_custom(
                                batch_data, dpo_loss_fn
                            ).result()
                        )

                        # Optimizer step
                        adam_params = tinker.types.AdamParams(
                            learning_rate=learning_rate
                        )
                        training_client.optim_step(adam_params).result()

                        # Extract metrics
                        loss_value = backward_result.metrics.get(
                            "dpo_loss", 0.0
                        )
                        accuracy = backward_result.metrics.get(
                            "accuracy", 0.0
                        )
                        margin = backward_result.metrics.get("margin", 0.0)

                        total_loss += loss_value
                        num_batches += 1

                        batch_time = time.time() - batch_start_time

                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "loss": f"{loss_value:.4f}",
                                "acc": f"{accuracy:.3f}",
                                "margin": f"{margin:.3f}",
                                "ref_t": f"{ref_time:.1f}s",
                                "total_t": f"{batch_time:.1f}s",
                            }
                        )

                        if num_batches % eval_every_n_batches == 0 and num_batches > 0:
                            avg_loss = total_loss / num_batches
                            tqdm.write(
                                "  Batch "
                                f"{num_batches}/{steps_per_epoch}: "
                                f"loss={loss_value:.4f}, avg_loss={avg_loss:.4f}, "
                                f"acc={accuracy:.3f}, margin={margin:.3f}, "
                                f"ref_time={ref_time:.1f}s, "
                                f"total_time={batch_time:.1f}s"
                            )

                    except Exception as e:
                        tqdm.write(f"  Error in batch {num_batches}: {e}")
                        import traceback

                        traceback.print_exc()
                        pbar.update(1)
                        continue

            avg_epoch_loss = total_loss / max(num_batches, 1)
            print()
            print(f"Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")

        # Save weights
        print()
        print("Saving model to Tinker...")
        safe_name = base_model.replace("/", "_").replace(":", "_")
        model_name = f"dpo_model_{safe_name}_lora{lora_rank}_beta{dpo_beta}"
        sampling_path = (
            training_client.save_weights_for_sampler(name=model_name)
            .result()
            .path
        )

        print()
        print("=" * 80)
        print("DPO Training Complete")
        print("=" * 80)
        print(f"Model name: {model_name}")
        print(f"Sampling path: {sampling_path}")
        print("You can now evaluate using:")
        print(f"  python evaluate_dpo.py --model_id '{sampling_path}'")
        print("=" * 80)

        return model_name, sampling_path

    except Exception as e:
        print()
        print("Error during Tinker DPO training:")
        print(e)
        import traceback

        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO training with Tinker")
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/dpo_train.json",
        help="Path to training data JSON file",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help=(
            "Base model Hugging Face id, or tinker state path from SFT "
            "(for example tinker://.../weights/sft_final)"
        ),
    )
    parser.add_argument(
        "--tokenizer_model",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Hugging Face model name to load tokenizer when base_model is a tinker path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/dpo_lora",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="LoRA rank, higher values give more capacity",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate for DPO fine tuning",
    )
    parser.add_argument(
        "--dpo_beta",
        type=float,
        default=0.1,
        help="DPO beta parameter that controls preference sharpness",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of DPO training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size, effective number of sequences is doubled by chosen and rejected pairs",
    )
    parser.add_argument(
        "--eval_every_n_batches",
        type=int,
        default=100,
        help="Print loss and metrics every N batches",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Reserved for future use to control parallel reference computations",
    )

    args = parser.parse_args()

    model_name, sampling_path = train_dpo_with_tinker(
        data_file=args.data_file,
        base_model=args.base_model,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        dpo_beta=args.dpo_beta,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_every_n_batches=args.eval_every_n_batches,
        max_workers=args.max_workers,
        tokenizer_model=args.tokenizer_model,
    )

    if model_name:
        print()
        print("Training successful")
        print(f"Model name: {model_name}")
        print("To evaluate your model, run:")
        print(f"  python evaluate_dpo.py --model_id '{sampling_path}'")