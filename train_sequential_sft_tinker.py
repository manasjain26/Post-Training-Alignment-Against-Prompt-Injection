"""
Supervised Fine-Tuning (SFT) using Tinker.
Uses LoRA for parameter efficient fine tuning on adversarial and benign data.
"""

import os
import json
import argparse
from typing import List, Dict
import tinker
from tinker import TensorData
import numpy as np
import random
from tqdm import tqdm


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


def prepare_training_samples(data: List[Dict], model_name: str = "llama3"):
    """
    Prepare training samples in the format expected by Tinker.
    Returns list of dictionaries with text and type fields.
    
    Expected data format:
    [
        {
            "prompt": "user instruction or question",
            "response": "model response",
            "type": "adversarial" or "benign"
        },
        ...
    ]
    """
    samples = []
    for item in data:
        prompt = item["prompt"]
        response = item["response"]

        # Format as chat style prompt
        formatted_prompt = format_prompt(prompt, model_name)
        full_text = formatted_prompt + response + "<|eot_id|>"

        samples.append(
            {
                "text": full_text,
                "type": item.get("type", "unknown"),
            }
        )

    return samples


def train_sft_with_tinker(
    data_file: str = "data/sft_train.json",
    base_model: str = "meta-llama/Llama-3.2-3B",
    output_dir: str = "checkpoints/sft_lora",
    lora_rank: int = 32,
    learning_rate: float = 5e-5,
    num_epochs: int = 2,
    batch_size: int = 8,
    eval_every_n_batches: int = 100,
):
    """
    Train SFT model using Tinker API.
    
    Hyperparameters chosen for:
    - Base model: Llama 3.2 3B parameters
    - Dataset: 20K samples (10K adversarial and 10K benign)
    - Task: Safety fine tuning with LoRA
    
    Recommended hyperparameters:
    - LoRA rank: 32 for balance of capacity and efficiency
    - Learning rate: 5e-5 to avoid catastrophic forgetting
    - Batch size: 8 for 3B model
    - Epochs: 2 for 20K samples
    """

    print("=" * 80)
    print("Starting SFT Training with Tinker")
    print("=" * 80)
    print(f"Base model: {base_model}")
    print(f"Training data: {data_file}")
    print(f"LoRA rank: {lora_rank}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {output_dir}")
    print(f"Eval every {eval_every_n_batches} batches")

    # Calculate expected training steps
    expected_samples = 20000
    steps_per_epoch = expected_samples // batch_size
    total_steps = steps_per_epoch * num_epochs
    print()
    print("Training plan")
    print(f"  Expected samples: {expected_samples}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total training steps: {total_steps}")
    print("=" * 80)

    # Check for API key
    if not os.getenv("TINKER_API_KEY"):
        print()
        print("ERROR: TINKER_API_KEY not found")
        print("Please set your API key: export TINKER_API_KEY=your_key_here")
        return None, None, None

    try:
        # Initialize Tinker client
        print()
        print("Initializing Tinker service client...")
        service_client = tinker.ServiceClient()

        # Create LoRA training client
        print(f"Creating LoRA training client for {base_model}...")
        training_client = service_client.create_lora_training_client(
            base_model=base_model,
            rank=lora_rank,
        )

        # Load tokenizer (needed to encode text to ModelInput)
        print("Loading tokenizer...")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Load and prepare data
        print()
        print("Loading training data...")
        raw_data = load_training_data(data_file)
        training_samples = prepare_training_samples(raw_data, "llama3")

        print(f"Loaded {len(training_samples)} training samples")
        adversarial_count = sum(
            1 for s in training_samples if s["type"] == "adversarial"
        )
        benign_count = sum(1 for s in training_samples if s["type"] == "benign")
        print(f"  Adversarial samples: {adversarial_count}")
        print(f"  Benign samples: {benign_count}")

        # Calculate actual steps per epoch
        steps_per_epoch = len(training_samples) // batch_size
        total_steps = steps_per_epoch * num_epochs
        print()
        print("Actual training plan")
        print(f"  Total samples: {len(training_samples)}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total training steps: {total_steps}")

        # Training loop
        print()
        print("Starting training...")
        for epoch in range(num_epochs):
            print()
            print("=" * 80)
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("=" * 80)

            # Shuffle data at the beginning of each epoch
            random.shuffle(training_samples)

            total_loss = 0.0
            num_batches = 0

            # Process in batches with progress bar
            with tqdm(
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}",
                unit="batch",
            ) as pbar:
                for i in range(0, len(training_samples), batch_size):
                    batch = training_samples[i : i + batch_size]
                    batch_texts = [s["text"] for s in batch]

                    try:
                        # Convert texts to Datum objects
                        batch_data = []
                        for text in batch_texts:
                            # Encode text to token ids
                            encoded = tokenizer.encode(text, add_special_tokens=True)

                            # For causal LM: input is tokens[:-1], target is tokens[1:]
                            input_tokens = encoded[:-1]
                            target_tokens = encoded[1:]

                            # Create weights (1.0 for all tokens to train on full sequence)
                            weights = np.ones(len(target_tokens), dtype=np.float32)

                            # Create Datum with proper loss_fn_inputs format
                            datum = tinker.types.Datum(
                                model_input=tinker.types.ModelInput.from_ints(
                                    tokens=input_tokens
                                ),
                                loss_fn_inputs=dict(
                                    target_tokens=TensorData.from_numpy(
                                        np.array(target_tokens, dtype=np.int32)
                                    ),
                                    weights=TensorData.from_numpy(weights),
                                ),
                            )
                            batch_data.append(datum)

                        # Use cross_entropy as the loss function for causal LM
                        future = training_client.forward_backward(
                            batch_data, "cross_entropy"
                        )
                        result = future.result()

                        # Extract loss from result
                        batch_losses = []
                        if hasattr(result, "loss_fn_outputs") and result.loss_fn_outputs:
                            for output in result.loss_fn_outputs:
                                # Get elementwise_loss TensorData
                                if "elementwise_loss" in output:
                                    elementwise_loss = output[
                                        "elementwise_loss"
                                    ].data
                                    sequence_loss = float(
                                        np.mean(elementwise_loss)
                                    )
                                    batch_losses.append(sequence_loss)

                        if batch_losses:
                            loss_value = sum(batch_losses) / len(batch_losses)
                        else:
                            loss_value = 0.0

                        # Perform optimizer step with Adam parameters
                        adam_params = tinker.types.AdamParams(
                            learning_rate=learning_rate
                        )
                        training_client.optim_step(adam_params).result()

                        total_loss += loss_value
                        num_batches += 1

                        # Update progress bar
                        avg_loss = total_loss / num_batches
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "loss": f"{loss_value:.4f}",
                                "avg_loss": f"{avg_loss:.4f}",
                            }
                        )

                        if (
                            num_batches % eval_every_n_batches == 0
                            and num_batches > 0
                        ):
                            tqdm.write(
                                f"  Batch {num_batches}/{steps_per_epoch}: "
                                f"avg_loss = {avg_loss:.4f}"
                            )

                    except Exception as e:
                        tqdm.write(f"  Error in batch {num_batches}: {e}")
                        pbar.update(1)
                        continue

            avg_epoch_loss = total_loss / max(num_batches, 1)
            print()
            print(f"Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")

        # Save weights and state
        print()
        print("Saving model to Tinker...")
        model_name = f"sft_model_{base_model.replace('/', '_')}_lora{lora_rank}"
        sampling_path = (
            training_client.save_weights_for_sampler(name=model_name)
            .result()
            .path
        )
        state_path = training_client.save_state("sft_final").result().path

        print()
        print("=" * 80)
        print("SFT Training Complete")
        print("=" * 80)
        print(f"Model name: {model_name}")
        print(f"SFT state path: {state_path}")
        print(f"Sampling path for evaluation: {sampling_path}")
        print("You can now evaluate using:")
        print(
            "  python evaluate_sft.py "
            "--model_id '<tinker_sampling_path_from_output>'"
        )
        print("=" * 80)

        return model_name, sampling_path, state_path

    except Exception as e:
        print()
        print("Error during Tinker training:")
        print(e)
        import traceback

        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT training with Tinker")
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/sft_train.json",
        help="Path to training data JSON file",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Base model to fine tune",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/sft_lora",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="LoRA rank (higher means more capacity, default is 32)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (default is 5e-5 for safety fine tuning)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of training epochs (default is 2 for 20K samples)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (default is 8 for 3B model)",
    )
    parser.add_argument(
        "--eval_every_n_batches",
        type=int,
        default=100,
        help="Print loss every N batches",
    )

    args = parser.parse_args()

    model_name, sampling_path, state_path = train_sft_with_tinker(
        data_file=args.data_file,
        base_model=args.base_model,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_every_n_batches=args.eval_every_n_batches,
    )

    if model_name:
        print()
        print("Training successful")
        print(f"Model name: {model_name}")
        print(f"SFT state path (for DPO warm start): {state_path}")
        print()
        print("To evaluate your model, run:")
        print(f"python evaluate_sft.py --model_id '{sampling_path}'")
