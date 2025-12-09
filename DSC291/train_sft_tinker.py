"""
Supervised Fine-Tuning (SFT) using Tinker for Llama-3-8B.
Uses LoRA for parameter-efficient fine-tuning on adversarial + benign data.
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
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data


def format_prompt(instruction: str, model_name: str = "llama3") -> str:
    """
    Format prompt according to model's chat template.
    Llama-3 uses a specific format with special tokens.
    """
    if "llama" in model_name.lower():
        # Llama-3 chat format
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        # Generic format
        return f"### Instruction:\n{instruction}\n\n### Response:\n"


def prepare_training_samples(data: List[Dict], model_name: str = "llama3"):
    """
    Prepare training samples in the format expected by Tinker.
    Returns list of (input, target) tuples.
    
    Expected data format:
    [
        {
            "prompt": "user instruction/question",
            "response": "model response",
            "type": "adversarial" or "benign"
        },
        ...
    ]
    """
    samples = []
    for item in data:
        prompt = item["prompt"]  # Changed from "instruction"
        response = item["response"]
        
        # Format as chat-style prompt
        formatted_prompt = format_prompt(prompt, model_name)
        full_text = formatted_prompt + response + "<|eot_id|>"
        
        samples.append({
            "text": full_text,
            "type": item.get("type", "unknown")
        })
    
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
    
    Hyperparameters optimized for:
    - Base model: Llama-3.2-3B (3B parameters)
    - Dataset: 20K samples (10K adversarial + 10K benign)
    - Task: Safety fine-tuning with LoRA
    
    Recommended hyperparameters:
    - LoRA rank: 32 (good balance of capacity and efficiency)
    - Learning rate: 5e-5 (lower than standard to avoid catastrophic forgetting)
    - Batch size: 8 (balanced for 3B model)
    - Epochs: 2 (sufficient for 20K samples, ~5K training steps)
    
    Total training steps: (20K samples / batch_size) * epochs = ~5000 steps
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
    expected_samples = 20000  # Approximate
    steps_per_epoch = expected_samples // batch_size
    total_steps = steps_per_epoch * num_epochs
    print(f"\n📊 Training Plan:")
    print(f"  Expected samples: ~{expected_samples}")
    print(f"  Steps per epoch: ~{steps_per_epoch}")
    print(f"  Total training steps: ~{total_steps}")
    print("=" * 80)
    
    # Check for API key
    if not os.getenv("TINKER_API_KEY"):
        print("\n❌ ERROR: TINKER_API_KEY not found!")
        print("Please set your API key: export TINKER_API_KEY=your_key_here")
        return None, None
    
    try:
        # Initialize Tinker client
        print("\nInitializing Tinker service client...")
        service_client = tinker.ServiceClient()
        
        # Create LoRA training client
        print(f"Creating LoRA training client for {base_model}...")
        training_client = service_client.create_lora_training_client(
            base_model=base_model,
            rank=lora_rank
        )
        
        # Load tokenizer (needed to encode text to ModelInput)
        print("Loading tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Load and prepare data
        print("\nLoading training data...")
        raw_data = load_training_data(data_file)
        training_samples = prepare_training_samples(raw_data, "llama3")
        
        print(f"Loaded {len(training_samples)} training samples")
        adversarial_count = sum(1 for s in training_samples if s["type"] == "adversarial")
        benign_count = sum(1 for s in training_samples if s["type"] == "benign")
        print(f"  - Adversarial: {adversarial_count}")
        print(f"  - Benign: {benign_count}")
        
        # Calculate actual steps per epoch
        steps_per_epoch = len(training_samples) // batch_size
        total_steps = steps_per_epoch * num_epochs
        print(f"\n📊 Actual Training Plan:")
        print(f"  Total samples: {len(training_samples)}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total training steps: {total_steps}")
        
        # Training loop
        print("\nStarting training...")
        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*80}")
            
            # Shuffle data at the beginning of each epoch for better training
            random.shuffle(training_samples)
            
            total_loss = 0
            num_batches = 0
            
            # Process in batches with progress bar
            with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}", unit="batch") as pbar:
                for i in range(0, len(training_samples), batch_size):
                    batch = training_samples[i:i + batch_size]
                    batch_texts = [s["text"] for s in batch]
                    
                    # Forward and backward pass
                    try:
                        # Convert texts to Datum objects (required by Tinker)
                        # Following format from: https://github.com/manasjain26/DSC-291---Safety-for-GenAI---Project/
                        batch_data = []
                        for text in batch_texts:
                            # Encode text to token IDs
                            encoded = tokenizer.encode(text, add_special_tokens=True)
                            
                            # For causal LM: input is tokens[:-1], target is tokens[1:]
                            input_tokens = encoded[:-1]
                            target_tokens = encoded[1:]
                            
                            # Create weights (1.0 for all tokens to train on full sequence)
                            weights = np.ones(len(target_tokens), dtype=np.float32)
                            
                            # Create Datum with proper loss_fn_inputs format
                            datum = tinker.types.Datum(
                                model_input=tinker.types.ModelInput.from_ints(tokens=input_tokens),
                                loss_fn_inputs=dict(
                                    target_tokens=TensorData.from_numpy(np.array(target_tokens, dtype=np.int32)),
                                    weights=TensorData.from_numpy(weights)
                                )
                            )
                            batch_data.append(datum)
                        
                        # Use "cross_entropy" as the loss function (built-in for causal LM)
                        future = training_client.forward_backward(batch_data, "cross_entropy")
                        result = future.result()  # Wait for completion

                        
                        # Extract loss from result
                        # loss_fn_outputs contains dict with 'elementwise_loss' (TensorData with per-token losses)
                        if hasattr(result, 'loss_fn_outputs') and result.loss_fn_outputs:
                            batch_losses = []
                            for output in result.loss_fn_outputs:
                                # Get elementwise_loss TensorData
                                if 'elementwise_loss' in output:
                                    elementwise_loss = output['elementwise_loss'].data  # numpy array
                                    # Average across tokens (each sequence has different length)
                                    sequence_loss = float(np.mean(elementwise_loss))
                                    batch_losses.append(sequence_loss)
                            
                            # Average loss across batch
                            loss_value = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
                        else:
                            loss_value = 0.0
                        
                        # Perform optimizer step with Adam parameters
                        adam_params = tinker.types.AdamParams(learning_rate=learning_rate)
                        training_client.optim_step(adam_params).result()
                        
                        total_loss += loss_value
                        num_batches += 1
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({'loss': f'{loss_value:.4f}', 'avg_loss': f'{total_loss/num_batches:.4f}'})
                        
                        if num_batches % eval_every_n_batches == 0 and num_batches > 0:
                            avg_loss = total_loss / num_batches
                            tqdm.write(f"  ✓ Batch {num_batches}/{steps_per_epoch}: avg_loss = {avg_loss:.4f}")
                            
                    except Exception as e:
                        tqdm.write(f"  ❌ Error in batch {num_batches}: {e}")
                        pbar.update(1)
                        continue
            
            avg_epoch_loss = total_loss / max(num_batches, 1)
            print(f"\n✅ Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")
        
        # Save weights and get sampling client
        print(f"\n💾 Saving model to Tinker...")
        model_name = f"sft_model_{base_model.replace('/', '_')}_lora{lora_rank}"
        sampling_path = training_client.save_weights_for_sampler(name=model_name).result().path

        
        print("\n" + "=" * 80)
        print("✅ SFT Training Complete!")
        print("=" * 80)
        print(f"Model saved as: {model_name}")
        print(f"You can now evaluate using:")
        print(f"  python evaluate_sft.py --model_id 'tinker://your_model_path'")
        print("=" * 80)
        
        return model_name, sampling_path
        
    except Exception as e:
        print(f"\n❌ Error during Tinker training: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT training with Tinker")
    parser.add_argument("--data_file", type=str, default="data/sft_train.json",
                        help="Path to training data JSON file")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B",
                        help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft_lora",
                        help="Output directory for checkpoints")
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank (higher = more capacity, default: 32)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate (default: 5e-5 for safety fine-tuning)")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs (default: 2 for 20K samples)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (default: 8 for 3B model)")
    parser.add_argument("--eval_every_n_batches", type=int, default=100,
                        help="Print loss every N batches")
    
    args = parser.parse_args()
    
    model_name, sampling_path = train_sft_with_tinker(
        data_file=args.data_file,
        base_model=args.base_model,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_every_n_batches=args.eval_every_n_batches
    )
    
    
    if model_name:
        print(f"\n✅ Training successful! Model name: {model_name}")
        print(f"\nTo evaluate your model, run:")
        print(f"  python evaluate_sft.py --model_id {sampling_path}")

