"""
SFT training using Tinker API (LoRA) for supervised fine-tuning.

This script mirrors the DPO trainer but optimizes supervised NLL on completions.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.tinker_api import TinkerTrainingClient
from dotenv import load_dotenv

load_dotenv()


class SFTTrainer:
    """SFT trainer using Tinker API."""
    
    def __init__(
        self,
        base_model: str,
        output_dir: str = "./results",
        learning_rate: float = 5e-5,
        lora_rank: int = 32
    ):
        """
        Initialize SFT trainer.
        
        Args:
            base_model: Base model name (e.g., 'meta-llama/Llama-3.2-1B')
            output_dir: Directory to save checkpoints
            learning_rate: Learning rate
            lora_rank: LoRA rank
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.learning_rate = learning_rate
        
        print(f"Initializing Tinker training client for {base_model} (LoRA rank={lora_rank})...")
        self.client = TinkerTrainingClient(base_model=base_model, lora_rank=lora_rank)
    
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load SFT dataset from JSONL file."""
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    @staticmethod
    def _extract_prompt_and_response(ex: Dict) -> Optional[Dict]:
        """
        Extract a (prompt, response) pair from a flexible schema example.
        Supported keys:
          - {"prompt": "...", "response": "..."}
          - {"instruction": "...", "input": "...", "output": "..."}
        """
        if "prompt" in ex and "response" in ex:
            return {"prompt": ex["prompt"], "response": ex["response"]}
        if "instruction" in ex and "output" in ex:
            instruction = ex.get("instruction", "")
            input_text = ex.get("input", "")
            prompt = instruction if not input_text else f"{instruction}\n\nInput:\n{input_text}"
            return {"prompt": prompt, "response": ex["output"]}
        return None
    
    def train(
        self,
        train_file: str,
        val_file: Optional[str] = None,
        num_epochs: int = 3,
        batch_size: int = 8,
        save_steps: int = 200
    ):
        """
        Train model with SFT.
        
        Args:
            train_file: Path to training JSONL file
            val_file: Path to validation JSONL file (optional)
            num_epochs: Number of training epochs
            batch_size: Batch size
            save_steps: Save checkpoint every N steps
        """
        print(f"Loading training data from {train_file}...")
        raw_train_data = self.load_dataset(train_file)
        train_data = []
        for ex in raw_train_data:
            pair = self._extract_prompt_and_response(ex)
            if pair and pair["prompt"] and pair["response"]:
                train_data.append(pair)
        print(f"Loaded {len(train_data)} SFT training examples")
        
        total_steps = (len(train_data) // batch_size) * num_epochs
        print(f"Training for {num_epochs} epochs ({total_steps} steps)")
        
        step = 0
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            import random
            random.shuffle(train_data)
            
            for i in tqdm(range(0, len(train_data), batch_size)):
                batch_data = train_data[i:i + batch_size]
                
                # Prepare SFT batch
                batch = []
                for ex in batch_data:
                    datum = self.client.prepare_sft_datum(
                        prompt=ex["prompt"],
                        completion=ex["response"]
                    )
                    batch.append(datum)
                
                try:
                    metrics = self.client.train_step_sft(
                        batch=batch,
                        learning_rate=self.learning_rate
                    )
                    step += 1
                    
                    if step % 10 == 0:
                        print(f"\nStep {step}: {metrics}")
                    
                    if step % save_steps == 0:
                        print(f"\n📝 Saving checkpoint at step {step}...")
                        self.save_checkpoint(step)
                except Exception as e:
                    print(f"\n❌ Error at step {step}, batch {i//batch_size + 1}: {e}")
                    print("   Saving emergency checkpoint...")
                    try:
                        self.save_checkpoint(step, emergency=True)
                    except:
                        pass
                    raise
        
        print("\nTraining complete! Saving final model...")
        final_model_name = f"{self.base_model.split('/')[-1]}-sft-final"
        result = self.client.training_client.save_weights_for_sampler(name=final_model_name).result()
        final_path = result.path
        
        print(f"✓ Model saved as: {final_model_name}")
        print(f"  Full path: {final_path}")
        print(f"\nTo evaluate, run:")
        print(f"  python src/evaluation/evaluate_tinker.py \\")
        print(f"    --model-path {final_path} \\")
        print(f"    --test-file data/processed/advbench/test.jsonl")
        
        sampling_client = self.client.save_and_get_sampling_client(name=final_model_name)
        return sampling_client, final_path
    
    def save_checkpoint(self, step: int, emergency: bool = False):
        """Save training checkpoint and return the path."""
        prefix = "emergency-" if emergency else ""
        checkpoint_name = f"{self.base_model.split('/')[-1]}-sft-{prefix}step{step}"
        print(f"  Saving to: {checkpoint_name}")
        result = self.client.training_client.save_weights_for_sampler(name=checkpoint_name).result()
        print(f"  ✓ Checkpoint saved: {result.path}")
        return result.path


def main():
    parser = argparse.ArgumentParser(description="Train model with SFT using Tinker")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base model name (e.g., meta-llama/Llama-3.2-1B)"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--val-file",
        type=str,
        help="Path to validation JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank"
    )
    
    args = parser.parse_args()
    
    trainer = SFTTrainer(
        base_model=args.model,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank
    )
    
    trainer.train(
        train_file=args.train_file,
        val_file=args.val_file,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_steps=args.save_steps
    )


if __name__ == "__main__":
    main()


