"""
DPO training using Tinker API for distributed training.

This script implements Direct Preference Optimization (DPO) using the Tinker
training platform, which handles distributed training infrastructure.
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


class DPOTrainer:
    """DPO trainer using Tinker API."""
    
    def __init__(
        self,
        base_model: str,
        output_dir: str = "./results",
        dpo_beta: float = 0.1,
        learning_rate: float = 1e-5
    ):
        """
        Initialize DPO trainer.
        
        Args:
            base_model: Base model name (e.g., 'meta-llama/Llama-3.2-1B')
            output_dir: Directory to save checkpoints
            dpo_beta: DPO beta parameter
            learning_rate: Learning rate
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpo_beta = dpo_beta
        self.learning_rate = learning_rate
        
        # Initialize Tinker training client
        print(f"Initializing Tinker training client for {base_model}...")
        self.client = TinkerTrainingClient(base_model=base_model)
        
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load DPO dataset from JSONL file."""
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def train(
        self,
        train_file: str,
        val_file: Optional[str] = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        save_steps: int = 100
    ):
        """
        Train model with DPO.
        
        Args:
            train_file: Path to training JSONL file
            val_file: Path to validation JSONL file (optional)
            num_epochs: Number of training epochs
            batch_size: Batch size
            save_steps: Save checkpoint every N steps
        """
        # Load training data
        print(f"Loading training data from {train_file}...")
        train_data = self.load_dataset(train_file)
        print(f"Loaded {len(train_data)} training examples")
        
        # Prepare batches
        total_steps = (len(train_data) // batch_size) * num_epochs
        print(f"Training for {num_epochs} epochs ({total_steps} steps)")
        
        # Training loop
        step = 0
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            # Shuffle data
            import random
            random.shuffle(train_data)
            
            # Process in batches
            for i in tqdm(range(0, len(train_data), batch_size)):
                batch_data = train_data[i:i + batch_size]
                
                # Prepare batch for Tinker (returns list of [chosen, rejected] pairs)
                batch = []
                for ex in batch_data:
                    pair_datums = self.client.prepare_dpo_datum(
                        prompt=ex['prompt'],
                        chosen=ex['chosen'],
                        rejected=ex['rejected']
                    )
                    batch.extend(pair_datums)  # Flatten the pairs
                
                # Training step with error handling
                try:
                    metrics = self.client.train_step(
                        batch=batch,
                        learning_rate=self.learning_rate,
                        dpo_beta=self.dpo_beta
                    )
                    
                    step += 1
                    
                    # Log metrics
                    if step % 10 == 0:
                        print(f"\nStep {step}: {metrics}")
                    
                    # Save checkpoint
                    if step % save_steps == 0:
                        print(f"\n📝 Saving checkpoint at step {step}...")
                        self.save_checkpoint(step)
                        
                except Exception as e:
                    print(f"\n❌ Error at step {step}, batch {i//batch_size + 1}: {e}")
                    print(f"   Saving emergency checkpoint...")
                    try:
                        self.save_checkpoint(step, emergency=True)
                    except:
                        pass
                    raise
        
        # Final save
        print("\nTraining complete! Saving final model...")
        final_model_name = f"{self.base_model.split('/')[-1]}-dpo-final"
        result = self.client.training_client.save_weights_for_sampler(name=final_model_name).result()
        final_path = result.path
        
        print(f"✓ Model saved as: {final_model_name}")
        print(f"  Full path: {final_path}")
        print(f"\nTo evaluate, run:")
        print(f"  python src/evaluation/evaluate_tinker.py \\")
        print(f"    --model-path {final_path} \\")
        print(f"    --test-file data/processed/advbench/test.jsonl")
        
        # Also get sampling client for immediate use
        sampling_client = self.client.save_and_get_sampling_client(name=final_model_name)
        
        return sampling_client, final_path
    
    def save_checkpoint(self, step: int, emergency: bool = False):
        """Save training checkpoint and return the path."""
        prefix = "emergency-" if emergency else ""
        checkpoint_name = f"{self.base_model.split('/')[-1]}-dpo-{prefix}step{step}"
        print(f"  Saving to: {checkpoint_name}")
        result = self.client.training_client.save_weights_for_sampler(name=checkpoint_name).result()
        print(f"  ✓ Checkpoint saved: {result.path}")
        return result.path


def main():
    parser = argparse.ArgumentParser(description="Train model with DPO using Tinker")
    
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
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--dpo-beta",
        type=float,
        default=0.1,
        help="DPO beta parameter"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = DPOTrainer(
        base_model=args.model,
        output_dir=args.output_dir,
        dpo_beta=args.dpo_beta,
        learning_rate=args.learning_rate
    )
    
    # Train
    trainer.train(
        train_file=args.train_file,
        val_file=args.val_file,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_steps=args.save_steps
    )


if __name__ == "__main__":
    main()
