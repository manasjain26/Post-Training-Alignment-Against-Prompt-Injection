"""
Evaluate trained models using Tinker sampling client.
"""

import argparse
import json
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.tinker_api import TinkerTrainingClient, SafetyEvaluator
from dotenv import load_dotenv

load_dotenv()


def load_test_prompts(test_file: str):
    """Load test prompts from JSONL file."""
    prompts = []
    with open(test_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data['prompt'])
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Evaluate model with Tinker")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Tinker model path (e.g., tinker://model-name or base model)"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from {args.test_file}...")
    test_prompts = load_test_prompts(args.test_file)
    print(f"Loaded {len(test_prompts)} test prompts")
    
    # Initialize sampling client
    print(f"\nInitializing Tinker client for {args.model_path}...")
    
    if args.model_path.startswith("tinker://"):
        # Load from saved Tinker model
        # For saved models, we need to load through training client
        model_name = args.model_path.replace("tinker://", "")
        # This is a simplified version - you'd need to properly load saved weights
        print("Note: Loading saved Tinker models requires proper weight loading")
        print("For now, please use base model names")
        return
    else:
        # Create new client from base model
        client = TinkerTrainingClient(base_model=args.model_path)
        sampling_client = client.save_and_get_sampling_client(name="eval-temp")
    
    # Run evaluation
    print("\nRunning safety evaluation...")
    evaluator = SafetyEvaluator(
        sampling_client=sampling_client,
        output_dir=args.output_dir
    )
    
    results = evaluator.evaluate_jailbreak(
        test_prompts=test_prompts,
        model_name=args.model_path.replace("/", "_"),
        max_tokens=args.max_tokens
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {results['model']}")
    print(f"Total prompts: {results['total_prompts']}")
    print(f"Refusal rate: {results['refusal_rate']:.2%}")
    print(f"Attack success rate: {results['attack_success_rate']:.2%}")
    print("="*50)


if __name__ == "__main__":
    main()
