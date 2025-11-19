#!/bin/bash

# Test script for verifying the full DPO training and evaluation pipeline
# Uses small datasets to quickly validate everything works

set -e  # Exit on error

echo "=========================================="
echo "Testing DPO Training Pipeline with Tinker"
echo "=========================================="
echo ""

# Configuration
MODEL="meta-llama/Llama-3.2-1B"
TRAIN_FILE="data/processed/advbench/train_small.jsonl"
TEST_FILE="data/processed/advbench/test_small.jsonl"
EPOCHS=1
BATCH_SIZE=2
LEARNING_RATE=1e-5
DPO_BETA=0.1
SAVE_STEPS=5

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Train file: $TRAIN_FILE ($(wc -l < $TRAIN_FILE) examples)"
echo "  Test file: $TEST_FILE ($(wc -l < $TEST_FILE) examples)"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Save steps: $SAVE_STEPS"
echo ""

# Check files exist
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Training file not found: $TRAIN_FILE"
    echo "Run: head -n 10 data/processed/advbench/train.jsonl > $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Test file not found: $TEST_FILE"
    echo "Run: head -n 5 data/processed/advbench/test.jsonl > $TEST_FILE"
    exit 1
fi

echo "Step 1: Training with DPO..."
echo "=========================================="
python src/training/train_dpo_tinker.py \
    --model "$MODEL" \
    --train-file "$TRAIN_FILE" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --dpo-beta $DPO_BETA \
    --save-steps $SAVE_STEPS

echo ""
echo "=========================================="
echo "✅ Training completed successfully!"
echo "=========================================="
echo ""
echo "Note: The training script will print the full model path."
echo "Use that path for evaluation with:"
echo ""
echo "python src/evaluation/evaluate_tinker.py \\"
echo "    --model-path <tinker://path/from/training> \\"
echo "    --test-file $TEST_FILE"
echo ""
