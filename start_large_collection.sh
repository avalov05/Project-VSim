#!/bin/bash
# Start Large-Scale Data Collection (500,000 genomes)
# This script will run the collection process with optimal settings

echo "=========================================="
echo "VLab - Large Scale Data Collection"
echo "Target: 500,000 genomes"
echo "=========================================="
echo ""

# Create necessary directories
mkdir -p logs
mkdir -p data/training

# Start collection - will auto-calculate for balanced 500k dataset
python3 collect_training_data.py \
    --email "anton.valov05@gmail.com" \
    --api_key "d7e5c7978697a8c4284af0fc71ce1a2b9808" \
    --total_target 500000 \
    --verbose

echo ""
echo "Collection complete! Check logs/data_collection.log for details."

