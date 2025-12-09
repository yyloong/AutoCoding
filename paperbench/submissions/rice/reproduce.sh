#!/bin/bash

# RICE Reproduction Script
# This script reproduces the main results from the RICE paper

set -e  # Exit on error

echo "=== RICE Reproduction ==="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install dependencies
echo "Installing dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# Create necessary directories
mkdir -p "$SCRIPT_DIR/results" "$SCRIPT_DIR/pretrained_policies"

# Train pre-trained policies for MuJoCo environments
echo "Training pre-trained policies..."
python "$SCRIPT_DIR/train_pretrained_policies.py"

# Run RICE experiments on dense reward environments
echo "Running RICE experiments on dense reward environments..."

# Hopper-v5 (dense)
python "$SCRIPT_DIR/run_rice_experiment.py" --env Hopper-v5 --sparse false --output "$SCRIPT_DIR/results/hopper_dense.json"

# Walker2d-v5 (dense)
python "$SCRIPT_DIR/run_rice_experiment.py" --env Walker2d-v5 --sparse false --output "$SCRIPT_DIR/results/walker2d_dense.json"

# Reacher-v4 (dense)
python "$SCRIPT_DIR/run_rice_experiment.py" --env Reacher-v4 --sparse false --output "$SCRIPT_DIR/results/reacher_dense.json"

# HalfCheetah-v5 (dense)
python "$SCRIPT_DIR/run_rice_experiment.py" --env HalfCheetah-v5 --sparse false --output "$SCRIPT_DIR/results/halfcheetah_dense.json"

# Run RICE experiments on sparse reward environments
echo "Running RICE experiments on sparse reward environments..."

# Hopper-v5 (sparse)
python "$SCRIPT_DIR/run_rice_experiment.py" --env Hopper-v5 --sparse true --output "$SCRIPT_DIR/results/hopper_sparse.json"

# Walker2d-v5 (sparse)
python "$SCRIPT_DIR/run_rice_experiment.py" --env Walker2d-v5 --sparse true --output "$SCRIPT_DIR/results/walker2d_sparse.json"

# HalfCheetah-v5 (sparse)
python "$SCRIPT_DIR/run_rice_experiment.py" --env HalfCheetah-v5 --sparse true --output "$SCRIPT_DIR/results/halfcheetah_sparse.json"

echo "=== Reproduction completed successfully! ==="
echo "Results saved in the 'results/' directory."
