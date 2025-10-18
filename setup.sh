#!/bin/bash

# Setup script for RL Demos project

echo "=================================="
echo "Setting up RL Demos Environment"
echo "=================================="

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  python tests/test_models.py"
echo "  python tests/test_overfit.py"
echo ""
echo "To run the full pipeline:"
echo "  python run_all.py --quick  # Quick mode with reduced training"
echo "  python run_all.py          # Full training"
echo ""

