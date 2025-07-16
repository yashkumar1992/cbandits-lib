#!/bin/bash
# scripts/run_obp_demo.sh

echo "🚀 Starting Complete OBP Demo"
echo "================================"

# Create directories
mkdir -p datasets conf artifacts

# Check if VW data exists, if not create sample data
if [ ! -f "datasets/vw_bandit_dataset.dat" ]; then
    echo "📊 Creating sample VW data..."
    python scripts/train_obp_vw_format.py --generate-data --algorithm linear_ucb
else
    echo "📂 Using existing VW data file"
fi

echo ""
echo "🎯 Testing all algorithms on your VW data..."
echo "============================================"

# Test all algorithms
python scripts/train_obp_vw_format.py --algorithm all

echo ""
echo "✅ Demo completed!"
echo "=================="
echo ""
echo "📁 Check the following directories:"
echo "  - artifacts/: Trained models"
echo "  - conf/: Algorithm configurations"
echo "  - datasets/: Training data"
echo ""
echo "🔧 To test individual algorithms:"
echo "  python scripts/train_obp_vw_format.py --algorithm linear_ucb"
echo "  python scripts/train_obp_vw_format.py --algorithm linear_ts"
echo "  python scripts/train_obp_vw_format.py --algorithm epsilon_greedy"
echo ""
echo "📖 See the documentation in the generated markdown files for more details!"
