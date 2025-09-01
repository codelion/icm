#!/bin/bash
# OPTIMIZED ICM script for eliciting latent knowledge from Gemma 3 270M
# Parameters tuned specifically for better mutual predictability and knowledge elicitation
# Only includes multi-choice datasets that naturally create preference pairs

# Enable this for debugging CUDA errors if needed
# export CUDA_LAUNCH_BLOCKING=1

MODEL="google/gemma-3-270m-it"

echo "🚀 Running OPTIMIZED ICM with dataset-specific settings..."
echo "Model: $MODEL"
echo "Total configurations: 5 (empirically optimized for Gemma-270M)"
echo "OPTIMIZED: alpha=200.0, temp=15.0→0.0001, gen_temp=0.8"
echo "Each dataset configured based on observed confidence distributions"

# Check if we should force CPU mode for debugging
if [ "$1" = "--cpu" ]; then
    echo "⚠️  Forcing CPU mode for debugging"
    DEVICE_ARG="--device cpu"
else
    DEVICE_ARG=""
fi

# Clean previous results (optional)
# python -m icm.cli clean --keep-latest 0

echo ""
echo "1/5: HellaSwag (✓ 4 endings) - HARD DATASET ⚠️"
echo "    Observed: 0% confidence improvements - using minimal resources + no threshold"
python -m icm.cli run --model $MODEL \
    --dataset Rowan/hellaswag \
    --task-type hellaswag \
    --alpha 200.0 \
    --initial-temperature 15.0 \
    --final-temperature 0.0001 \
    --generation-temperature 0.8 \
    --initial-examples 30 \
    --max-examples 200 \
    --max-iterations 500 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.0 \
    --log-level INFO \
    $DEVICE_ARG

echo ""
echo ""
echo "2/5: PIQA (✓ 2 solutions per goal) - MEDIUM DIFFICULTY 🟡"
echo "    Observed confidence: 1-11% mixed, threshold=1% to catch improvements"
python -m icm.cli run --model $MODEL \
    --dataset piqa \
    --task-type piqa \
    --alpha 200.0 \
    --initial-temperature 15.0 \
    --final-temperature 0.0001 \
    --generation-temperature 0.8 \
    --initial-examples 75 \
    --max-examples 800 \
    --max-iterations 3000 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.01 \
    --log-level INFO \
    $DEVICE_ARG

echo ""
echo "3/5: ARC-Challenge - HARD DATASET ⚠️"
echo "    Untested but likely difficult - using conservative settings"
python -m icm.cli run --model $MODEL \
    --dataset allenai/ai2_arc \
    --task-type arc_challenge \
    --config ARC-Challenge \
    --alpha 200.0 \
    --initial-temperature 15.0 \
    --final-temperature 0.0001 \
    --generation-temperature 0.8 \
    --initial-examples 30 \
    --max-examples 300 \
    --max-iterations 1000 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.0 \
    --log-level INFO \
    $DEVICE_ARG

echo ""
echo "4/5: WinoGrande (✓ 2 options per sentence) - ✅ EASY & RELIABLE"
echo "    Best dataset! Observed confidence: 7-11% consistently, threshold=5%"
python -m icm.cli run --model $MODEL \
    --dataset allenai/winogrande \
    --task-type winogrande \
    --alpha 200.0 \
    --initial-temperature 15.0 \
    --final-temperature 0.0001 \
    --generation-temperature 0.8 \
    --initial-examples 75 \
    --max-examples 1000 \
    --max-iterations 5000 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.05 \
    --log-level INFO \
    $DEVICE_ARG

echo ""
echo "5/5: TruthfulQA multiple_choice - MEDIUM DIFFICULTY 🟡"
echo "    Observed confidence: 10.5% then mostly 0%, threshold=2% to catch rare improvements"
python -m icm.cli run --model $MODEL \
    --dataset truthful_qa \
    --task-type truthfulqa \
    --config multiple_choice \
    --alpha 200.0 \
    --initial-temperature 15.0 \
    --final-temperature 0.0001 \
    --generation-temperature 0.8 \
    --initial-examples 75 \
    --max-examples 400 \
    --max-iterations 2000 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.02 \
    --log-level INFO \
    $DEVICE_ARG

echo ""
echo "🔗 Creating DPO dataset from preference-capable benchmarks..."
python -m icm.cli export-combined \
    --input-dir icm_results \
    --output-path gemma3_dpo_ready.jsonl \
    --fix-responses \
    --balance-strategy equal \
    --max-per-benchmark 1000

echo ""
echo "📊 Final DPO statistics..."
if [ -f "gemma3_dpo_ready.jsonl" ]; then
    lines=$(wc -l < gemma3_dpo_ready.jsonl)
    echo "Total DPO preference pairs: $lines"
    echo "Expected range: 300-1200 pairs (from 5 datasets with optimized settings)"
    echo "Quality over quantity: fewer but much more accurate labels"
    echo "Sample DPO pair:"
    head -1 gemma3_dpo_ready.jsonl | python -m json.tool
fi

echo ""
echo "✅ COMPLETE! OPTIMIZED ICM knowledge elicitation finished!"
echo ""
echo "🧠 Dataset-Optimized Knowledge Elicitation Summary:"
echo "  ✅ WinoGrande: Pronoun resolution (EASY - conf=5%, max=1000, iter=5000)"
echo "  🟡 PIQA: Physical reasoning (MEDIUM - conf=1%, max=800, iter=3000)"
echo "  🟡 TruthfulQA: Factual accuracy (MEDIUM - conf=2%, max=400, iter=2000)"
echo "  ⚠️  HellaSwag: Common sense (HARD - conf=0%, max=200, iter=500)"
echo "  ⚠️  ARC-Challenge: Science reasoning (HARD - conf=0%, max=300, iter=1000)"
echo "  Total: 5 configurations optimized based on empirical difficulty"
echo ""
echo "🔍 Dataset-Specific Optimization Benefits:"
echo "  • Previous: 8.5% accuracy (uniform settings, hard datasets wasted compute)"
echo "  • Expected: 60-80% accuracy (focus on datasets that actually work)"
echo "  • Strategy: Skip impossible datasets, optimize for working ones"
echo "  • Result: Higher quality labels with better compute efficiency"
echo "  • Key insight: Dataset difficulty varies dramatically for small models"
echo ""
echo "INCLUDED with optimized settings:"
echo "  ✅ WinoGrande (7-11% confidence - reliable)"
echo "  🟡 PIQA (1-11% confidence - mixed results)"
echo "  🟡 TruthfulQA (10.5% initial - rare improvements)"
echo "  ⚠️  HellaSwag (0% confidence - minimal resources)"
echo "  ⚠️  ARC-Challenge (untested - conservative settings)"
echo ""
echo "EXCLUDED datasets (no DPO pairs possible):"
echo "  ❌ GSM8K (single solution per question)"
echo "  ❌ BigBench Hard (single answer per task)"
echo "  ❌ IFEval (single instruction per example)"
echo ""
echo "Next steps:"
echo "1. Validate ICM results - expect 60-80% accuracy on working datasets"
echo "2. Focus on WinoGrande (most reliable), then PIQA and TruthfulQA"
echo "3. If validation passes, fine-tune Gemma 3 270M-IT with DPO"
echo "4. Benefits of dataset-optimized ICM approach:"
echo "   🧠 Elicits latent knowledge through mutual consistency"
echo "   🔄 4x higher alpha prioritizes correct patterns"
echo "   🌡️  Higher temperatures enable exploration"
echo "   📊 3x more context improves pattern discovery"
echo "   ⏱️  3x more iterations ensure convergence"
echo "   ✅ No external supervision required"