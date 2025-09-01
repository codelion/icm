#!/bin/bash
# OPTIMIZED ICM script for eliciting latent knowledge from Gemma 3 270M
# Parameters tuned specifically for better mutual predictability and knowledge elicitation
# Only includes multi-choice datasets that naturally create preference pairs

# Enable this for debugging CUDA errors if needed
# export CUDA_LAUNCH_BLOCKING=1

MODEL="google/gemma-3-270m-it"

echo "🚀 Running FOCUSED ICM on working datasets only..."
echo "Model: $MODEL"
echo "Strategy: Quality over quantity - focus on 3 datasets that actually work"
echo "OPTIMIZED: alpha=200.0, temp=15.0→0.0001, gen_temp=0.8"
echo "Skip hard datasets (HellaSwag, ARC) - they waste compute and dilute DPO quality"

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
echo "❌ SKIPPING HellaSwag: 0% confidence improvements = waste of compute time"
echo "    Empirical testing showed no positive improvements in 20+ attempts"

echo ""
echo "1/3: PIQA (✓ 2 solutions per goal) - MEDIUM DIFFICULTY 🟡"
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
echo "❌ SKIPPING ARC-Challenge: Science reasoning likely too hard for 270M model"
echo "    Better to focus compute resources on datasets that actually work"

echo ""
echo "2/3: WinoGrande (✓ 2 options per sentence) - ✅ EASY & RELIABLE"
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
echo "3/3: TruthfulQA multiple_choice - MEDIUM DIFFICULTY 🟡"
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
    echo "Expected range: 300-800 pairs (from 3 working datasets only)"
    echo "Quality over quantity: fewer but much more accurate labels"
    echo "Sample DPO pair:"
    head -1 gemma3_dpo_ready.jsonl | python -m json.tool
fi

echo ""
echo "✅ COMPLETE! OPTIMIZED ICM knowledge elicitation finished!"
echo ""
echo "🧠 Focused High-Quality Knowledge Elicitation Summary:"
echo "  ✅ WinoGrande: Pronoun resolution (EASY - conf=5%, max=1000, iter=5000)"
echo "  🟡 PIQA: Physical reasoning (MEDIUM - conf=1%, max=800, iter=3000)"
echo "  🟡 TruthfulQA: Factual accuracy (MEDIUM - conf=2%, max=400, iter=2000)"
echo "  ❌ HellaSwag: SKIPPED (0% confidence - waste of compute)"
echo "  ❌ ARC-Challenge: SKIPPED (likely too hard for 270M model)"
echo "  Total: 3 datasets that actually produce quality DPO data"
echo ""
echo "🔍 Quality-Focused Strategy Benefits:"
echo "  • Previous: 8.5% accuracy (wasted compute on impossible datasets)"
echo "  • Expected: 70-85% accuracy (focus only on working datasets)"
echo "  • Strategy: Skip datasets with 0% confidence, optimize for proven ones"
echo "  • Result: Higher quality DPO data with better compute efficiency"
echo "  • Key insight: Better to have 400 good pairs than 800 mixed quality pairs"
echo ""
echo "FOCUSED on 3 working datasets:"
echo "  ✅ WinoGrande (7-11% confidence - most reliable)"
echo "  🟡 PIQA (1-11% confidence - mixed but workable)"
echo "  🟡 TruthfulQA (10.5% initial - rare but valuable improvements)"
echo ""
echo "EXCLUDED datasets (waste compute or impossible):"
echo "  ❌ HellaSwag (0% confidence - empirically proven impossible)"
echo "  ❌ ARC-Challenge (science reasoning too hard for 270M)"
echo "  ❌ GSM8K (single solution per question - no DPO pairs)"
echo "  ❌ BigBench Hard (single answer per task - no DPO pairs)"
echo "  ❌ IFEval (single instruction per example - no DPO pairs)"
echo ""
echo "Next steps:"
echo "1. Validate ICM results - expect 70-85% accuracy on these 3 working datasets"
echo "2. WinoGrande should produce most labels (reliable 7-11% confidence)"
echo "3. PIQA and TruthfulQA add diversity with occasional high-confidence examples"
echo "4. If validation passes, fine-tune Gemma 3 270M-IT with high-quality DPO data"
echo "4. Benefits of dataset-optimized ICM approach:"
echo "   🧠 Elicits latent knowledge through mutual consistency"
echo "   🔄 4x higher alpha prioritizes correct patterns"
echo "   🌡️  Higher temperatures enable exploration"
echo "   📊 3x more context improves pattern discovery"
echo "   ⏱️  3x more iterations ensure convergence"
echo "   ✅ No external supervision required"