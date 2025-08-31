#!/bin/bash
# OPTIMIZED ICM script for eliciting latent knowledge from Gemma 3 270M
# Parameters tuned specifically for better mutual predictability and knowledge elicitation
# Only includes multi-choice datasets that naturally create preference pairs

# Enable this for debugging CUDA errors if needed
# export CUDA_LAUNCH_BLOCKING=1

MODEL="google/gemma-3-270m-it"

echo "🚀 Running CONFIDENCE-FILTERED ICM for knowledge elicitation..."
echo "Model: $MODEL"
echo "Total configurations: 6 (filtered for DPO compatibility)"
echo "OPTIMIZED: alpha=200.0, temp=15.0→0.0001, gen_temp=0.8, K=75, conf=0.05"
echo "Key insight: Fixed confidence calculation, only label examples with 5%+ confidence"

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
echo "1/6: HellaSwag (✓ 4 endings per context)..."
echo "    Confidence filtering: Only label high-confidence common sense predictions"
python -m icm.cli run --model $MODEL \
    --dataset Rowan/hellaswag \
    --task-type hellaswag \
    --alpha 200.0 \
    --initial-temperature 15.0 \
    --final-temperature 0.0001 \
    --generation-temperature 0.8 \
    --initial-examples 75 \
    --max-examples 600 \
    --max-iterations 1500 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.05 \
    --log-level INFO \
    $DEVICE_ARG

echo ""
echo "2/6: PIQA (✓ 2 solutions per goal)..."
echo "    Confidence filtering: Only label high-confidence physical reasoning"
python -m icm.cli run --model $MODEL \
    --dataset piqa \
    --task-type piqa \
    --alpha 200.0 \
    --initial-temperature 15.0 \
    --final-temperature 0.0001 \
    --generation-temperature 0.8 \
    --initial-examples 75 \
    --max-examples 600 \
    --max-iterations 1500 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.05 \
    --log-level INFO \
    $DEVICE_ARG

echo ""
echo "3/6: ARC - Both configs (✓ 4 choices per question)..."
echo "    Confidence filtering: Only label high-confidence science knowledge"
echo "  ARC-Challenge..."
python -m icm.cli run --model $MODEL \
    --dataset allenai/ai2_arc \
    --task-type arc_challenge \
    --config ARC-Challenge \
    --alpha 200.0 \
    --initial-temperature 15.0 \
    --final-temperature 0.0001 \
    --generation-temperature 0.8 \
    --initial-examples 75 \
    --max-examples 500 \
    --max-iterations 1500 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.05 \
    --log-level INFO \
    $DEVICE_ARG

echo "  ARC-Easy..."
python -m icm.cli run --model $MODEL \
    --dataset allenai/ai2_arc \
    --task-type arc_challenge \
    --config ARC-Easy \
    --alpha 200.0 \
    --initial-temperature 15.0 \
    --final-temperature 0.0001 \
    --generation-temperature 0.8 \
    --initial-examples 75 \
    --max-examples 500 \
    --max-iterations 1500 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.05 \
    --log-level INFO \
    $DEVICE_ARG

echo ""
echo "4/6: WinoGrande (✓ 2 options per sentence)..."
echo "    Confidence filtering: Only label high-confidence pronoun resolution"
python -m icm.cli run --model $MODEL \
    --dataset allenai/winogrande \
    --task-type winogrande \
    --alpha 200.0 \
    --initial-temperature 15.0 \
    --final-temperature 0.0001 \
    --generation-temperature 0.8 \
    --initial-examples 75 \
    --max-examples 600 \
    --max-iterations 1500 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.05 \
    --log-level INFO \
    $DEVICE_ARG

echo ""
echo "5/6: TruthfulQA multiple_choice (✓ Multiple answer choices)..."
echo "    Confidence filtering: Only label high-confidence truth detection"
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
    --max-iterations 1500 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.05 \
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
    echo "Expected range: 300-1,500 pairs (only from multi-choice datasets)"
    echo "Sample DPO pair:"
    head -1 gemma3_dpo_ready.jsonl | python -m json.tool
fi

echo ""
echo "✅ COMPLETE! OPTIMIZED ICM knowledge elicitation finished!"
echo ""
echo "🧠 Confidence-Filtered Knowledge Elicitation Summary:"
echo "  ✅ HellaSwag: Common sense (α=200, K=75, conf=0.15, max=600)"
echo "  ✅ PIQA: Physical reasoning (α=200, K=75, conf=0.15, max=600)"
echo "  ✅ ARC: Science knowledge (α=200, K=75, conf=0.15, max=500)"
echo "  ✅ WinoGrande: Logical reasoning (α=200, K=75, conf=0.15, max=600)"
echo "  ✅ TruthfulQA: Factual accuracy (α=200, K=75, conf=0.15, max=400)"
echo "  Total: 6 configurations with confidence-filtered labeling"
echo ""
echo "🔍 Confidence Filtering Benefits:"
echo "  • Previous: 8.5% accuracy (no filtering, all examples labeled)"
echo "  • Expected: 50-70% accuracy (only confident examples labeled)"
echo "  • Key: Quality over quantity - skip uncertain predictions"
echo "  • Result: Fewer but much more accurate labels for DPO"
echo ""
echo "EXCLUDED (no DPO pairs possible):"
echo "  ❌ GSM8K (single solution per question)"
echo "  ❌ BigBench Hard (single answer per task)"
echo "  ❌ IFEval (single instruction per example)"
echo "  ❌ TruthfulQA generation (single generation task)"
echo ""
echo "Next steps:"
echo "1. Validate ICM results - check if accuracy improved from ~8.5% to 40%+"
echo "2. If validation passes, fine-tune Gemma 3 270M-IT with DPO"
echo "3. Benefits of optimized ICM approach:"
echo "   🧠 Elicits latent knowledge through mutual consistency"
echo "   🔄 4x higher alpha prioritizes correct patterns"
echo "   🌡️  Higher temperatures enable exploration"
echo "   📊 3x more context improves pattern discovery"
echo "   ⏱️  3x more iterations ensure convergence"
echo "   ✅ No external supervision required"