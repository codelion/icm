#!/bin/bash
# OPTIMIZED ICM script for eliciting latent knowledge from Gemma 3 270M
# Parameters tuned specifically for better mutual predictability and knowledge elicitation
# Only includes multi-choice datasets that naturally create preference pairs

# Enable this for debugging CUDA errors if needed
# export CUDA_LAUNCH_BLOCKING=1

MODEL="google/gemma-3-270m-it"

echo "🎯 Running PROVEN ICM on reliable datasets only..."
echo "Model: $MODEL"
echo "Strategy: Empirically-tested producers - focus on 3 datasets with consistent results"
echo "OPTIMIZED: alpha=200.0, temp=15.0→0.0001, gen_temp=0.8"
echo "Excluded datasets with 0% confidence after extensive testing"

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
echo "1/3: PIQA (✓ 2 solutions per goal) - GOOD PRODUCER 🟡"
echo "    Empirical confidence: 1.1%, 9.6%, 10.6% mixed - threshold=0.5%"
python -m icm.cli run --model $MODEL \
    --dataset piqa \
    --task-type piqa \
    --alpha 200.0 \
    --initial-temperature 15.0 \
    --final-temperature 0.0001 \
    --generation-temperature 0.8 \
    --initial-examples 50 \
    --max-examples -1 \
    --max-iterations 20000 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.005 \
    --log-level INFO \
    $DEVICE_ARG

echo ""
echo "2/3: WinoGrande (✓ 2 options per sentence) - ✅ MOST RELIABLE"
echo "    Empirical confidence: 7.4%, 10.4%, 10.9%, 11.2% consistently - threshold=3.0%"
python -m icm.cli run --model $MODEL \
    --dataset allenai/winogrande \
    --task-type winogrande \
    --alpha 200.0 \
    --initial-temperature 15.0 \
    --final-temperature 0.0001 \
    --generation-temperature 0.8 \
    --initial-examples 75 \
    --max-examples -1 \
    --max-iterations 20000 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.03 \
    --log-level INFO \
    $DEVICE_ARG

echo ""
echo "3/3: TruthfulQA multiple_choice - MODERATE PRODUCER 🟡"
echo "    Observed confidence: 10.5% then occasional finds, threshold=2% for high-value examples"
python -m icm.cli run --model $MODEL \
    --dataset truthful_qa \
    --task-type truthfulqa \
    --config multiple_choice \
    --alpha 200.0 \
    --initial-temperature 15.0 \
    --final-temperature 0.0001 \
    --generation-temperature 0.8 \
    --initial-examples 75 \
    --max-examples -1 \
    --max-iterations 10000 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.02 \
    --log-level INFO \
    $DEVICE_ARG

echo ""
echo "🔗 Creating DPO dataset from 3 proven benchmarks..."
python -m icm.cli export-combined \
    --input-dir icm_results \
    --output-path gemma3_dpo_ready.jsonl \
    --fix-responses \
    --balance-strategy equal \
    --max-per-benchmark 1500

echo ""
echo "📊 Final DPO statistics..."
if [ -f "gemma3_dpo_ready.jsonl" ]; then
    lines=$(wc -l < gemma3_dpo_ready.jsonl)
    echo "Total DPO preference pairs: $lines"
    echo "Expected range: 500-1500 pairs (from 3 proven datasets)"
    echo "Quality: High-confidence examples from empirically-tested producers"
    echo "Sample DPO pair:"
    head -1 gemma3_dpo_ready.jsonl | python -m json.tool
fi

echo ""
echo "✅ COMPLETE! FOCUSED ICM knowledge elicitation finished!"
echo ""
echo "🧠 Proven Producer Knowledge Elicitation Summary:"
echo "  🟡 PIQA: Physical reasoning (GOOD - conf=0.5%, max=-1, iter=20000)"
echo "  ✅ WinoGrande: Pronoun resolution (EXCELLENT - conf=3%, max=-1, iter=20000)"
echo "  🟡 TruthfulQA: Factual accuracy (MODERATE - conf=2%, max=-1, iter=10000)"
echo "  Total: 3 datasets with proven positive confidence rates"
echo ""
echo "🔍 Focused Producer Strategy Benefits:"
echo "  • Previous: 8.5% accuracy (included datasets with 0% confidence)"
echo "  • New approach: Only proven producers with consistent results"
echo "  • Strategy: Exclude datasets showing 0% confidence after extensive testing"
echo "  • Result: Higher quality DPO data with efficient compute usage"
echo "  • Key insight: Better to have 500 good pairs than mixed quality results"
echo ""
echo "INCLUDED datasets with proven confidence ranges:"
echo "  ✅ WinoGrande (7.4-11.2% confidence - most reliable producer)"
echo "  🟡 PIQA (1.1-10.6% confidence - consistent medium producer)"
echo "  🟡 TruthfulQA (10.5% initial - occasional high-value finds)"
echo ""
echo "EXCLUDED datasets (0% confidence after extensive testing):"
echo "  ❌ HellaSwag (0% confidence after 1166+ iterations - compute waste)"
echo "  ❌ ARC-Challenge (single 7.4% then all 0% - unreliable)"
echo ""
echo "BENEFITS of focused approach:"
echo "  • No compute waste on non-producing datasets"
echo "  • Faster completion time (1-2 days vs 7+ days)"
echo "  • Higher confidence in labeling accuracy"
echo "  • Still provides good task diversity across 3 different types"
echo ""
echo "Next steps:"
echo "1. Run ICM on 3 proven datasets with full scanning (max=-1)"
echo "2. WinoGrande should produce most reliable labels (7.4-11.2% confidence)"
echo "3. PIQA provides steady medium-confidence examples (1.1-10.6% range)"
echo "4. TruthfulQA contributes occasional high-value examples (10.5% initial)"
echo "5. Expect 500-1500 total DPO pairs from focused approach"
echo "6. Complete in 1-2 days instead of 7+ days with better quality"
echo "7. If results validate, fine-tune Gemma 3 270M-IT with high-quality dataset"
echo ""
echo "Benefits of focused ICM approach:"
echo "   🧠 Elicits latent knowledge from proven reliable sources only"
echo "   🔄 Empirical testing eliminates non-producing datasets upfront"
echo "   ⚡ Faster completion with higher confidence in results"
echo "   📊 Quality over quantity - better DPO training data"
echo "   💰 Efficient compute usage - no wasted iterations on 0% datasets"
echo "   ✅ No external supervision - purely self-elicited knowledge"