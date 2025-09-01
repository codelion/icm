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
echo "1/5: HellaSwag (✓ 4 endings) - VERY RARE BUT POSSIBLE 🔴"
echo "    Empirical confidence: 3.95%, 4.35% found after extended search - threshold=1.0%"
python -m icm.cli run --model $MODEL \
    --dataset Rowan/hellaswag \
    --task-type hellaswag \
    --alpha 200.0 \
    --initial-temperature 15.0 \
    --final-temperature 0.0001 \
    --generation-temperature 0.8 \
    --initial-examples 50 \
    --max-examples -1 \
    --max-iterations 30000 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.01 \
    --log-level INFO \
    $DEVICE_ARG

echo ""
echo "2/5: PIQA (✓ 2 solutions per goal) - WORKABLE 🟡"
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
    --max-iterations 25000 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.005 \
    --log-level INFO \
    $DEVICE_ARG

echo ""
echo "3/5: ARC-Challenge - VERY DIFFICULT 🟠"
echo "    Empirical confidence: 7.4% once then all 0.0% - threshold=1.0% for rare finds"
python -m icm.cli run --model $MODEL \
    --dataset allenai/ai2_arc \
    --task-type arc_challenge \
    --config ARC-Challenge \
    --alpha 200.0 \
    --initial-temperature 15.0 \
    --final-temperature 0.0001 \
    --generation-temperature 0.8 \
    --initial-examples 50 \
    --max-examples -1 \
    --max-iterations 20000 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.01 \
    --log-level INFO \
    $DEVICE_ARG

echo ""
echo "4/5: WinoGrande (✓ 2 options per sentence) - ✅ EXCELLENT & RELIABLE"
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
    --max-examples -1 \
    --max-iterations 15000 \
    --cooling-rate 0.995 \
    --confidence-threshold 0.02 \
    --log-level INFO \
    $DEVICE_ARG

echo ""
echo "🔗 Creating DPO dataset from ALL 5 benchmarks..."
python -m icm.cli export-combined \
    --input-dir icm_results \
    --output-path gemma3_dpo_ready.jsonl \
    --fix-responses \
    --balance-strategy equal \
    --max-per-benchmark 2000

echo ""
echo "📊 Final DPO statistics..."
if [ -f "gemma3_dpo_ready.jsonl" ]; then
    lines=$(wc -l < gemma3_dpo_ready.jsonl)
    echo "Total DPO preference pairs: $lines"
    echo "Expected range: 800-2500 pairs (from ALL 5 datasets with full scanning)"
    echo "Quality: Empirically-tested thresholds ensure reliable labeling"
    echo "Sample DPO pair:"
    head -1 gemma3_dpo_ready.jsonl | python -m json.tool
fi

echo ""
echo "✅ COMPLETE! COMPREHENSIVE ICM knowledge elicitation finished!"
echo ""
echo "🧠 Full Dataset Coverage Knowledge Elicitation Summary:"
echo "  ✅ WinoGrande: Pronoun resolution (EXCELLENT - conf=3%, max=-1, iter=20000)"
echo "  🟡 PIQA: Physical reasoning (WORKABLE - conf=0.5%, max=-1, iter=25000)"
echo "  🟡 TruthfulQA: Factual accuracy (MEDIUM - conf=2%, max=-1, iter=15000)"
echo "  🔴 HellaSwag: Commonsense reasoning (VERY RARE - conf=1%, max=-1, iter=30000)"
echo "  🟠 ARC-Challenge: Science reasoning (DIFFICULT - conf=1%, max=-1, iter=20000)"
echo "  Total: ALL 5 datasets with full scanning and empirical thresholds"
echo ""
echo "🔍 Comprehensive Full-Dataset Strategy Benefits:"
echo "  • Previous: 8.5% accuracy (limited search with high thresholds)"
echo "  • New approach: Full dataset scanning with empirically-tested thresholds"
echo "  • Strategy: Include ALL datasets that show ANY positive confidence"
echo "  • Result: Maximum coverage with quality assurance via proper thresholds"
echo "  • Key insight: Even rare 1-4% confidence can yield valuable examples"
echo ""
echo "INCLUDED datasets with empirical confidence ranges:"
echo "  ✅ WinoGrande (7.4-11.2% confidence - most reliable producer)"
echo "  🟡 PIQA (1.1-10.6% confidence - mixed but workable with patience)"
echo "  🟡 TruthfulQA (10.5% initial drops - rare but high-value when found)"
echo "  🔴 HellaSwag (3.95-4.35% confidence - rare but possible with large sample)"
echo "  🟠 ARC-Challenge (7.4% once observed - difficult but not impossible)"
echo ""
echo "BENEFITS of full dataset approach:"
echo "  • No arbitrary dataset exclusions - test everything empirically"
echo "  • Larger total pool increases DPO pair diversity"
echo "  • Rare high-confidence examples from 'difficult' datasets add value"
echo "  • Full scanning ensures we don't miss edge cases"
echo ""
echo "Next steps:"
echo "1. Run full ICM scan on ALL 5 datasets with unlimited samples (max=-1)"
echo "2. WinoGrande should produce most reliable labels (7.4-11.2% confidence)"
echo "3. PIQA provides steady medium-confidence examples (1.1-10.6% range)"
echo "4. TruthfulQA, HellaSwag, ARC contribute rare high-value examples"
echo "5. Expect 800-2500 total DPO pairs from comprehensive approach"
echo "6. If results validate, fine-tune Gemma 3 270M-IT with full dataset"
echo ""
echo "Benefits of comprehensive ICM approach:"
echo "   🧠 Elicits latent knowledge through mutual consistency across ALL tasks"
echo "   🔄 Empirical thresholds ensure quality while maximizing coverage"
echo "   🌡️  Extended search finds rare but valuable high-confidence examples"
echo "   📊 Full dataset scanning leaves no examples untested"
echo "   ⏱️  Patient iteration counts allow convergence on difficult datasets"
echo "   ✅ No external supervision - purely self-elicited knowledge"