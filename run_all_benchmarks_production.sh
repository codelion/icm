#!/bin/bash
# DPO-focused production script to run ICM on datasets that produce valid DPO pairs
# Only includes multi-choice datasets that naturally create preference pairs

# Enable this for debugging CUDA errors if needed
# export CUDA_LAUNCH_BLOCKING=1

MODEL="google/gemma-3-270m-it"

echo "🚀 Running ICM on DPO-COMPATIBLE benchmark configurations..."
echo "Model: $MODEL"
echo "Total configurations: 10 (filtered for DPO compatibility)"
echo "Optimization: alpha=50.0, temp=8.0→0.001, gen_temp=0.3, K=50, iter=500"

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
python -m icm.cli run --model $MODEL \
    --dataset Rowan/hellaswag \
    --task-type hellaswag \
    --alpha 50.0 \
    --initial-temperature 8.0 \
    --generation-temperature 0.3 \
    --initial-examples 50 \
    --max-examples 1000 \
    --max-iterations 500 \
    $DEVICE_ARG

echo ""
echo "2/6: PIQA (✓ 2 solutions per goal)..."
python -m icm.cli run --model $MODEL \
    --dataset piqa \
    --task-type piqa \
    --alpha 50.0 \
    --initial-temperature 8.0 \
    --generation-temperature 0.3 \
    --initial-examples 50 \
    --max-examples 1000 \
    --max-iterations 500 \
    $DEVICE_ARG

echo ""
echo "3/6: ARC - Both configs (✓ 4 choices per question)..."
echo "  ARC-Challenge..."
python -m icm.cli run --model $MODEL \
    --dataset allenai/ai2_arc \
    --task-type arc_challenge \
    --config ARC-Challenge \
    --alpha 50.0 \
    --initial-temperature 8.0 \
    --generation-temperature 0.3 \
    --initial-examples 50 \
    --max-examples 800 \
    --max-iterations 500 \
    $DEVICE_ARG

echo "  ARC-Easy..."
python -m icm.cli run --model $MODEL \
    --dataset allenai/ai2_arc \
    --task-type arc_challenge \
    --config ARC-Easy \
    --alpha 50.0 \
    --initial-temperature 8.0 \
    --generation-temperature 0.3 \
    --initial-examples 50 \
    --max-examples 800 \
    --max-iterations 500 \
    $DEVICE_ARG

echo ""
echo "4/6: WinoGrande - All 5 size configs (✓ 2 options per sentence)..."
for size in winogrande_xs winogrande_s winogrande_m winogrande_l winogrande_xl; do
    echo "  WinoGrande $size..."
    python -m icm.cli run --model $MODEL \
        --dataset allenai/winogrande \
        --task-type winogrande \
        --config $size \
        --alpha 50.0 \
        --initial-temperature 8.0 \
        --generation-temperature 0.3 \
        --initial-examples 50 \
        --max-examples 600 \
        --max-iterations 500 \
    $DEVICE_ARG
done

echo ""
echo "5/6: TruthfulQA multiple_choice (✓ Multiple answer choices)..."
python -m icm.cli run --model $MODEL \
    --dataset truthful_qa \
    --task-type truthfulqa \
    --config multiple_choice \
    --alpha 50.0 \
    --initial-temperature 8.0 \
    --generation-temperature 0.3 \
    --initial-examples 50 \
    --max-examples 400 \
    --max-iterations 500 \
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
    echo "Expected range: 500-2,000 pairs (only from multi-choice datasets)"
    echo "Sample DPO pair:"
    head -1 gemma3_dpo_ready.jsonl | python -m json.tool
fi

echo ""
echo "✅ COMPLETE! DPO-focused production run finished!"
echo ""
echo "Summary of DPO-compatible datasets processed:"
echo "  ✅ HellaSwag: 1 config (4 endings → preference pairs)"
echo "  ✅ PIQA: 1 config (2 solutions → preference pairs)"
echo "  ✅ ARC: 2 configs (4 choices → preference pairs)"
echo "  ✅ WinoGrande: 5 configs (2 options → preference pairs)"
echo "  ✅ TruthfulQA: 1 config (multiple choices → preference pairs)"
echo "  Total: 10 configurations producing valid DPO pairs"
echo ""
echo "EXCLUDED (no DPO pairs possible):"
echo "  ❌ GSM8K (single solution per question)"
echo "  ❌ BigBench Hard (single answer per task)"
echo "  ❌ IFEval (single instruction per example)"
echo "  ❌ TruthfulQA generation (single generation task)"
echo ""
echo "Next steps:"
echo "1. Fine-tune Gemma 3 270M-IT with DPO using: gemma3_dpo_ready.jsonl"
echo "2. Benefits of this approach:"
echo "   ✅ Only valid preference pairs (no empty chosen/rejected)"
echo "   ✅ True ICM-discovered preferences (no synthetic data)"
echo "   ✅ Efficient training with high-quality pairs"
echo "   ✅ Natural multi-choice structure preserved"