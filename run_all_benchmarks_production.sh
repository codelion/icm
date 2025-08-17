#!/bin/bash
# Optimized production script to run ICM on ALL benchmark configurations
# Based on HuggingFace blog analysis with optimal parameters for maximum quality

MODEL="google/gemma-3-270m-it"

echo "🚀 Running OPTIMIZED ICM on ALL benchmark configurations for production..."
echo "Model: $MODEL"
echo "Total configurations: 41"
echo "Optimization: alpha=50.0, temp=8.0→0.001, gen_temp=0.3, K=50, iter=500"

# Clean previous results (optional)
# icm clean --keep-latest 0

echo ""
echo "1/8: HellaSwag (1 config)..."
icm run --model $MODEL \
    --dataset Rowan/hellaswag \
    --task-type hellaswag \
    --alpha 50.0 \
    --initial-temperature 8.0 \
    --generation-temperature 0.3 \
    --initial-examples 50 \
    --max-examples 1000 \
    --max-iterations 500

echo ""
echo "2/8: PIQA (1 config)..."
icm run --model $MODEL \
    --dataset piqa \
    --task-type piqa \
    --alpha 50.0 \
    --initial-temperature 8.0 \
    --generation-temperature 0.3 \
    --initial-examples 50 \
    --max-examples 1000 \
    --max-iterations 500

echo ""
echo "3/8: ARC - Both configs (2 total)..."
echo "  ARC-Challenge..."
icm run --model $MODEL \
    --dataset allenai/ai2_arc \
    --task-type arc_challenge \
    --config ARC-Challenge \
    --alpha 50.0 \
    --initial-temperature 8.0 \
    --generation-temperature 0.3 \
    --initial-examples 50 \
    --max-examples 800 \
    --max-iterations 500

echo "  ARC-Easy..."
icm run --model $MODEL \
    --dataset allenai/ai2_arc \
    --task-type arc_challenge \
    --config ARC-Easy \
    --alpha 50.0 \
    --initial-temperature 8.0 \
    --generation-temperature 0.3 \
    --initial-examples 50 \
    --max-examples 800 \
    --max-iterations 500

echo ""
echo "4/8: WinoGrande - All 5 size configs..."
for size in winogrande_xs winogrande_s winogrande_m winogrande_l winogrande_xl; do
    echo "  WinoGrande $size..."
    icm run --model $MODEL \
        --dataset allenai/winogrande \
        --task-type winogrande \
        --config $size \
        --alpha 50.0 \
        --initial-temperature 8.0 \
        --generation-temperature 0.3 \
        --initial-examples 50 \
        --max-examples 600 \
        --max-iterations 500
done

echo ""
echo "5/8: BIG-Bench Hard - All 27 tasks..."
BBH_TASKS=(
    "boolean_expressions" 
    "causal_judgement" 
    "date_understanding" 
    "disambiguation_qa" 
    "dyck_languages" 
    "formal_fallacies"
    "geometric_shapes" 
    "hyperbaton" 
    "logical_deduction_five_objects"
    "logical_deduction_seven_objects" 
    "logical_deduction_three_objects"
    "movie_recommendation" 
    "multistep_arithmetic_two" 
    "navigate"
    "object_counting" 
    "penguins_in_a_table" 
    "reasoning_about_colored_objects"
    "ruin_names" 
    "salient_translation_error_detection" 
    "snarks"
    "sports_understanding" 
    "temporal_sequences" 
    "tracking_shuffled_objects_five_objects"
    "tracking_shuffled_objects_seven_objects"
    "tracking_shuffled_objects_three_objects"
    "web_of_lies" 
    "word_sorting"
)

task_count=1
for task in "${BBH_TASKS[@]}"; do
    echo "  BBH ($task_count/27): $task..."
    icm run --model $MODEL \
        --dataset maveriq/bigbenchhard \
        --config $task \
        --task-type bigbench_hard \
        --alpha 50.0 \
        --initial-temperature 8.0 \
        --generation-temperature 0.3 \
        --initial-examples 50 \
        --max-examples 200 \
        --max-iterations 500
    ((task_count++))
done

echo ""
echo "6/8: IFEval (1 config)..."
icm run --model $MODEL \
    --dataset google/IFEval \
    --task-type ifeval \
    --alpha 50.0 \
    --initial-temperature 8.0 \
    --generation-temperature 0.3 \
    --initial-examples 50 \
    --max-examples 500 \
    --max-iterations 500

echo ""
echo "7/8: TruthfulQA - Both configs (2 total)..."
echo "  TruthfulQA multiple_choice..."
icm run --model $MODEL \
    --dataset truthful_qa \
    --task-type truthfulqa \
    --config multiple_choice \
    --alpha 50.0 \
    --initial-temperature 8.0 \
    --generation-temperature 0.3 \
    --initial-examples 50 \
    --max-examples 400 \
    --max-iterations 500

echo "  TruthfulQA generation..."
icm run --model $MODEL \
    --dataset truthful_qa \
    --task-type truthfulqa \
    --config generation \
    --alpha 50.0 \
    --initial-temperature 8.0 \
    --generation-temperature 0.3 \
    --initial-examples 50 \
    --max-examples 400 \
    --max-iterations 500

echo ""
echo "8/8: GSM8K - Both configs (2 total)..."
echo "  GSM8K main..."
icm run --model $MODEL \
    --dataset gsm8k \
    --task-type gsm8k \
    --config main \
    --alpha 50.0 \
    --initial-temperature 8.0 \
    --generation-temperature 0.3 \
    --initial-examples 50 \
    --max-examples 600 \
    --max-iterations 500

echo "  GSM8K socratic..."
icm run --model $MODEL \
    --dataset gsm8k \
    --task-type gsm8k \
    --config socratic \
    --alpha 50.0 \
    --initial-temperature 8.0 \
    --generation-temperature 0.3 \
    --initial-examples 50 \
    --max-examples 600 \
    --max-iterations 500

echo ""
echo "🔗 Combining all results into DPO dataset..."
icm export-combined \
    --input-dir icm_results \
    --output-path gemma3_complete_all_configs_dpo.jsonl

echo ""
echo "📊 Final statistics..."
if [ -f "gemma3_complete_all_configs_dpo.jsonl" ]; then
    lines=$(wc -l < gemma3_complete_all_configs_dpo.jsonl)
    echo "Total DPO preference pairs: $lines"
    echo "Expected range: 12,000-20,000 pairs (optimized parameters)"
    echo "Sample DPO pair:"
    head -1 gemma3_complete_all_configs_dpo.jsonl | jq .
fi

echo ""
echo "✅ COMPLETE! OPTIMIZED production run with ALL 41 configurations finished!"
echo ""
echo "Summary of coverage:"
echo "  - HellaSwag: 1 config"
echo "  - PIQA: 1 config"
echo "  - ARC: 2 configs (Challenge + Easy)"
echo "  - WinoGrande: 5 configs (xs, s, m, l, xl)"
echo "  - BIG-Bench Hard: 27 tasks"
echo "  - IFEval: 1 config"
echo "  - TruthfulQA: 2 configs (multiple_choice + generation)"
echo "  - GSM8K: 2 configs (main + socratic)"
echo "  Total: 41 different dataset configurations"
echo ""
echo "Next steps:"
echo "1. Fine-tune Gemma 3 270M-IT with DPO using: gemma3_complete_all_configs_dpo.jsonl"
echo "2. Evaluate improved model on all benchmarks"
echo "3. Compare improvements across all configurations"
echo "4. Expected improvements: 2-5% on each benchmark based on blog analysis"