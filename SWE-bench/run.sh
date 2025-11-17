python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --num_workers 4 \
    --predictions_path ./tmp

python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path results/predictions.jsonl \
    --max_workers 8 \
    --run_id my_evaluation_run