# 建议 conda python 3.10 环境
pip install -e .

# set up docker
# 在电脑上装好 docker
# 可能需要按照这个教程配置 nju docker mirror
# https://sci.nju.edu.cn/9e/05/c30384a564741/page.htm 
# docker 测试命令
# docker run hello-world

# 测试 docker 是否配置正确
# 如果该问题能验证通过，则说明 docker 环境配置正确
python -m swebench.harness.run_evaluation \
    --max_workers 1 \
    --instance_ids sympy__sympy-20590 \
    --predictions_path gold \
    --run_id validate-gold

# 本地运行 SWE-Llama 模型进行推理
# 这个脚本好像有问题，输出都是 <unk>
python -m swebench.inference.run_llama \
    --dataset_path princeton-nlp/SWE-bench_oracle \
    --model_name_or_path princeton-nlp/SWE-Llama-7b \
    --output_dir ./outputs
python -m swebench.inference.run_llama \
    --dataset_path princeton-nlp/SWE-bench_oracle \
    --model_name_or_path princeton-nlp/SWE-Llama-13b \
    --output_dir ./outputs

# qwen3-coder-flash 在 SWE-bench_oracle 上的推理
python run_qwen.py \
  --dataset_name_or_path princeton-nlp/SWE-bench_oracle \
  --model_name_or_path qwen3-coder-flash \
  --output_dir ./outputs \
  --model_args "temperature=0.2,top_p=0.95"

# 生成 SWE-bench_Verified-oracle 数据集
python -m swebench.inference.make_datasets.create_text_dataset \
    --dataset_name_or_path princeton-nlp/SWE-bench_Verified \
    --output_dir ./base_datasets \
    --split test \
    --prompt_style style-3 \
    --file_source oracle

# qwen3-coder-flash 在 SWE-bench_Verified-oracle 上的推理
python run_qwen.py \
  --dataset_name_or_path /home/u-wuhc/AutoCoding/SWE-bench/base_datasets/SWE-bench-Verified-oracle \
  --model_name_or_path qwen3-coder-flash \
  --output_dir ./outputs \
  --model_args "temperature=0.2,top_p=0.95"

# 运行单个样例的评测
# 这个结果不通过
python -m swebench.harness.run_evaluation \
    --max_workers 1 \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --instance_ids django__django-15127 \
    --predictions_path v.jsonl \
    --run_id validate-gold

# 这个结果是通过的
python -m swebench.harness.run_evaluation \
    --max_workers 1 \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --instance_ids django__django-11099 \
    --predictions_path v.jsonl \
    --run_id validate-gold
