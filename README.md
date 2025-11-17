### Set up

建议 conda python 3.10 环境
```bash
pip install -e .
```

### Set up docker
在电脑上装好 docker

docker 测试命令
```bash
docker run hello-world
```
### 下载数据集
将数据集下载到本地 huggingface cache 中
```bash
cd SWE-bench
python download.py
```

### 测试 docker 是否配置正确
如果该问题能验证通过，则说明 docker 环境配置正确
```bash
python -m swebench.harness.run_evaluation \
    --max_workers 1 \
    --instance_ids sympy__sympy-20590 \
    --predictions_path gold \
    --run_id validate-gold
```

### 运行评测
用我处理好的数据 `SWE-bench/base_datasets/SWE-bench-Verified-oracle` 运行推理脚本

`run_qwen.py` 用的是 qwen3-coder-flash，你们可以换成自己的模型推理脚本
```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```
```bash
python run_qwen.py \
  --dataset_name_or_path /home/u-wuhc/AutoCoding/SWE-bench/base_datasets/SWE-bench-Verified-oracle \
  --model_name_or_path qwen3-coder-flash \
  --output_dir ./outputs \
  --model_args "temperature=0.2,top_p=0.95"
```

### 评测结果
单个样例的评估
```bash
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --max_workers 1 \
    --predictions_path SWE-bench/outputs/qwen3-coder-flash__SWE-bench-Verified-oracle__test.jsonl \
    --instance_ids django__django-11099 \
    --run_id single-instance-eval
```
总体评测
```bash
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --max_workers 1 \
    --predictions_path SWE-bench/outputs/qwen3-coder-flash__SWE-bench-Verified-oracle__test.jsonl \
    --run_id full-eval
```