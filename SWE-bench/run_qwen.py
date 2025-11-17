import os
import json
import time
import traceback
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from datasets import load_dataset, load_from_disk
from swebench.inference.make_datasets.utils import extract_diff
from argparse import ArgumentParser
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Qwen3 coder flash最大上下文长度
MODEL_LIMITS = {
    "qwen3-coder-flash": 32768,
    "qwen3-coder": 32768,
    "qwen3": 32768,
}

def qwen_inference(
    test_dataset,
    model_name_or_path,
    output_file,
    model_args,
    existing_ids,
    max_cost,
    api_key,
    base_url,
):
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请先 pip install openai>=1.0.0")
    client = OpenAI(api_key=api_key, base_url=base_url)
    temperature = float(model_args.get("temperature", 0.2))
    top_p = float(model_args.get("top_p", 0.95 if temperature > 0 else 1))
    print(f"Using temperature={temperature}, top_p={top_p}")
    basic_args = {
        "model_name_or_path": model_name_or_path,
    }
    total_cost = 0
    print(f"Filtered to {len(test_dataset)} instances")
    with open(output_file, "a+") as f:
        for datum in tqdm(test_dataset, desc=f"Inference for {model_name_or_path}"):
            instance_id = datum["instance_id"]
            if instance_id in existing_ids:
                continue
            output_dict = {"instance_id": instance_id}
            output_dict.update(basic_args)
            # 兼容 SWE-bench 格式，假定输入字段为 text
            prompt = datum.get("text", "")
            if not prompt:
                # 自动找第一个字符串字段
                for k, v in datum.items():
                    if isinstance(v, str):
                        prompt = v
                        break
            output_dict["text"] = prompt
            try:
                completion = client.chat.completions.create(
                    model=model_name_or_path,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                )
                reply = completion.choices[0].message.content
            except Exception as e:
                logger.error(e)
                traceback.print_exc()
                time.sleep(10)
                continue
            output_dict["full_output"] = reply
            output_dict["model_patch"] = extract_diff(reply)
            print(json.dumps(output_dict, ensure_ascii=False), file=f, flush=True)
            # Qwen暂不计费，如需计费可在此处加统计
            if max_cost is not None and total_cost >= max_cost:
                print(f"Reached max cost {max_cost}, exiting")
                break

def parse_model_args(model_args):
    kwargs = dict()
    if model_args is not None:
        for arg in model_args.split(","):
            key, value = arg.split("=")
            if value in {"True", "False"}:
                kwargs[key] = value == "True"
            elif value.isnumeric():
                kwargs[key] = int(value)
            elif value.replace(".", "", 1).isnumeric():
                kwargs[key] = float(value)
            elif value in {"None"}:
                kwargs[key] = None
            elif value in {"[]"}:
                kwargs[key] = []
            elif value in {"{}"}:
                kwargs[key] = {}
            elif value.startswith("'") and value.endswith("'"):
                kwargs[key] = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                kwargs[key] = value[1:-1]
            else:
                kwargs[key] = value
    return kwargs

def main(
    dataset_name_or_path,
    split,
    model_name_or_path,
    shard_id,
    num_shards,
    output_dir,
    model_args,
    max_cost,
    api_key,
    base_url,
):
    model_args = parse_model_args(model_args)
    model_nickname = model_name_or_path
    output_file = f"{model_nickname}__{dataset_name_or_path.split('/')[-1]}__{split}"
    if shard_id is not None and num_shards is not None:
        output_file += f"__shard-{shard_id}__num_shards-{num_shards}"
    output_file = Path(output_dir, output_file + ".jsonl")
    logger.info(f"Will write to {output_file}")
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                instance_id = data["instance_id"]
                existing_ids.add(instance_id)
    logger.info(f"Read {len(existing_ids)} already completed ids from {output_file}")
    if Path(dataset_name_or_path).exists():
        dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path)
    if split not in dataset:
        raise ValueError(f"Invalid split {split} for dataset {dataset_name_or_path}")
    dataset = dataset[split]
    # 自动过滤超长样本
    lens = np.array([len(str(d.get("text", ""))) for d in dataset])
    dataset = dataset.select(np.argsort(lens))
    if len(existing_ids) > 0:
        dataset = dataset.filter(
            lambda x: x["instance_id"] not in existing_ids,
            desc="Filtering out existing ids",
            load_from_cache_file=False,
        )
    if shard_id is not None and num_shards is not None:
        dataset = dataset.shard(num_shards, shard_id, contiguous=True)
    inference_args = {
        "test_dataset": dataset,
        "model_name_or_path": model_name_or_path,
        "output_file": output_file,
        "model_args": model_args,
        "existing_ids": existing_ids,
        "max_cost": max_cost,
        "api_key": api_key,
        "base_url": base_url,
    }
    qwen_inference(**inference_args)
    logger.info("Done!")

if __name__ == "__main__":
    parser = ArgumentParser(description="Qwen API inference for SWE-bench format")
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="HuggingFace dataset name or local path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        choices=sorted(list(MODEL_LIMITS.keys())),
        help="Qwen model name",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=None,
        help="Shard id to process. If None, process all shards.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Number of shards. If None, process all shards.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output file.",
    )
    parser.add_argument(
        "--model_args",
        type=str,
        default=None,
        help="List of model arguments separated by commas. (e.g. 'top_p=0.95,temperature=0.70')",
    )
    parser.add_argument(
        "--max_cost",
        type=float,
        default=None,
        help="Maximum cost to spend on inference.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("DASHSCOPE_API_KEY", None),
        help="Qwen API key, or set DASHSCOPE_API_KEY env variable.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="Qwen API base url.",
    )
    args = parser.parse_args()
    if not args.api_key:
        raise ValueError("请通过 --api_key 或 DASHSCOPE_API_KEY 环境变量提供 Qwen API Key")
    main(**vars(args))