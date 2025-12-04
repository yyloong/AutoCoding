import os
import json
import traceback
import shutil
import asyncio
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from datasets import load_dataset, load_from_disk
import logging
import dotenv

from ms_agent import LLMAgent
from ms_agent.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

dotenv.load_dotenv()


def _extract_and_save_code_blocks(prompt: str, base_dir: Path) -> None:
    """
    从 prompt 文本中提取 <code>...</code> 段落里的代码块，
    对于形如:
      [start of astropy/modeling/separable.py]
      1 line...
      ...
      [end of astropy/modeling/separable.py]
    的内容：
      - 去掉每行开头的行号和一个空格
      - 保存为 base_dir/astropy/modeling/separable.py
    """
    in_code_section = False
    in_file_block = False
    current_path = None
    current_lines = []

    lines = prompt.splitlines()

    for line in lines:
        stripped = line.strip()
        if stripped == "<code>":
            in_code_section = True
            continue
        if stripped == "</code>":
            # 结束所有文件块
            if in_file_block and current_path is not None:
                target = base_dir / current_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("".join(current_lines), encoding="utf-8")
            break

        if not in_code_section:
            continue

        if stripped.startswith("[start of ") and stripped.endswith("]"):
            # 若已有未关闭的块，先写入
            if in_file_block and current_path is not None:
                target = base_dir / current_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("".join(current_lines), encoding="utf-8")
            in_file_block = True
            current_lines = []
            # 解析路径
            inner = stripped[len("[start of ") : -1].strip()
            current_path = inner
            continue

        if stripped.startswith("[end of ") and stripped.endswith("]"):
            # 结束当前文件块并写入
            if in_file_block and current_path is not None:
                target = base_dir / current_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("".join(current_lines), encoding="utf-8")
            in_file_block = False
            current_path = None
            current_lines = []
            continue

        if in_file_block and current_path is not None:
            # 去掉行号：形如 "1 xxx" -> "xxx"
            # 简单规则：如果行首是整数+空格，则去掉这一段
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                content = parts[1]
            else:
                content = line

            # 统一去掉末尾换行，再手动补一个，这样每行恰好一个 '\n'
            content = content.rstrip("\n")
            current_lines.append(content + "\n")


async def run_agent_inference(query, api_key):
    """
    调用 Agent 执行任务
    """
    # 在调用 Agent 之前，先根据 prompt 把代码文件写入到 ./output 目录
    output_root = Path("output")
    output_root.mkdir(exist_ok=True, parents=True)
    _extract_and_save_code_blocks(query, output_root)

    config = Config.from_task('simple')
    config.llm.modelscope_api_key = api_key
    engine = LLMAgent(config=config)

    full_response = ""
    try:
        # 运行 Agent
        generator = await engine.run(query, stream=True)
        async for response_message in generator:
            if response_message and len(response_message) > 0:
                full_response = response_message[-1].content
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        traceback.print_exc()
        full_response += f"\nError during execution: {str(e)}"
    
    return full_response

def cleanup_environment():
    """
    清空 output 文件夹，并删除 memory 文件夹
    """
    # 清空 output 文件夹
    output_path = Path("output")
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(exist_ok=True)
    
    # 删除 memory 文件夹
    memory_path = Path("memory")
    if memory_path.exists():
        shutil.rmtree(memory_path)

async def process_dataset(
    dataset,
    output_file,
    existing_ids,
    api_key,
):
    # 确保 output 目录存在，供 Agent 使用
    Path("output").mkdir(exist_ok=True, parents=True)

    with open(output_file, "a+") as f:
        for datum in tqdm(dataset, desc=f"Inference with Agent"):
            instance_id = datum["instance_id"]
            if instance_id in existing_ids:
                continue

            # 用 while True 保证 patch 非空才跳出
            while True:
                cleanup_environment()
                output_dict = {"instance_id": instance_id}
                output_dict["model_name_or_path"] = "qwen3-coder-flash"

                prompt = datum.get("text", "")
                if not prompt:
                    for k, v in datum.items():
                        if isinstance(v, str):
                            prompt = v
                            break
                output_dict["text"] = prompt

                logger.info(f"Processing instance: {instance_id}")

                full_output = await run_agent_inference(prompt, api_key)
                output_dict["full_output"] = full_output

                patch_file = Path("output").rglob("fix.patch")
                patch_file = next(patch_file, None)
                model_patch = None
                if patch_file and patch_file.exists():
                    try:
                        model_patch = patch_file.read_text(encoding='utf-8')
                        logger.info(f"Found patch for {instance_id}")
                    except Exception as e:
                        logger.error(f"Failed to read patch file: {e}")
                else:
                    logger.warning(f"No patch file found for {instance_id}")

                # 检查 patch 是否为空，如果为空则重新跑本轮
                if not model_patch or len(model_patch.strip()) == 0:
                    logger.warning(f"Patch file for {instance_id} is empty, retrying this instance.")
                    continue  # 重新跑本轮

                output_dict["model_patch"] = model_patch
                print(json.dumps(output_dict, ensure_ascii=False), file=f, flush=True)
                cleanup_environment()
                break  # patch 非空，跳出 while，进入下一个 datum

def main(
    dataset_name_or_path,
    split,
    shard_id,
    num_shards,
    output_dir,
    max_instances,
    api_key,
):
    model_nickname = "qwen3-coder-flash"
    output_file = f"{model_nickname}__{dataset_name_or_path.split('/')[-1]}__{split}"
    if shard_id is not None and num_shards is not None:
        output_file += f"__shard-{shard_id}__num_shards-{num_shards}"
    output_file = Path(output_dir, output_file + ".jsonl")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"Will write to {output_file}")
    
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    instance_id = data["instance_id"]
                    existing_ids.add(instance_id)
                except json.JSONDecodeError:
                    pass
    logger.info(f"Read {len(existing_ids)} already completed ids from {output_file}")
    
    if Path(dataset_name_or_path).exists():
        dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path)
    
    if split in dataset:
        dataset = dataset[split]
        
    # 自动过滤超长样本 (参考 run_qwen.py)
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
        
    if max_instances is not None and len(dataset) > max_instances:
        dataset = dataset.select(range(max_instances))
        logger.info(f"Limited to first {max_instances} instances")

    # 运行异步处理循环
    asyncio.run(
        process_dataset(
            dataset=dataset,
            output_file=output_file,
            existing_ids=existing_ids,
            api_key=api_key,
        )
    )
    logger.info("Done!")

if __name__ == "__main__":
    api_key = os.getenv("MODELSCOPE_API_KEY", "")
    main(
        dataset_name_or_path="/home/u-wuhc/AutoCoding/SWE-bench/base_datasets/SWE-bench-Verified-oracle",
        split="test",
        shard_id=None,
        num_shards=None,
        output_dir="my_output",
        max_instances=100,
        api_key=api_key,
    )