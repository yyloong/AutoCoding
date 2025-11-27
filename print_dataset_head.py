"""
打印数据集 text 字段的前 3 个样本
"""

import os
from pathlib import Path
from datasets import load_dataset, load_from_disk

def main():
    # 数据集路径
    dataset_path = "SWE-bench/base_datasets/SWE-bench-Verified-oracle"

    print("正在加载数据集...")
    try:
        if Path(dataset_path).exists():
            dataset = load_from_disk(dataset_path)
        else:
            dataset = load_dataset(dataset_path)

        # 获取 test split（默认 split）
        if "test" in dataset:
            test_dataset = dataset["test"]
        else:
            # 如果没有 test split，使用第一个可用的 split
            test_dataset = dataset[list(dataset.keys())[0]]

        print(f"数据集大小: {len(test_dataset)}")
        print(f"数据集字段: {test_dataset.column_names}")
        print("\n前 3 个样本的 text 字段:")
        print("=" * 50)

        for i in range(min(1, len(test_dataset))):
            sample = test_dataset[i]
            instance_id = sample.get("instance_id", f"sample_{i}")
            text = sample.get("text", "")

            print(f"\n--- 样本 {i+1} (ID: {instance_id}) ---")
            print(text)  # 只显示前500字符
            print("-" * 50)

    except Exception as e:
        print(f"加载数据集时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
