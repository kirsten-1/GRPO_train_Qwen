import os
from datasets import load_dataset
from tqdm import tqdm
import time

# 设置缓存目录
CACHE_DIR = "/root/autodl-tmp/datasets"

# 确保目录存在
os.makedirs(CACHE_DIR, exist_ok=True)

# 定义要下载的数据集
datasets_to_download = [
    {"name": "GSM8K", "path": "openai/gsm8k", "config": "main", "desc": "小学数学应用题", "source": "huggingface"},
    {"name": "MATH", "path": "EleutherAI/hendrycks_math", "config": "algebra", "desc": "高中竞赛级别数学(代数)", "source": "huggingface"},
    {"name": "CMATH", "path": "weitianwen/cmath", "config": None, "desc": "中文高中数学", "source": "huggingface"},
    {"name": "MBPP", "path": "google-research-datasets/mbpp", "config": "sanitized", "desc": "Python代码生成-基础", "source": "huggingface"},
    {"name": "HumanEval", "path": "openai/openai_humaneval", "config": None, "desc": "Python代码生成-标准benchmark", "source": "huggingface"},
]

# 存储下载的数据集
downloaded_datasets = {}

print("=" * 70)
print(f"开始下载数据集到: {CACHE_DIR}")
print("=" * 70)

# 使用进度条显示总体进度
for idx, dataset_info in enumerate(tqdm(datasets_to_download, desc="总体进度", unit="dataset")):
    name = dataset_info["name"]
    path = dataset_info.get("path")
    config = dataset_info.get("config")
    desc = dataset_info["desc"]
    source = dataset_info.get("source", "huggingface")

    print(f"\n[{idx+1}/{len(datasets_to_download)}] 正在下载: {name} ({desc})")
    if path:
        print(f"    数据集路径: {path}")
    print(f"    数据源: {source.upper()}")

    try:
        start_time = time.time()

        # 使用 Hugging Face 下载
        kwargs = {"cache_dir": CACHE_DIR}
        if config:
            kwargs["name"] = config

        dataset = load_dataset(path, **kwargs)

        elapsed_time = time.time() - start_time
        downloaded_datasets[name] = dataset

        print(f"    ✓ 下载完成 (耗时: {elapsed_time:.2f}秒)")

    except Exception as e:
        print(f"    ✗ 下载失败: {str(e)}")
        downloaded_datasets[name] = None

# 显示下载结果摘要
print("\n" + "=" * 70)
print("下载结果摘要:")
print("=" * 70)

for name, dataset in downloaded_datasets.items():
    if dataset is not None:
        print(f"\n{name}:")
        print(f"  {dataset}")
    else:
        print(f"\n{name}: 下载失败")

print("\n" + "=" * 70)
print(f"所有数据集已保存到: {CACHE_DIR}")
print("=" * 70)