#!/usr/bin/env python3
"""
检查数据预处理环境是否就绪
"""
import os
from pathlib import Path

def check_file_exists(path, description):
    """检查文件是否存在"""
    if Path(path).exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description} 不存在: {path}")
        return False

def check_directory_exists(path, description):
    """检查目录是否存在"""
    if Path(path).is_dir():
        file_count = len(list(Path(path).rglob('*')))
        print(f"✅ {description}: {path} ({file_count} 个文件)")
        return True
    else:
        print(f"❌ {description} 不存在: {path}")
        return False

def main():
    print("="*60)
    print("检查数据预处理环境")
    print("="*60)
    print()

    all_ok = True

    # 检查模型
    print("📦 检查模型:")
    model_path = "/root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct"
    all_ok &= check_directory_exists(model_path, "Qwen2.5-3B-Instruct 模型")
    print()

    # 检查数据集
    print("📊 检查原始数据集:")
    datasets = [
        ("/root/autodl-tmp/datasets/ape210k-master/data", "Ape210K"),
        ("/root/autodl-tmp/datasets/openai___gsm8k/main", "GSM8K"),
        ("/root/autodl-tmp/datasets/EleutherAI___hendrycks_math/algebra", "MATH"),
        ("/root/autodl-tmp/datasets/weitianwen___cmath/default", "CMATH"),
        ("/root/autodl-tmp/datasets/google-research-datasets___mbpp/sanitized", "MBPP"),
        ("/root/autodl-tmp/datasets/openai___openai_humaneval/openai_humaneval", "HumanEval"),
    ]

    for path, name in datasets:
        all_ok &= check_directory_exists(path, name)
    print()

    # 检查 Python 依赖
    print("🐍 检查 Python 依赖:")
    try:
        import transformers
        print(f"✅ transformers: {transformers.__version__}")
    except ImportError:
        print("❌ transformers 未安装")
        all_ok = False

    try:
        import tqdm
        print(f"✅ tqdm: {tqdm.__version__}")
    except ImportError:
        print("❌ tqdm 未安装")
        all_ok = False

    print()

    # 检查输出目录
    print("📁 检查输出目录:")
    output_dirs = [
        "/root/autodl-tmp/processed_datasets",
        "/root/autodl-tmp/training_data"
    ]

    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path} (已创建)")

    print()
    print("="*60)

    if all_ok:
        print("✅ 环境检查通过！可以开始数据预处理。")
        print()
        print("运行命令:")
        print("  cd /root/grpo/preprocess")
        print("  ./run_all.sh")
    else:
        print("❌ 环境检查失败，请先解决上述问题。")

    print("="*60)

if __name__ == "__main__":
    main()
