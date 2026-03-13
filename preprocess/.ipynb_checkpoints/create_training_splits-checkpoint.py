#!/usr/bin/env python3
"""
创建训练数据集 - 包含过采样和经验回放
"""
import json
import random
from pathlib import Path
from collections import Counter

# 输入输出目录
PROCESSED_DIR = "/root/autodl-tmp/processed_datasets"
OUTPUT_DIR = "/root/autodl-tmp/training_data"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def oversample_data(data, multiplier):
    """过采样数据"""
    return data * multiplier


def create_stage1_math_only():
    """Stage 1: 纯数学训练"""
    print("\n" + "="*60)
    print("创建 Stage 1: 纯数学训练集")
    print("="*60)

    math_datasets = [
        "ape210k_train.json",
        "gsm8k_train.json",
        "math_train.json",
        "cmath_validation.json"
    ]

    all_math_data = []
    for dataset in math_datasets:
        file_path = Path(PROCESSED_DIR) / dataset
        if file_path.exists():
            data = load_json(file_path)
            all_math_data.extend(data)
            print(f"✅ 加载 {dataset}: {len(data)} 条")
        else:
            print(f"⚠️  文件不存在: {file_path}")

    # Shuffle
    random.shuffle(all_math_data)

    # 保存
    output_file = Path(OUTPUT_DIR) / "stage1_math_train.json"
    save_json(all_math_data, output_file)

    print(f"\n📊 Stage 1 统计:")
    print(f"  总样本数: {len(all_math_data)}")
    task_types = Counter([s['task_type'] for s in all_math_data])
    for task_type, count in task_types.items():
        print(f"  {task_type}: {count} 条")
    print(f"  保存至: {output_file}")


def create_stage2_code_with_replay():
    """Stage 2: 代码训练 + 数学经验回放"""
    print("\n" + "="*60)
    print("创建 Stage 2: 代码训练 + 数学经验回放")
    print("="*60)

    # 1. 加载代码数据
    code_data = []
    mbpp_train = Path(PROCESSED_DIR) / "mbpp_train.json"
    if mbpp_train.exists():
        mbpp = load_json(mbpp_train)
        # MBPP 过采样 10倍
        mbpp_oversampled = oversample_data(mbpp, 10)
        code_data.extend(mbpp_oversampled)
        print(f"✅ MBPP: {len(mbpp)} → {len(mbpp_oversampled)} 条 (×10)")

    # HumanEval 可选（通常用于测试）
    # humaneval_test = Path(PROCESSED_DIR) / "humaneval_test.json"
    # if humaneval_test.exists():
    #     humaneval = load_json(humaneval_test)
    #     humaneval_oversampled = oversample_data(humaneval, 5)
    #     code_data.extend(humaneval_oversampled)
    #     print(f"✅ HumanEval: {len(humaneval)} → {len(humaneval_oversampled)} 条 (×5)")

    # 2. 加载所有数学数据用于经验回放
    math_datasets = [
        "ape210k_train.json",
        "gsm8k_train.json",
        "math_train.json",
        "cmath_validation.json"
    ]

    all_math_data = []
    for dataset in math_datasets:
        file_path = Path(PROCESSED_DIR) / dataset
        if file_path.exists():
            data = load_json(file_path)
            all_math_data.extend(data)

    # 3. 经验回放：随机抽取 15% 数学数据
    replay_size = int(len(all_math_data) * 0.15)
    math_replay = random.sample(all_math_data, replay_size)
    print(f"✅ 数学经验回放: {replay_size} 条 (15% of {len(all_math_data)})")

    # 4. 合并并 shuffle
    stage2_data = code_data + math_replay
    random.shuffle(stage2_data)

    # 保存
    output_file = Path(OUTPUT_DIR) / "stage2_code_train.json"
    save_json(stage2_data, output_file)

    print(f"\n📊 Stage 2 统计:")
    print(f"  总样本数: {len(stage2_data)}")
    task_types = Counter([s['task_type'] for s in stage2_data])
    for task_type, count in task_types.items():
        percentage = (count / len(stage2_data)) * 100
        print(f"  {task_type}: {count} 条 ({percentage:.1f}%)")
    print(f"  保存至: {output_file}")


def create_stage3_mixed():
    """Stage 3: 混合训练（可选）"""
    print("\n" + "="*60)
    print("创建 Stage 3: 混合训练集（数学+代码）")
    print("="*60)

    # 加载所有数学数据
    math_datasets = [
        "ape210k_train.json",
        "gsm8k_train.json",
        "math_train.json",
        "cmath_validation.json"
    ]

    all_math_data = []
    for dataset in math_datasets:
        file_path = Path(PROCESSED_DIR) / dataset
        if file_path.exists():
            data = load_json(file_path)
            all_math_data.extend(data)

    # 加载代码数据（适度过采样）
    code_data = []
    mbpp_train = Path(PROCESSED_DIR) / "mbpp_train.json"
    if mbpp_train.exists():
        mbpp = load_json(mbpp_train)
        mbpp_oversampled = oversample_data(mbpp, 5)  # Stage 3 只需 5倍
        code_data.extend(mbpp_oversampled)
        print(f"✅ MBPP: {len(mbpp)} → {len(mbpp_oversampled)} 条 (×5)")

    # 合并并 shuffle
    stage3_data = all_math_data + code_data
    random.shuffle(stage3_data)

    # 保存
    output_file = Path(OUTPUT_DIR) / "stage3_mixed_train.json"
    save_json(stage3_data, output_file)

    print(f"\n📊 Stage 3 统计:")
    print(f"  总样本数: {len(stage3_data)}")
    task_types = Counter([s['task_type'] for s in stage3_data])
    for task_type, count in task_types.items():
        percentage = (count / len(stage3_data)) * 100
        print(f"  {task_type}: {count} 条 ({percentage:.1f}%)")
    print(f"  保存至: {output_file}")


def create_test_sets():
    """创建测试集"""
    print("\n" + "="*60)
    print("创建测试集")
    print("="*60)

    test_datasets = {
        "gsm8k_test.json": "gsm8k_test",
        "ape210k_test.json": "ape210k_test",
        "math_test.json": "math_test",
        "cmath_test.json": "cmath_test",
        "mbpp_test.json": "mbpp_test",
        "humaneval_test.json": "humaneval_test"
    }

    for source_file, output_name in test_datasets.items():
        source_path = Path(PROCESSED_DIR) / source_file
        if source_path.exists():
            data = load_json(source_path)
            output_file = Path(OUTPUT_DIR) / f"{output_name}.json"
            save_json(data, output_file)
            print(f"✅ {output_name}: {len(data)} 条 -> {output_file}")
        else:
            print(f"⚠️  文件不存在: {source_path}")


if __name__ == "__main__":
    print("="*60)
    print("开始创建训练数据集")
    print("="*60)

    # 设置随机种子以保证可复现
    random.seed(42)

    # 创建各阶段训练集
    create_stage1_math_only()
    create_stage2_code_with_replay()
    create_stage3_mixed()

    # 创建测试集
    create_test_sets()

    print("\n" + "="*60)
    print("训练数据集创建完成！")
    print(f"所有数据保存在: {OUTPUT_DIR}")
    print("="*60)
    print("\n推荐训练流程:")
    print("  1. Stage 1: 使用 stage1_math_train.json 训练数学能力")
    print("  2. Stage 2: 使用 stage2_code_train.json 训练代码能力（含数学回放）")
    print("  3. Stage 3: 使用 stage3_mixed_train.json 进行混合微调（可选）")
    print("="*60)
