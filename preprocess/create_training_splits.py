#!/usr/bin/env python3
"""
创建训练数据集 - 课程学习三阶段
Stage 1: 简单数学 (GSM8K + Ape210K)
Stage 2: 困难数学 (MATH×2 + CMATH)
Stage 3: 代码 (MBPP×30 + APPS×8 + 数学回放上限15k)
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


def create_stage1_simple_math():
    """Stage 1: 简单数学训练 (GSM8K + Ape210K)"""
    print("\n" + "="*60)
    print("创建 Stage 1: 简单数学训练集")
    print("="*60)

    simple_math_datasets = [
        "ape210k_train.json",
        "gsm8k_train.json"
    ]

    all_simple_math = []
    for dataset in simple_math_datasets:
        file_path = Path(PROCESSED_DIR) / dataset
        if file_path.exists():
            data = load_json(file_path)
            all_simple_math.extend(data)
            print(f"✅ 加载 {dataset}: {len(data)} 条")
        else:
            print(f"⚠️  文件不存在: {file_path}")

    # 下采样 Ape210K，使 GSM8K 占比接近 20%
    ape_samples = [s for s in all_simple_math if s.get('dataset') == 'ape210k']
    gsm_samples = [s for s in all_simple_math if s.get('dataset') == 'gsm8k']
    target_ape_count = len(gsm_samples) * 4

    original_ape_count = len(ape_samples)
    if gsm_samples and original_ape_count > target_ape_count:
        ape_samples = random.sample(ape_samples, target_ape_count)
        print(f"✅ 下采样 Ape210K: {len(ape_samples)} 条 (原 {original_ape_count})")

    balanced_data = ape_samples + gsm_samples
    random.shuffle(balanced_data)

    # 保存
    output_file = Path(OUTPUT_DIR) / "stage1_simple_math_train.json"
    save_json(balanced_data, output_file)

    print(f"\n📊 Stage 1 平衡后统计:")
    print(f"  总样本数: {len(balanced_data)}")
    datasets = Counter([s['dataset'] for s in balanced_data])
    for dataset, count in datasets.items():
        percentage = (count / len(balanced_data)) * 100
        print(f"  {dataset}: {count} 条 ({percentage:.1f}%)")
    if balanced_data and ape_samples:
        print(f"  下采样后 Ape210K 占比: {len(ape_samples)/len(balanced_data)*100:.1f}%")
    print(f"  保存至: {output_file}")


def create_stage2_hard_math():
    """Stage 2: 困难数学训练 (MATH×2 + CMATH)"""
    print("\n" + "="*60)
    print("创建 Stage 2: 困难数学训练集")
    print("="*60)

    all_hard_math = []

    # 加载 MATH 并上采样 ×2
    math_train = Path(PROCESSED_DIR) / "math_train.json"
    if math_train.exists():
        math_data = load_json(math_train)
        math_oversampled = oversample_data(math_data, 2)
        all_hard_math.extend(math_oversampled)
        print(f"✅ MATH: {len(math_data)} → {len(math_oversampled)} 条 (×2)")
    else:
        print(f"⚠️  文件不存在: {math_train}")

    # 加载 CMATH
    cmath_validation = Path(PROCESSED_DIR) / "cmath_validation.json"
    if cmath_validation.exists():
        cmath_data = load_json(cmath_validation)
        all_hard_math.extend(cmath_data)
        print(f"✅ CMATH: {len(cmath_data)} 条")
    else:
        print(f"⚠️  文件不存在: {cmath_validation}")

    # Shuffle
    random.shuffle(all_hard_math)

    # 保存
    output_file = Path(OUTPUT_DIR) / "stage2_hard_math_train.json"
    save_json(all_hard_math, output_file)

    print(f"\n📊 Stage 2 统计:")
    print(f"  总样本数: {len(all_hard_math)}")
    datasets = Counter([s['dataset'] for s in all_hard_math])
    for dataset, count in datasets.items():
        percentage = (count / len(all_hard_math)) * 100
        print(f"  {dataset}: {count} 条 ({percentage:.1f}%)")
    print(f"  保存至: {output_file}")


def create_stage3_code_with_replay():
    """Stage 3: 代码训练 + 数学经验回放（优化版）"""
    print("\n" + "="*60)
    print("创建 Stage 3: 代码训练 + 数学经验回放（优化版）")
    print("="*60)

    code_data = []

    # MBPP ×30（激进上采样）
    mbpp_train = Path(PROCESSED_DIR) / "mbpp_train.json"
    if mbpp_train.exists():
        mbpp = load_json(mbpp_train)
        mbpp_oversampled = oversample_data(mbpp, 30)
        code_data.extend(mbpp_oversampled)
        print(f"✅ MBPP: {len(mbpp)} → {len(mbpp_oversampled)} 条 (×30)")
    else:
        print(f"⚠️  文件不存在: {mbpp_train}")

    # APPS ×8（按难度分层采样，提升高难题占比）
    apps_train = Path(PROCESSED_DIR) / "apps_train.json"
    if apps_train.exists():
        apps = load_json(apps_train)

        by_diff = {}
        for item in apps:
            difficulty = item.get('difficulty', 'unknown')
            by_diff.setdefault(difficulty, []).append(item)

        sampled_apps = []

        # competition 尽可能全取
        competition_samples = by_diff.get('competition', [])
        sampled_apps.extend(competition_samples)

        # interview 采样上限 1800
        interview_pool = by_diff.get('interview', [])
        interview_take = min(1800, len(interview_pool))
        if interview_take > 0:
            sampled_apps.extend(random.sample(interview_pool, interview_take))

        # introductory 采样上限 2000
        intro_pool = by_diff.get('introductory', [])
        intro_take = min(2000, len(intro_pool))
        if intro_take > 0:
            sampled_apps.extend(random.sample(intro_pool, intro_take))

        # 其他难度标签全保留（避免丢失潜在有价值样本）
        for diff_key, diff_samples in by_diff.items():
            if diff_key not in {"competition", "interview", "introductory"}:
                sampled_apps.extend(diff_samples)

        sampled_counter = Counter([x.get('difficulty', 'unknown') for x in sampled_apps])
        print(
            "✅ APPS 难度分层采样: "
            f"{len(sampled_apps)} 条 "
            f"(competition: {sampled_counter.get('competition', 0)}, "
            f"interview: {sampled_counter.get('interview', 0)}, "
            f"introductory: {sampled_counter.get('introductory', 0)})"
        )

        apps_oversampled = oversample_data(sampled_apps, 8)
        code_data.extend(apps_oversampled)
        print(f"✅ APPS 过采样后: {len(apps_oversampled)} 条 (×8)")
    else:
        print(f"⚠️  文件不存在: {apps_train}")

    all_math_data = []
    math_datasets = [
        "ape210k_train.json",
        "gsm8k_train.json",
        "math_train.json",
        "cmath_validation.json"
    ]
    for ds in math_datasets:
        p = Path(PROCESSED_DIR) / ds
        if p.exists():
            all_math_data.extend(load_json(p))

    # 数学回放下采样到 10k–15k（上限15k）
    if all_math_data:
        replay_size = min(15000, int(len(all_math_data) * 0.15))
        math_replay = random.sample(all_math_data, replay_size)
        print(f"✅ 数学回放: {replay_size} 条 (控制上限)")
    else:
        math_replay = []
        print("⚠️  没有数学数据可用于经验回放")

    stage3_data = code_data + math_replay
    random.shuffle(stage3_data)

    output_file = Path(OUTPUT_DIR) / "stage3_code_train.json"
    save_json(stage3_data, output_file)

    print(f"\n📊 Stage 3 统计:")
    print(f"  总样本数: {len(stage3_data)}")
    task_types = Counter([s['task_type'] for s in stage3_data])
    for task_type, count in task_types.items():
        percentage = (count / len(stage3_data)) * 100 if stage3_data else 0
        print(f"  {task_type}: {count} 条 ({percentage:.1f}%)")

    code_samples = [s for s in stage3_data if s['task_type'] == 'code']
    if code_samples:
        code_datasets = Counter([s['dataset'] for s in code_samples])
        print(f"\n  代码来源分布:")
        for dataset, count in code_datasets.items():
            percentage = (count / len(code_samples)) * 100
            print(f"    {dataset}: {count} 条 ({percentage:.1f}%)")

    print(f"\n  保存至: {output_file}")


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
        "apps_test.json": "apps_test",
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
    print("开始创建训练数据集 - 课程学习三阶段")
    print("="*60)

    # 设置随机种子以保证可复现
    random.seed(42)

    # 创建各阶段训练集
    create_stage1_simple_math()
    create_stage2_hard_math()
    create_stage3_code_with_replay()

    # 创建测试集
    create_test_sets()

    print("\n" + "="*60)
    print("训练数据集创建完成！")
    print(f"所有数据保存在: {OUTPUT_DIR}")
    print("="*60)
    print("\n📚 课程学习训练流程:")
    print("  Stage 1: stage1_simple_math_train.json")
    print("           简单数学 (GSM8K + Ape210K)")
    print("           训练 2-3 epochs, lr=1e-5~5e-5")
    print()
    print("  Stage 2: stage2_hard_math_train.json")
    print("           困难数学 (MATH×2 + CMATH)")
    print("           训练 2-3 epochs, lr=1e-5~5e-5")
    print()
    print("  Stage 3: stage3_code_train.json")
    print("           代码生成 (MBPP×30 + APPS×8 + 数学回放上限15k)")
    print("           训练 3-5 epochs, lr=5e-6~2e-5")
    print()
    print("  💡 提示: 代码占比显著提升，若过拟合可回调 MBPP/APPS 倍数")
    print("="*60)
