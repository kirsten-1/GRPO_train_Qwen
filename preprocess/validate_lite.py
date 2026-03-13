#!/usr/bin/env python3
"""
轻量级数据验证脚本 - 只检查文件元信息，不加载完整数据
"""
import json
import os
from pathlib import Path

PROCESSED_DIR = "/root/autodl-tmp/processed_datasets"
TRAINING_DIR = "/root/autodl-tmp/training_data"


def get_json_line_count(file_path):
    """快速统计 JSON 数组中的元素数量（不加载完整数据）"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取第一行判断格式
            first_char = f.read(1)
            f.seek(0)

            if first_char == '[':
                # JSON 数组格式，统计逗号数量 + 1
                content = f.read()
                # 简单估算：统计顶层对象数量
                count = content.count('"id":')
                return count
            else:
                # JSONL 格式，统计行数
                return sum(1 for line in f if line.strip())
    except Exception as e:
        return f"Error: {e}"


def check_file_info(file_path):
    """检查文件基本信息"""
    if not file_path.exists():
        return None

    size_mb = file_path.stat().st_size / (1024 * 1024)
    count = get_json_line_count(file_path)

    return {
        'size_mb': size_mb,
        'count': count
    }


def validate_processed_datasets():
    """验证处理后的数据集"""
    print("="*60)
    print("验证处理后的数据集")
    print("="*60)

    datasets = [
        ("ape210k_train.json", "Ape210K Train"),
        ("ape210k_valid.json", "Ape210K Valid"),
        ("ape210k_test.json", "Ape210K Test"),
        ("gsm8k_train.json", "GSM8K Train"),
        ("gsm8k_test.json", "GSM8K Test"),
        ("math_train.json", "MATH Train"),
        ("math_test.json", "MATH Test"),
        ("cmath_validation.json", "CMATH Validation"),
        ("cmath_test.json", "CMATH Test"),
        ("mbpp_train.json", "MBPP Train"),
        ("mbpp_test.json", "MBPP Test"),
        ("apps_train.json", "APPS Train"),
        ("apps_test.json", "APPS Test"),
        ("humaneval_test.json", "HumanEval Test"),
    ]

    for filename, name in datasets:
        file_path = Path(PROCESSED_DIR) / filename
        info = check_file_info(file_path)

        if info:
            print(f"✅ {name:20s}: {info['count']:6} 条, {info['size_mb']:6.1f} MB")
        else:
            print(f"❌ {name:20s}: 文件不存在")

    print()


def validate_training_splits():
    """验证训练数据集"""
    print("="*60)
    print("验证训练数据集")
    print("="*60)

    stages = [
        ("stage1_simple_math_train.json", "Stage 1: 简单数学"),
        ("stage2_hard_math_train.json", "Stage 2: 困难数学"),
        ("stage3_code_train.json", "Stage 3: 代码+回放"),
    ]

    total_samples = 0
    total_size = 0

    for filename, name in stages:
        file_path = Path(TRAINING_DIR) / filename
        info = check_file_info(file_path)

        if info:
            print(f"✅ {name:25s}: {info['count']:6} 条, {info['size_mb']:6.1f} MB")
            if isinstance(info['count'], int):
                total_samples += info['count']
            total_size += info['size_mb']
        else:
            print(f"❌ {name:25s}: 文件不存在")

    print(f"\n总计: {total_samples:,} 条样本, {total_size:.1f} MB")
    print()


def validate_test_sets():
    """验证测试集"""
    print("="*60)
    print("验证测试集")
    print("="*60)

    test_sets = [
        ("gsm8k_test.json", "GSM8K"),
        ("ape210k_test.json", "Ape210K"),
        ("math_test.json", "MATH"),
        ("cmath_test.json", "CMATH"),
        ("mbpp_test.json", "MBPP"),
        ("apps_test.json", "APPS"),
        ("humaneval_test.json", "HumanEval"),
    ]

    for filename, name in test_sets:
        file_path = Path(TRAINING_DIR) / filename
        info = check_file_info(file_path)

        if info:
            print(f"✅ {name:15s}: {info['count']:5} 条, {info['size_mb']:5.1f} MB")
        else:
            print(f"❌ {name:15s}: 文件不存在")

    print()


def sample_check_format():
    """抽样检查数据格式（只读取第一条）"""
    print("="*60)
    print("抽样检查数据格式")
    print("="*60)

    # 检查 Stage 3 的第一条数据
    stage3_path = Path(TRAINING_DIR) / "stage3_code_train.json"

    if stage3_path.exists():
        try:
            with open(stage3_path, 'r', encoding='utf-8') as f:
                # 只读取开头部分
                content = f.read(5000)  # 读取前 5KB

                # 尝试解析第一个对象
                if content.startswith('['):
                    # 找到第一个完整对象
                    first_obj_end = content.find('},')
                    if first_obj_end > 0:
                        first_obj_str = content[1:first_obj_end+1]
                        try:
                            first_obj = json.loads(first_obj_str)

                            print("✅ Stage 3 第一条数据格式检查:")
                            print(f"   - ID: {first_obj.get('id', 'N/A')}")
                            print(f"   - Dataset: {first_obj.get('dataset', 'N/A')}")
                            print(f"   - Task Type: {first_obj.get('task_type', 'N/A')}")

                            if 'messages' in first_obj:
                                print(f"   - Messages: {len(first_obj['messages'])} 条")
                                if len(first_obj['messages']) > 2:
                                    assistant_msg = first_obj['messages'][2].get('content', '')
                                    has_reasoning = '<reasoning>' in assistant_msg
                                    has_answer = '<answer>' in assistant_msg
                                    print(f"   - 包含 <reasoning>: {has_reasoning}")
                                    print(f"   - 包含 <answer>: {has_answer}")
                        except:
                            print("⚠️  无法解析第一条数据")
        except Exception as e:
            print(f"❌ 读取失败: {e}")
    else:
        print("❌ Stage 3 文件不存在")

    print()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("轻量级数据验证（不加载完整数据）")
    print("="*60)
    print()

    validate_processed_datasets()
    validate_training_splits()
    validate_test_sets()
    sample_check_format()

    print("="*60)
    print("验证完成！")
    print("="*60)
