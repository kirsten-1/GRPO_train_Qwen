#!/usr/bin/env python3
"""
验证处理后的数据集质量
"""
import json
import random
from collections import Counter
from pathlib import Path

# 期望的 System Prompt（前150字符用于验证）
EXPECTED_SYSTEM_PROMPT_PREFIX = """你是一个专业的AI助手，擅长数学推理和编程。请一步步思考并解决问题。

回答格式：

<reasoning>
在这里展示你的思考过程：
- 理解问题的关键点
- 分析解题思路
- 展示详细的推导或实现步骤"""


def validate_processed_data(data_path, num_samples=10):
    """验证处理后的数据"""

    # 加载数据
    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        print(f"❌ 不支持的文件格式: {data_path}")
        return

    print(f"\n{'='*60}")
    print(f"验证数据集: {data_path}")
    print(f"总样本数: {len(data)}")
    print(f"{'='*60}\n")

    # 统计任务类型分布
    task_types = [sample.get('task_type', 'unknown') for sample in data]
    print("📊 任务类型分布：")
    for task_type, count in Counter(task_types).items():
        percentage = (count / len(data)) * 100
        print(f"  {task_type}: {count} 条 ({percentage:.1f}%)")
    print()

    # 随机抽样
    samples = random.sample(data, min(num_samples, len(data)))

    issues = []

    for idx, sample in enumerate(samples, 1):
        print(f"\n--- 样本 {idx} ---")
        print(f"ID: {sample.get('id', 'N/A')}")
        print(f"数据集: {sample.get('dataset', 'N/A')}")
        print(f"任务类型: {sample.get('task_type', 'N/A')}")

        # 检查1：messages格式
        messages = sample.get('messages', [])
        if not messages:
            issues.append(f"样本{idx}: 缺少messages字段")
            continue

        # 检查2：system prompt
        if messages[0]['role'] != 'system':
            issues.append(f"样本{idx}: 缺少system消息")
        else:
            print(f"\n✅ System prompt存在")
            print(f"内容预览: {messages[0]['content'][:100]}...")

            # 检查 system prompt 一致性
            system_content = messages[0]['content']
            if not system_content.startswith(EXPECTED_SYSTEM_PROMPT_PREFIX[:150]):
                issues.append(f"样本{idx}: system prompt 内容不一致")
                print(f"⚠️  System prompt 内容与预期不一致")

        # 检查3：user消息
        if len(messages) < 2 or messages[1]['role'] != 'user':
            issues.append(f"样本{idx}: 缺少user消息")
        else:
            print(f"\n✅ User消息存在")
            print(f"问题: {messages[1]['content'][:200]}...")

        # 检查4：assistant消息（如果有）
        if len(messages) >= 3:
            assistant_msg = messages[2]['content']

            # 检查reasoning标签
            if '<reasoning>' not in assistant_msg or '</reasoning>' not in assistant_msg:
                issues.append(f"样本{idx}: 缺少<reasoning>标签")
            else:
                print(f"\n✅ <reasoning>标签存在")
                try:
                    reas_content = assistant_msg.split('<reasoning>')[1].split('</reasoning>')[0].strip()
                    word_count = len(reas_content.split())
                    task_type = sample.get('task_type')

                    # 仅对 code 任务检查常规阈值；全任务保留极短兜底检查
                    if task_type == 'code' and word_count < 20:
                        issues.append(f"样本{idx}: <reasoning> 内容过短 (词数: {word_count}, task: code)")
                    elif len(reas_content) < 20:
                        issues.append(f"样本{idx}: <reasoning> 极短 (长度: {len(reas_content)}字符)")
                except Exception:
                    issues.append(f"样本{idx}: <reasoning> 标签解析失败")

            # 检查answer标签
            if '<answer>' not in assistant_msg or '</answer>' not in assistant_msg:
                issues.append(f"样本{idx}: 缺少<answer>标签")
            else:
                print(f"\n✅ <answer>标签存在")

            # 增强检查：answer内容合理性
            if '<answer>' in assistant_msg:
                try:
                    ans_content = assistant_msg.split('<answer>')[1].split('</answer>')[0].strip()
                    if sample.get('task_type') == 'code':
                        ans_word_count = len(ans_content.split())
                        if ans_word_count < 5:
                            issues.append(f"样本{idx}: <answer> 内容过短 (词数: {ans_word_count}, task: code)")
                        if '```' in ans_content and 'python' not in ans_content.lower():
                            issues.append(f"样本{idx}: 疑似代码但缺少 python 标记")
                except:
                    issues.append(f"样本{idx}: <answer> 标签解析失败")

            # 打印完整assistant回复
            print(f"\nAssistant回复:")
            print(assistant_msg[:500])
            if len(assistant_msg) > 500:
                print("...")

        # 检查5：中英文混合
        full_text = str(sample)
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in full_text)
        has_english = any('a' <= char.lower() <= 'z' for char in full_text)
        print(f"\n语言检测: 中文={has_chinese}, 英文={has_english}")

        print(f"\n{'-'*60}")

    # 总结
    print(f"\n{'='*60}")
    print(f"验证完成")
    if issues:
        print(f"\n⚠️  发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✅ 所有检查通过！")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 验证所有处理后的数据集
    datasets_to_validate = [
        "/root/autodl-tmp/processed_datasets/gsm8k_train.json",
        "/root/autodl-tmp/processed_datasets/gsm8k_test.json",
        "/root/autodl-tmp/processed_datasets/ape210k_train.json",
        "/root/autodl-tmp/processed_datasets/ape210k_valid.json",
        "/root/autodl-tmp/processed_datasets/ape210k_test.json",
        "/root/autodl-tmp/processed_datasets/math_train.json",
        "/root/autodl-tmp/processed_datasets/math_test.json",
        "/root/autodl-tmp/processed_datasets/cmath_validation.json",
        "/root/autodl-tmp/processed_datasets/cmath_test.json",
        "/root/autodl-tmp/processed_datasets/mbpp_train.json",
        "/root/autodl-tmp/processed_datasets/mbpp_test.json",
        "/root/autodl-tmp/processed_datasets/mbpp_validation.json",
        "/root/autodl-tmp/processed_datasets/apps_train.json",
        "/root/autodl-tmp/processed_datasets/apps_test.json",
        "/root/autodl-tmp/processed_datasets/humaneval_test.json",
    ]

    for dataset_path in datasets_to_validate:
        try:
            if Path(dataset_path).exists():
                validate_processed_data(dataset_path, num_samples=5)
            else:
                print(f"⚠️  文件不存在，跳过: {dataset_path}\n")
        except Exception as e:
            print(f"❌ 验证失败: {dataset_path}")
            print(f"错误: {e}\n")
