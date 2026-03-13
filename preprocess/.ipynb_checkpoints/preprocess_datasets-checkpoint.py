#!/usr/bin/env python3
"""
数据预处理脚本 - 统一格式化所有数据集
"""
import json
import os
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

# 推荐的 System Prompt
SYSTEM_PROMPT_RECOMMENDED = """你是一个专业的AI助手，擅长数学推理和编程。请一步步思考并解决问题。

回答格式：

<reasoning>
在这里展示你的思考过程：
- 理解问题的关键点
- 分析解题思路
- 展示详细的推导或实现步骤
- 验证答案的正确性
</reasoning>

<answer>
在这里给出最终答案（数学题给出数值，编程题给出完整代码）
</answer>

注意：推理过程要详细、有逻辑，但避免冗余重复。"""

# 加载 tokenizer
MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 输出目录
OUTPUT_DIR = "/root/autodl-tmp/processed_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def format_with_reasoning_tags(question, answer, task_type="math"):
    """将答案格式化为带有 reasoning 和 answer 标签的格式"""
    if '<reasoning>' in answer and '<answer>' in answer:
        return answer

    if task_type == "code":
        reasoning = f"让我分析这个编程问题：\n\n1. 理解需求：{question[:100]}...\n2. 设计思路：需要实现相应的功能\n3. 编写代码并验证"
        formatted_answer = f"<reasoning>\n{reasoning}\n</reasoning>\n\n<answer>\n{answer}\n</answer>"
    else:
        reasoning = f"让我一步步分析这道数学题：\n\n1. 理解题意\n2. 分析解题思路\n3. 计算并验证"
        formatted_answer = f"<reasoning>\n{reasoning}\n</reasoning>\n\n<answer>\n{answer}\n</answer>"

    return formatted_answer


def process_ape210k():
    """处理 Ape210K 数据集"""
    print("\n处理 Ape210K...")
    base_path = "/root/autodl-tmp/datasets/ape210k-master/data"

    for split in ['train', 'valid', 'test']:
        input_file = f"{base_path}/{split}.ape.json"
        output_file = f"{OUTPUT_DIR}/ape210k_{split}.json"

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed = []
        for item in tqdm(data, desc=f"Ape210K {split}"):
            question = item.get('original_text', item.get('question', ''))
            answer = str(item.get('ans', item.get('answer', '')))

            formatted_answer = format_with_reasoning_tags(question, answer, "math")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_RECOMMENDED},
                {"role": "user", "content": question},
                {"role": "assistant", "content": formatted_answer}
            ]

            processed.append({
                "id": f"ape210k_{split}_{len(processed)}",
                "dataset": "ape210k",
                "task_type": "math",
                "messages": messages
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        print(f"✅ Ape210K {split}: {len(processed)} 条 -> {output_file}")


def process_gsm8k():
    """处理 GSM8K 数据集"""
    print("\n处理 GSM8K...")
    base_path = "/root/autodl-tmp/datasets/openai___gsm8k/main"

    for split in ['train', 'test']:
        input_file = f"{base_path}/{split}.jsonl"
        output_file = f"{OUTPUT_DIR}/gsm8k_{split}.json"

        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        processed = []
        for item in tqdm(data, desc=f"GSM8K {split}"):
            question = item.get('question', '')
            answer = item.get('answer', '')

            formatted_answer = format_with_reasoning_tags(question, answer, "math")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_RECOMMENDED},
                {"role": "user", "content": question},
                {"role": "assistant", "content": formatted_answer}
            ]

            processed.append({
                "id": f"gsm8k_{split}_{len(processed)}",
                "dataset": "gsm8k",
                "task_type": "math",
                "messages": messages
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        print(f"✅ GSM8K {split}: {len(processed)} 条 -> {output_file}")


def process_math():
    """处理 MATH 数据集（代数部分）"""
    print("\n处理 MATH...")
    base_path = "/root/autodl-tmp/datasets/EleutherAI___hendrycks_math/algebra"

    for split in ['train', 'test']:
        input_file = f"{base_path}/{split}.jsonl"
        output_file = f"{OUTPUT_DIR}/math_{split}.json"

        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        processed = []
        for item in tqdm(data, desc=f"MATH {split}"):
            question = item.get('problem', '')
            answer = item.get('solution', '')

            formatted_answer = format_with_reasoning_tags(question, answer, "math")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_RECOMMENDED},
                {"role": "user", "content": question},
                {"role": "assistant", "content": formatted_answer}
            ]

            processed.append({
                "id": f"math_{split}_{len(processed)}",
                "dataset": "math",
                "task_type": "math",
                "messages": messages
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        print(f"✅ MATH {split}: {len(processed)} 条 -> {output_file}")


def process_cmath():
    """处理 CMATH 数据集"""
    print("\n处理 CMATH...")
    base_path = "/root/autodl-tmp/datasets/weitianwen___cmath/default"

    for split in ['validation', 'test']:
        input_file = f"{base_path}/{split}.jsonl"
        output_file = f"{OUTPUT_DIR}/cmath_{split}.json"

        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        processed = []
        for item in tqdm(data, desc=f"CMATH {split}"):
            question = item.get('question', item.get('problem', ''))
            answer = item.get('answer', item.get('solution', ''))

            formatted_answer = format_with_reasoning_tags(question, answer, "math")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_RECOMMENDED},
                {"role": "user", "content": question},
                {"role": "assistant", "content": formatted_answer}
            ]

            processed.append({
                "id": f"cmath_{split}_{len(processed)}",
                "dataset": "cmath",
                "task_type": "math",
                "messages": messages
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        print(f"✅ CMATH {split}: {len(processed)} 条 -> {output_file}")


def process_mbpp():
    """处理 MBPP 数据集"""
    print("\n处理 MBPP...")
    base_path = "/root/autodl-tmp/datasets/google-research-datasets___mbpp/sanitized"

    for split in ['train', 'test', 'validation', 'prompt']:
        input_file = f"{base_path}/{split}.jsonl"
        output_file = f"{OUTPUT_DIR}/mbpp_{split}.json"

        if not os.path.exists(input_file):
            print(f"⚠️  文件不存在: {input_file}")
            continue

        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        processed = []
        for item in tqdm(data, desc=f"MBPP {split}"):
            question = item.get('text', item.get('prompt', ''))
            code = item.get('code', '')
            test_list = item.get('test_list', [])

            # 构建完整的问题描述
            full_question = f"{question}\n\n测试用例：\n" + "\n".join(test_list) if test_list else question

            formatted_answer = format_with_reasoning_tags(full_question, code, "code")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_RECOMMENDED},
                {"role": "user", "content": full_question},
                {"role": "assistant", "content": formatted_answer}
            ]

            processed.append({
                "id": f"mbpp_{split}_{len(processed)}",
                "dataset": "mbpp",
                "task_type": "code",
                "messages": messages
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        print(f"✅ MBPP {split}: {len(processed)} 条 -> {output_file}")


def process_humaneval():
    """处理 HumanEval 数据集"""
    print("\n处理 HumanEval...")
    base_path = "/root/autodl-tmp/datasets/openai___openai_humaneval/openai_humaneval"

    input_file = f"{base_path}/test.jsonl"
    output_file = f"{OUTPUT_DIR}/humaneval_test.json"

    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    processed = []
    for item in tqdm(data, desc="HumanEval test"):
        prompt = item.get('prompt', '')
        canonical_solution = item.get('canonical_solution', '')
        test = item.get('test', '')

        # 构建完整的问题描述
        full_question = f"{prompt}\n\n测试用例：\n{test}"

        formatted_answer = format_with_reasoning_tags(full_question, canonical_solution, "code")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_RECOMMENDED},
            {"role": "user", "content": full_question},
            {"role": "assistant", "content": formatted_answer}
        ]

        processed.append({
            "id": f"humaneval_test_{len(processed)}",
            "dataset": "humaneval",
            "task_type": "code",
            "messages": messages
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"✅ HumanEval test: {len(processed)} 条 -> {output_file}")


if __name__ == "__main__":
    print("="*60)
    print("开始数据预处理")
    print("="*60)

    try:
        process_ape210k()
    except Exception as e:
        print(f"❌ Ape210K 处理失败: {e}")

    try:
        process_gsm8k()
    except Exception as e:
        print(f"❌ GSM8K 处理失败: {e}")

    try:
        process_math()
    except Exception as e:
        print(f"❌ MATH 处理失败: {e}")

    try:
        process_cmath()
    except Exception as e:
        print(f"❌ CMATH 处理失败: {e}")

    try:
        process_mbpp()
    except Exception as e:
        print(f"❌ MBPP 处理失败: {e}")

    try:
        process_humaneval()
    except Exception as e:
        print(f"❌ HumanEval 处理失败: {e}")

    print("\n" + "="*60)
    print("数据预处理完成！")
    print(f"所有处理后的数据保存在: {OUTPUT_DIR}")
    print("="*60)
