#!/usr/bin/env python3
"""
预览数据格式示例 - 展示处理后的数据结构
"""
import json

# 示例数据
EXAMPLE_MATH = {
    "id": "gsm8k_train_0",
    "dataset": "gsm8k",
    "task_type": "math",
    "messages": [
        {
            "role": "system",
            "content": "你是一个专业的AI助手，擅长数学推理和编程。请一步步思考并解决问题。\n\n回答格式：\n\n<reasoning>\n在这里展示你的思考过程：\n- 理解问题的关键点\n- 分析解题思路\n- 展示详细的推导或实现步骤\n- 验证答案的正确性\n</reasoning>\n\n<answer>\n在这里给出最终答案（数学题给出数值，编程题给出完整代码）\n</answer>\n\n注意：推理过程要详细、有逻辑，但避免冗余重复。"
        },
        {
            "role": "user",
            "content": "小明有 15 个苹果，他给了小红 3 个，又给了小刚 5 个。小明还剩多少个苹果？"
        },
        {
            "role": "assistant",
            "content": "<reasoning>\n让我一步步分析这道数学题：\n\n1. 理解题意：小明原本有 15 个苹果\n2. 分析解题思路：\n   - 给了小红 3 个，剩余：15 - 3 = 12 个\n   - 又给了小刚 5 个，剩余：12 - 5 = 7 个\n3. 计算并验证：15 - 3 - 5 = 7 个\n</reasoning>\n\n<answer>\n7\n</answer>"
        }
    ]
}

EXAMPLE_CODE = {
    "id": "mbpp_train_0",
    "dataset": "mbpp",
    "task_type": "code",
    "messages": [
        {
            "role": "system",
            "content": "你是一个专业的AI助手，擅长数学推理和编程。请一步步思考并解决问题。\n\n回答格式：\n\n<reasoning>\n在这里展示你的思考过程：\n- 理解问题的关键点\n- 分析解题思路\n- 展示详细的推导或实现步骤\n- 验证答案的正确性\n</reasoning>\n\n<answer>\n在这里给出最终答案（数学题给出数值，编程题给出完整代码）\n</answer>\n\n注意：推理过程要详细、有逻辑，但避免冗余重复。"
        },
        {
            "role": "user",
            "content": "编写一个函数，判断一个数是否为质数。\n\n测试用例：\nassert is_prime(7) == True\nassert is_prime(10) == False"
        },
        {
            "role": "assistant",
            "content": "<reasoning>\n让我分析这个编程问题：\n\n1. 理解需求：判断一个数是否为质数\n2. 设计思路：\n   - 质数定义：大于 1 且只能被 1 和自身整除\n   - 特殊情况：小于等于 1 的数不是质数\n   - 优化：只需检查到 sqrt(n)\n3. 编写代码并验证\n</reasoning>\n\n<answer>\ndef is_prime(n):\n    if n <= 1:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(n**0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True\n</answer>"
        }
    ]
}

def print_example(example, title):
    """打印示例"""
    print("\n" + "="*70)
    print(f"📝 {title}")
    print("="*70)
    print(json.dumps(example, ensure_ascii=False, indent=2))
    print()

def main():
    print("="*70)
    print("数据格式预览")
    print("="*70)
    print("\n这是处理后的数据格式示例。实际运行脚本后，所有数据都会转换为这种格式。")

    print_example(EXAMPLE_MATH, "数学题示例（GSM8K）")
    print_example(EXAMPLE_CODE, "编程题示例（MBPP）")

    print("="*70)
    print("关键特征:")
    print("="*70)
    print("✅ 统一的 System Prompt（中文、结构化）")
    print("✅ 清晰的 <reasoning> 推理过程")
    print("✅ 明确的 <answer> 最终答案")
    print("✅ 标准的 messages 格式（system + user + assistant）")
    print("✅ 任务类型标记（math / code）")
    print("="*70)
    print("\n运行 preprocess_datasets.py 后，所有数据都会转换为这种格式！")
    print("="*70)

if __name__ == "__main__":
    main()
