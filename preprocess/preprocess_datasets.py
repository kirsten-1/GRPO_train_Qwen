#!/usr/bin/env python3
"""
数据预处理脚本 - 统一格式化所有数据集
"""
import json
import os
import random
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk, load_dataset

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

# 输出目录
OUTPUT_DIR = "/root/autodl-tmp/processed_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_ape210k_reasoning(question, equation, answer):
    """利用 equation 生成结构化推理过程 + 轻微随机化"""
    question = str(question)
    equation = str(equation)
    answer = str(answer)
    question_short = question[:80] + "..." if len(question) > 80 else question
    equation_text = equation if equation.strip() else "直接计算或观察规律"

    templates = [
        "让我仔细分析这道题：\n问题：{q}\n关键步骤：\n1. 找出数量关系\n2. 列出方程：{e}\n3. 解方程得到：{a}\n4. 检查是否合理",
        "一步步来解这道题：\n首先理解：{q}\n然后列式：{e}\n计算过程：\n最后答案：{a}\n验证一下：",
        "题目要求：{q}\n我们可以用方程表示：{e}\n解得：{a}\n再想想有没有其他方法验证？",
        "分析思路：\n已知：{q}\n建立模型：{e}\n求解：{a}\n确认正确性："
    ]
    template = random.choice(templates)
    return template.format(q=question_short, e=equation_text, a=answer)


def create_code_reasoning(question):
    """统一的代码任务推理模板（MBPP / HumanEval / APPS）"""
    question = str(question)
    question_clean = question[:80].replace('\n', ' ').strip()
    question_suffix = "..." if len(question) > 80 else ""
    return f"""让我仔细分析这个编程问题：

1. 理解需求：题目要求编写一个函数，输入是 {question_clean}{question_suffix}，输出需要满足所有测试用例。
2. 分析关键约束和边界情况：
   - 输入类型和范围（例如列表、整数、正负数、空输入）
   - 特殊情况（如空列表、单一元素、重复元素）
   - 时间/空间复杂度要求（如果有隐含提示）
3. 设计解题思路：
   - 核心算法：排序 / 遍历 / 双指针 / 哈希表 / 递归等
   - 逐步分解：先处理简单情况，再考虑边界
4. 计划实现步骤：
   - 初始化变量
   - 主逻辑循环或递归
   - 返回结果
5. 验证思路：代入测试用例检查正确性，确保不漏边界"""


def process_ape210k():
    """处理 Ape210K 数据集"""
    print("\n处理 Ape210K...")
    base_path = "/root/autodl-tmp/datasets/ape210k-master/data"

    for split in ['train', 'valid', 'test']:
        input_file = f"{base_path}/{split}.ape.json"
        output_file = f"{OUTPUT_DIR}/ape210k_{split}.json"

        # Ape210K 是多行 JSON，每行一个对象
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        processed = []
        for item in tqdm(data, desc=f"Ape210K {split}"):
            question = item.get('original_text', item.get('question', ''))
            answer = str(item.get('ans', item.get('answer', '')))
            equation = str(item.get('equation', ''))

            reasoning = create_ape210k_reasoning(question, equation, answer)
            formatted_answer = f"<reasoning>\n{reasoning}\n</reasoning>\n\n<answer>\n{answer}\n</answer>"

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
    
    try:
        # 使用 datasets 库加载
        dataset = load_dataset("openai/gsm8k", "main", cache_dir="/root/autodl-tmp/datasets")
        
        for split in ['train', 'test']:
            output_file = f"{OUTPUT_DIR}/gsm8k_{split}.json"
            data = dataset[split]

            processed = []
            for item in tqdm(data, desc=f"GSM8K {split}"):
                question = item.get('question', '')
                full_answer = item.get('answer', '')

                # 解析 GSM8K 标准格式：推理过程 #### 最终答案
                if '####' in full_answer:
                    parts = full_answer.split('####')
                    reasoning = parts[0].strip() if len(parts) > 0 else ''
                    final_ans = parts[1].strip() if len(parts) > 1 else ''
                else:
                    reasoning = full_answer
                    final_ans = ''

                # 包装成标签格式（保留原始 reasoning）
                formatted_answer = f"<reasoning>\n{reasoning}\n</reasoning>\n\n<answer>\n{final_ans}\n</answer>"

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
    except Exception as e:
        print(f"❌ GSM8K 加载失败: {e}")
        raise


def process_math():
    """处理 MATH 数据集（代数部分）"""
    print("\n处理 MATH...")
    
    try:
        # 使用 datasets 库加载
        dataset = load_dataset("EleutherAI/hendrycks_math", "algebra", cache_dir="/root/autodl-tmp/datasets")
        
        for split in ['train', 'test']:
            output_file = f"{OUTPUT_DIR}/math_{split}.json"
            data = dataset[split]

            processed = []
            for item in tqdm(data, desc=f"MATH {split}"):
                question = item.get('problem', '')
                solution = item.get('solution', '')

                # 解析 MATH 格式（最后一行通常是 \boxed{答案}）
                lines = solution.strip().split('\n')
                if len(lines) > 1 and '\\boxed' in lines[-1]:
                    reasoning = '\n'.join(lines[:-1]).strip()
                    answer = lines[-1].strip()
                else:
                    # 如果没有明确分隔，尝试查找 \boxed
                    if '\\boxed' in solution:
                        reasoning = solution
                        # 提取最后一个 \boxed{...} 作为答案
                        import re
                        boxed_matches = re.findall(r'\\boxed\{[^}]+\}', solution)
                        answer = boxed_matches[-1] if boxed_matches else solution
                    else:
                        reasoning = solution
                        answer = solution

                # 包装成标签格式
                formatted_answer = f"<reasoning>\n{reasoning}\n</reasoning>\n\n<answer>\n{answer}\n</answer>"

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
    except Exception as e:
        print(f"❌ MATH 加载失败: {e}")
        raise


def process_cmath():
    """处理 CMATH 数据集"""
    print("\n处理 CMATH...")
    
    try:
        # 使用 datasets 库加载
        dataset = load_dataset("weitianwen/cmath", cache_dir="/root/autodl-tmp/datasets")
        
        for split in ['validation', 'test']:
            output_file = f"{OUTPUT_DIR}/cmath_{split}.json"
            data = dataset[split]

            processed = []
            for item in tqdm(data, desc=f"CMATH {split}"):
                question = item.get('question', item.get('problem', ''))
                answer = item.get('golden', item.get('answer', ''))

                # 简单的推理过程
                reasoning = f"让我一步步分析这道数学题：\n\n1. 理解题意\n2. 分析解题思路\n3. 计算并验证"
                formatted_answer = f"<reasoning>\n{reasoning}\n</reasoning>\n\n<answer>\n{answer}\n</answer>"

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
    except Exception as e:
        print(f"❌ CMATH 加载失败: {e}")
        raise


def process_mbpp():
    """处理 MBPP 数据集"""
    print("\n处理 MBPP...")
    
    try:
        # 使用 datasets 库加载
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", cache_dir="/root/autodl-tmp/datasets")
        
        for split in ['train', 'test', 'validation', 'prompt']:
            if split not in dataset:
                print(f"⚠️  MBPP 没有 {split} split")
                continue
                
            output_file = f"{OUTPUT_DIR}/mbpp_{split}.json"
            data = dataset[split]

            processed = []
            reasoning_lengths = []
            for item in tqdm(data, desc=f"MBPP {split}"):
                question = item.get('text', item.get('prompt', ''))
                code = item.get('code', '')
                test_list = item.get('test_list', [])

                # 构建完整的问题描述
                full_question = f"{question}\n\n测试用例：\n" + "\n".join(test_list) if test_list else question

                reasoning = create_code_reasoning(question)
                formatted_answer = f"<reasoning>\n{reasoning}\n</reasoning>\n\n<answer>\n{code}\n</answer>"

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
                reasoning_lengths.append(len(reasoning))

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed, f, ensure_ascii=False, indent=2)

            print(f"✅ MBPP {split}: {len(processed)} 条 -> {output_file}")
            if reasoning_lengths:
                avg_length = sum(reasoning_lengths) / len(reasoning_lengths)
                print(f"   平均reasoning长度: {avg_length:.0f}字符")
    except Exception as e:
        print(f"❌ MBPP 加载失败: {e}")
        raise


def process_humaneval():
    """处理 HumanEval 数据集"""
    print("\n处理 HumanEval...")
    
    try:
        # 使用 datasets 库加载
        dataset = load_dataset("openai/openai_humaneval", cache_dir="/root/autodl-tmp/datasets")
        
        output_file = f"{OUTPUT_DIR}/humaneval_test.json"
        data = dataset['test']

        processed = []
        reasoning_lengths = []
        for item in tqdm(data, desc="HumanEval test"):
            prompt = item.get('prompt', '')
            canonical_solution = item.get('canonical_solution', '')
            test = item.get('test', '')

            # 构建完整的问题描述
            full_question = f"{prompt}\n\n测试用例：\n{test}"

            reasoning = create_code_reasoning(prompt)
            formatted_answer = f"<reasoning>\n{reasoning}\n</reasoning>\n\n<answer>\n{canonical_solution}\n</answer>"

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
            reasoning_lengths.append(len(reasoning))

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        print(f"✅ HumanEval test: {len(processed)} 条 -> {output_file}")
        if reasoning_lengths:
            avg_length = sum(reasoning_lengths) / len(reasoning_lengths)
            print(f"   平均reasoning长度: {avg_length:.0f}字符")
    except Exception as e:
        print(f"❌ HumanEval 加载失败: {e}")
        raise


def process_apps():
    """处理 APPS 数据集（从本地目录加载）"""
    print("\n处理 APPS...")

    try:
        base_path = "/root/autodl-tmp/datasets/APPS"

        for split in ['train', 'test']:
            split_dir = Path(base_path) / split
            if not split_dir.exists():
                print(f"⚠️  目录不存在: {split_dir}")
                continue

            output_file = f"{OUTPUT_DIR}/apps_{split}.json"

            # 获取所有问题目录
            problem_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

            processed = []
            skipped = 0
            reasoning_lengths = []

            for problem_dir in tqdm(problem_dirs, desc=f"APPS {split}"):
                # 读取问题描述
                question_file = problem_dir / "question.txt"
                if not question_file.exists():
                    skipped += 1
                    continue

                with open(question_file, 'r', encoding='utf-8') as f:
                    question = f.read().strip()

                # 读取元信息（difficulty）
                metadata_file = problem_dir / "metadata.json"
                difficulty = "unknown"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as mf:
                            meta = json.load(mf)
                        difficulty = str(meta.get('difficulty', 'unknown')).strip() or "unknown"
                    except Exception:
                        difficulty = "unknown"

                # 读取解决方案
                solutions_file = problem_dir / "solutions.json"
                if not solutions_file.exists():
                    skipped += 1
                    continue

                try:
                    with open(solutions_file, 'r', encoding='utf-8') as f:
                        solutions = json.load(f)

                    # 取第一个解决方案
                    if solutions and len(solutions) > 0:
                        code = solutions[0]
                    else:
                        skipped += 1
                        continue
                except Exception as e:
                    skipped += 1
                    continue

                reasoning = create_code_reasoning(question)
                formatted_answer = f"<reasoning>\n{reasoning}\n</reasoning>\n\n<answer>\n{code}\n</answer>"

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_RECOMMENDED},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": formatted_answer}
                ]

                processed.append({
                    "id": f"apps_{split}_{len(processed)}",
                    "dataset": "apps",
                    "task_type": "code",
                    "difficulty": difficulty,
                    "messages": messages
                })
                reasoning_lengths.append(len(reasoning))

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed, f, ensure_ascii=False, indent=2)

            print(f"✅ APPS {split}: {len(processed)} 条 (跳过 {skipped} 条) -> {output_file}")
            if reasoning_lengths:
                avg_length = sum(reasoning_lengths) / len(reasoning_lengths)
                print(f"   平均reasoning长度: {avg_length:.0f}字符")
    except Exception as e:
        print(f"❌ APPS 加载失败: {e}")
        raise


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

    try:
        process_apps()
    except Exception as e:
        print(f"❌ APPS 处理失败: {e}")

    print("\n" + "="*60)
    print("数据预处理完成！")
    print(f"所有处理后的数据保存在: {OUTPUT_DIR}")
    print("="*60)
