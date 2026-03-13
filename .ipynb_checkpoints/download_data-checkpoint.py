from datasets import load_dataset

# 1. GSM8K（小学数学应用题）
dataset_gsm8k = load_dataset(
  "openai/gsm8k",
  "main",
  cache_dir="/root/autodl-tmp/datasets"
)

# 2. Ape210K（中文小学数学）
dataset_ape210k = load_dataset(
  "m-a-p/Ape210K",
  cache_dir="/root/autodl-tmp/datasets"
)

# 3. MATH（高中竞赛级别数学）
dataset_math = load_dataset(
  "hendrycks/math",
  cache_dir="/root/autodl-tmp/datasets"
)

# 4. CMATH（中文高中数学）
dataset_cmath = load_dataset(
  "weitianwen/cmath",
  cache_dir="/root/autodl-tmp/datasets"
)

# 5. MBPP（Python代码生成 - 基础）
dataset_mbpp = load_dataset(
  "google-research-datasets/mbpp",
  "sanitized",  # 使用清洗过的版本
  cache_dir="/root/autodl-tmp/datasets"
)

# 6. HumanEval（Python代码生成 - 标准benchmark）
dataset_humaneval = load_dataset(
  "openai/openai_humaneval",
  cache_dir="/root/autodl-tmp/datasets"
)

# 7. APPS（算法竞赛题）
dataset_apps = load_dataset(
  "codeparrot/apps",
  cache_dir="/root/autodl-tmp/datasets"
)

# 查看数据集结构
print("=" * 50)
print("GSM8K:", dataset_gsm8k)
print("Ape210K:", dataset_ape210k)
print("MATH:", dataset_math)
print("CMATH:", dataset_cmath)
print("MBPP:", dataset_mbpp)
print("HumanEval:", dataset_humaneval)
print("APPS:", dataset_apps)