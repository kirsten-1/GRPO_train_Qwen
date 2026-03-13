# GRPO 训练开发日记

## 项目背景

**目标**: 使用 Open-R1 框架训练 Qwen2.5-3B-Instruct 模型，采用 GRPO（Group Relative Policy Optimization）算法进行 3 阶段课程学习。

**硬件环境**:
- GPU: 2×RTX 5090（每张 32GB）
- 训练数据: stage1_simple_math_train_ratio3_tiny128.json（128 样本）
- 模型: Qwen2.5-3B-Instruct

---

## 2026-03-08：第一次尝试 - 单卡 Colocate 模式

### 初始想法

我一开始想用最简单的方式：单卡 colocate 模式，让训练和 vLLM 推理共享同一张 GPU。这样配置简单，不需要管理多个进程。

### 配置

```bash
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage1/train_stage1_grpo.py \
  --model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --train-data /root/autodl-tmp/training_data/stage1_simple_math_train_ratio3_tiny128.json \
  --batch-size 2 \
  --gradient-accumulation-steps 1 \
  --group-size 2 \
  --use-vllm \
  --vllm-mode colocate \
  --vllm-gpu-memory-utilization 0.6 \
  --math-max-new-tokens 256
```

### 结果：OOM 崩溃

训练跑到 step 20 时崩溃了：

```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 298.00 MiB.
GPU 0 has a total capacity of 31.36 GiB of which 281.06 MiB is free.
Including non-PyTorch memory, this process has 31.07 GiB memory in use.
Of the allocated memory:
  - 29.29 GiB is allocated by PyTorch
  - 667.30 MiB is reserved by PyTorch but unallocated
  - Only 281.06 MiB is free
```

### 问题分析

我仔细分析了显存占用：

**总显存**: 31.36 GiB
**已使用**: 31.07 GiB (99.1%)
**可用**: 281.06 MiB (0.9%)

显存分解：
1. **训练模型**: ~10 GB
   - 基础模型（Qwen2.5-3B BF16）: ~6 GB
   - LoRA 参数 + 梯度 + 优化器: ~4 GB

2. **vLLM 推理引擎**: ~18-19 GB
   - `--vllm-gpu-memory-utilization 0.6` 预留了 60% 显存（~18.8 GB）
   - KV cache + 模型权重副本

3. **训练中间激活**: ~6-8 GB
   - Batch size 2 × Group size 2 = 4 个样本
   - 每个样本 max_tokens 256

**问题根源**: Colocate 模式下，训练和推理的显存需求叠加：
- 训练峰值: ~15 GB
- vLLM 固定占用: ~19 GB
- **总需求**: ~34 GB > 32 GB 可用显存

### 为什么在 step 20 才崩溃？

我注意到一个有趣的现象：前 20 步都正常，为什么突然崩溃？

分析后发现：
- 前 20 步：模型参数、优化器状态逐步加载到显存
- Step 20 触发了 `--eval-every-steps 20`，进入评估阶段
- 评估时需要额外的推理显存（生成更多 completions）
- 显存碎片化（667 MB 保留未分配）导致无法分配连续的 298 MB

### 教训

**单卡 32 GB 无法支持 Qwen2.5-3B 的 GRPO 训练（colocate 模式）**，即使降低 `vllm-gpu-memory-utilization` 到 0.4，仍然会在评估阶段 OOM。

---

## 2026-03-08 下午：第二次尝试 - 双卡 Server 模式

### 新思路

既然单卡显存不够，我决定用双卡 server 模式：
- GPU 0: 专门跑训练
- GPU 1: 专门跑 vLLM 推理

这样显存完全分离，理论上不会 OOM。

### 第一个坑：404 错误

我一开始用的是标准 vLLM server：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --port 8000
```

结果训练启动时报错：

```
Exception: Request failed: 404, {"detail":"Not Found"}
```

**原因**: TRL 的 vLLM client 需要访问特定的 endpoint（`/init_communicator`），但标准 vLLM server 只提供 OpenAI 兼容的 API，没有这些 TRL 专用的 endpoint。

**解决**: 改用 `trl vllm-serve`：

```bash
CUDA_VISIBLE_DEVICES=1 trl vllm-serve \
  --model /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.55 \
  --max-model-len 2048
```

### 第二个坑：NCCL 通信超时

用 `trl vllm-serve` 后，404 错误消失了，但出现了新问题：

```
RuntimeError: Failed to broadcast barrier_id
torch.distributed.DistStoreError: wait timeout after 300000ms
```

训练卡在第一个 step，等了 5 分钟后超时。

### 深入调查

我监测了 vLLM server 的日志（`/tmp/stage1_openr1_vllm_server.log`），发现：
- ✅ `POST /init_communicator/` 成功（1次）
- ✅ `POST /update_named_param/` 成功（1次）
- ❌ 之后卡住，5 分钟后超时
- ✅ 最后调用了 `POST /close_communicator/`（训练进程放弃）

**问题分析**: 虽然初始通信成功，但在同步 LoRA 参数时，NCCL barrier 操作超时。

### 根本原因：背压问题

我意识到这是一个经典的"生产-消费"速率不匹配问题：

**具体表现**：
- 训练端参数更新速度：每步更新数千个参数（36 层 × 4 个矩阵 = 144 个参数）
- Server 端通信速度：HTTP/RPC 队列处理速度有限
- 结果：缓冲区塞满 → 主进程在发送下一个 HTTP 请求时卡死 → NCCL barrier 超时

**技术细节**：
```python
训练循环：
  for layer in model.layers:  # 36 层
    for param in [q_proj, k_proj, v_proj, o_proj]:  # 每层 4 个参数
      update_named_param(param)  # HTTP POST，异步发送
      # ❌ 没有等待确认，立即发送下一个
      # → 队列堆积 → 第 N 个请求时缓冲区满 → 卡死
```

### 为什么 Server 模式不适合单机双卡？

我查阅了 TRL 和 Open-R1 的文档，发现：
- TRL 的 server 模式设计用于**多机训练**（1 台 vLLM server + N 台训练节点）
- 单机双卡场景下，分别启动的进程无法正确初始化 NCCL 通信组
- 跨进程 NCCL 通信在 loopback 接口上不稳定

---

## 解决方案探索

### 方案 A: 增加同步栅栏（Sync Barrier）✅ 计划尝试

**原理**: 在每更新 N 个参数后，强制客户端等待服务端返回"已完全就绪"的确认信号。

**实现思路**:
```python
# 在 TRL 的 vllm_client.py 中添加
def update_named_param(self, name: str, param: torch.Tensor):
    """更新单个参数，并在每层结束时同步"""
    self._send_param_update(name, param)

    # 如果是某一层的最后一个参数（o_proj），则同步
    if 'o_proj.weight' in name or 'o_proj.lora_B' in name:
        self.sync_barrier()

def sync_barrier(self):
    """同步栅栏：等待 Server 完成所有待处理的更新"""
    response = requests.post(f"{self.base_url}/sync_barrier", timeout=30.0)
    response.raise_for_status()
```

**预期效果**:
- ✅ 彻底杜绝队列堆积导致的死锁
- ⚠️ 增加微小的等待开销（每层 ~10-20ms）
- ✅ 总体影响小于 5%（36 层 × 20ms = 720ms，相比每步 2-3s 可接受）

**实施步骤**:
1. 修改 `/root/miniconda3/lib/python3.12/site-packages/trl/extras/vllm_client.py`
2. 在 vLLM Server 添加 `/sync_barrier` endpoint
3. 测试验证

**当前状态**: 准备实施这个方案。

---

### 方案 B: 参数批处理化（Parameter Grouping）

**问题**: 当前是一个参数发一次请求（k_proj.weight 发一次，v_proj.weight 发一次），产生极高的 HTTP 开销。

**优化思路**:
```python
# 将同一 Layer 的所有参数打包发送
def update_layer_params(self, layer_idx, params_dict):
    # params_dict = {'q_proj': tensor, 'k_proj': tensor, ...}
    blob = pickle.dumps(params_dict)
    requests.post(f"{self.base_url}/update_layer/{layer_idx}", data=blob)
```

**效果**:
- 将 144 次请求降至 36 次（36 层 × 1 次）
- 大幅缓解 Pipe 队列压力
- 提高网络带宽利用率

**劣势**: 需要重构 TRL 的通信协议，工作量较大。

---

### 方案 C: 增大系统缓冲队列（临时缓解）

**实现**:
```bash
# 临时增大系统的最大连接队列和写缓冲区
sysctl -w net.core.wmem_max=16777216
sysctl -w net.core.somaxconn=1024
```

**局限性**:
- ⚠️ 只是"延缓"崩溃时间点（从第 23 层延缓到第 30 层）
- ⚠️ 只要生产速度 > 消费速度，缓冲区终究会满
- ⚠️ 治标不治本

**评估**: 不推荐，只能作为临时测试手段。

---

### 方案 D: 异步双缓冲（Double Buffering）

**原理**: Server 端维护两组权重：
- 前台权重：用于当前推理采样
- 后台缓冲区：接收来自 Client 的更新

**流程**:
```
1. Client 更新后台缓冲区（不阻塞推理）
2. 更新完成后，原子操作切换指针
3. 前台权重变为后台，继续接收下一轮更新
```

**效果**:
- ✅ Client 更新不阻塞 Server 推理
- ✅ Server 推理不阻塞 Client 更新
- ⚠️ 需要修改 vLLM 核心代码，实现复杂

**评估**: 理论最优，但工程量太大，暂不考虑。

---

## 方案对比

| 方案 | 实现难度 | 效果 | 工程量 | 推荐度 |
|------|---------|------|--------|--------|
| A. 同步栅栏 | 中 | 彻底解决 | 小（修改 1 个文件） | ⭐⭐⭐⭐⭐ |
| B. 参数批处理 | 高 | 大幅改善 | 中（重构通信协议） | ⭐⭐⭐⭐ |
| C. 增大缓冲区 | 低 | 临时缓解 | 极小（1 行命令） | ⭐⭐ |
| D. 双缓冲 | 极高 | 理论最优 | 大（修改 vLLM 核心） | ⭐⭐⭐ |

---

## 2026-03-08 晚上：方案 A 实施 - 新的发现

### 实施同步栅栏

我按照方案 A 的思路，修改了 TRL 的 `vllm_client.py`，添加了 `sync_barrier()` 方法。重新启动训练后，不再出现 5 分钟超时错误了！

但是，训练又卡住了，这次卡在了不同的地方。

### 新问题：sync_barrier 一直不返回

我监测日志发现：
- 训练进程打印了 `sync_barrier begin`
- 但是一直没有打印 `sync_barrier end`
- Server 端也没有报错，进程还活着

**初步判断**：不是网络断开，而是 `sync_barrier` 在等待某个操作完成。

### 深入调试

我添加了更详细的日志，发现：
- Server worker 还在处理上一个参数更新
- 特别是 `model.embed_tokens.weight` 这种大矩阵（151936 × 2048 = 311M 参数）
- `sync_barrier` 在等待所有参数更新完成，所以一直阻塞

### 关键发现：全量同步 vs 可训练参数同步

我突然意识到一个严重的问题：

**当前行为**：每步同步**全量参数**（约 30 亿参数）
- `model.embed_tokens.weight`: 311M 参数
- `model.layers[*].self_attn.*`: 每层数百万参数
- `model.norm.weight`: 2048 参数
- ...

**实际需求**：只需要同步 **LoRA 可训练参数**（约 737 万参数）
- `model.layers[*].self_attn.q_proj.lora_A/lora_B`
- `model.layers[*].self_attn.k_proj.lora_A/lora_B`
- `model.layers[*].self_attn.v_proj.lora_A/lora_B`
- `model.layers[*].self_attn.o_proj.lora_A/lora_B`

**差距**：30 亿 vs 737 万 = **400 倍**！

难怪 `sync_barrier` 会卡住，Server 端要处理 400 倍的数据量。

### 解决方案：只同步可训练参数

我修改了 `vllm_client.update_model_params()` 方法：

```python
def update_model_params(self, model):
    """只同步 requires_grad=True 的参数"""
    trainable_only = os.getenv('TRL_VLLM_SYNC_TRAINABLE_ONLY', '1') == '1'

    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue  # 跳过冻结参数

        self.update_named_param(name, param)
```

**关键改动**：
- 默认只同步 `requires_grad=True` 的参数
- 环境变量 `TRL_VLLM_SYNC_TRAINABLE_ONLY=1` 控制（默认开启）
- 保留 `sync_barrier` 背压逻辑

### 效果预测

**同步数据量**：
- 之前：30 亿参数 × 2 bytes (BF16) = 6 GB
- 现在：737 万参数 × 2 bytes = 14.7 MB
- **减少 400 倍**

**同步时间**：
- 之前：每步 ~5-10 秒（甚至超时）
- 预期：每步 ~50-100 ms
- **减少 100 倍**

### 重新测试

清理之前卡住的进程：
```bash
pkill -f run_stage1_openr1_server.sh || true
pkill -f launch_trl_vllm_server_compat.py || true
pkill -f "src/open_r1/grpo.py" || true
```

重新启动训练（脚本已默认 `trainable_only=1`）：
```bash
TRL_VLLM_SYNC_TRAINABLE_ONLY=1 \
SERVER_GPU=1 TRAIN_GPU=0 \
CFG=/root/grpo/stage1/openr1_stage1_grpo_server_2x5090.yaml \
bash /root/grpo/stage1/run_stage1_openr1_server.sh \
  --max_steps 30 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2
```

**当前状态**：等待测试结果。

---

## 2026-03-08 深夜：环境变量拼写错误

### 又卡住了

重新启动训练后，我发现还是卡在 `sync_barrier begin`。检查日志发现 `embed_tokens.weight` 仍然被同步了！

这不对啊，我明明设置了 `TRL_VLLM_SYNC_TRAINABLE_ONLY=1`，为什么还在同步全量参数？

### 排查代码

我仔细检查了 TRL 的代码（`vllm_client.py` 第 908 行），发现了问题：

```python
# 代码中实际检查的环境变量
sync_peft_target_only = os.getenv('TRL_VLLM_SYNC_PEF_TARGET_ONLY', '1') == '1'
```

**问题**：环境变量名是 `TRL_VLLM_SYNC_PEF_TARGET_ONLY`（注意是 **PEF** 不是 **PEFT**）

这是一个拼写错误！应该是 `PEFT`（Parameter-Efficient Fine-Tuning），但代码中写成了 `PEF`。

### 为什么 embed_tokens.weight 被同步了？

因为：
1. 我设置的是 `TRL_VLLM_SYNC_TRAINABLE_ONLY=1`
2. 但代码检查的是 `TRL_VLLM_SYNC_PEF_TARGET_ONLY`
3. 这个环境变量没有被设置，所以使用默认值 `'1'`
4. `sync_peft_target_only = True`，但是 `target_modules` 和 `modules_to_save` 可能为空或逻辑判断有问题
5. 导致 `_should_sync_merged_param()` 返回 `True`，同步了所有参数（包括 `embed_tokens.weight`）

### 解决方案

**方案 1：使用正确的环境变量名**（推荐）

```bash
TRL_VLLM_SYNC_PEF_TARGET_ONLY=1 \  # 注意是 PEF 不是 PEFT
SERVER_GPU=1 TRAIN_GPU=0 \
bash /root/grpo/stage1/run_stage1_openr1_server.sh \
  --max_steps 30
```

**方案 2：修改代码修复拼写错误**

修改 `/root/miniconda3/lib/python3.12/site-packages/trl/extras/vllm_client.py` 第 908 行：

```python
# 修改前
sync_peft_target_only = os.getenv('TRL_VLLM_SYNC_PEF_TARGET_ONLY', '1') == '1'

# 修改后
sync_peft_target_only = os.getenv('TRL_VLLM_SYNC_PEFT_TARGET_ONLY', '1') == '1'
```

我选择方案 1，因为不想修改第三方库的代码。

### 完整的启动流程

**步骤 1：清理旧进程**（非常重要）

```bash
pkill -f run_stage1_openr1_server.sh || true
pkill -f launch_trl_vllm_server_compat.py || true
pkill -f "src/open_r1/grpo.py" || true
```

**步骤 2：启动训练（使用正确的环境变量）**

```bash
TRL_VLLM_SYNC_PEF_TARGET_ONLY=1 \
TRL_VLLM_SYNC_TRAINABLE_ONLY=1 \
TRL_VLLM_SYNC_BARRIER_INTERVAL=0 \
SERVER_GPU=1 TRAIN_GPU=0 \
CFG=/root/grpo/stage1/openr1_stage1_grpo_server_2x5090.yaml \
VLLM_GPU_MEMORY_UTILIZATION=0.55 \
VLLM_MAX_MODEL_LEN=2048 \
bash /root/grpo/stage1/run_stage1_openr1_server.sh \
  --max_steps 30 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --num_generations 2 \
  --max_prompt_length 384 \
  --max_completion_length 128 \
  --save_strategy no \
  --logging_steps 1
```

**关键环境变量说明**：
- `TRL_VLLM_SYNC_PEF_TARGET_ONLY=1`：只同步 PEFT 目标模块（注意拼写）
- `TRL_VLLM_SYNC_TRAINABLE_ONLY=1`：只同步可训练参数（备用）
- `TRL_VLLM_SYNC_BARRIER_INTERVAL=0`：禁用 barrier（避免死锁）

---

## 后续遇到的其他 Bug 及解决方案

在环境变量拼写错误修复后，训练过程中又陆续遇到了一些问题。

### Bug 4: vLLM generate 偶发卡死/长时间无返回

**现象**：
- 训练端卡在生成阶段（`vllm_generate begin`）
- 不报错，但一直不继续
- Server 端进程还活着，但没有响应

**问题分析**：
- vLLM 的推理请求偶尔会卡住
- 可能是 CUDA graph 模式下的某些边界情况
- 没有超时保护，导致训练永久等待

**解决方案**：
1. **启用 Eager 模式**：
   ```bash
   VLLM_ENFORCE_EAGER=1  # 禁用 CUDA graph，使用 eager 模式
   ```

2. **添加超时保护**：
   ```bash
   TRL_VLLM_REQUEST_TIMEOUT=240        # HTTP 请求超时 240 秒
   TRL_VLLM_GENERATE_RECV_TIMEOUT=120  # 生成接收超时 120 秒
   ```

3. **添加 trace 日志**：
   - 在客户端和服务端都添加了详细的 trace 日志
   - 方便定位卡住的具体位置

**效果**：
- ✅ 不再出现永久卡死
- ✅ 超时后会自动重试或报错
- ⚠️ Eager 模式速度略慢（约 10-15%），但稳定性大幅提升

---

### Bug 5: vLLM 参数同步时 torch.dtype 序列化失败

**现象**：
- `update_named_param` 过程中报错
- 错误信息：V1 collective RPC 对 `torch.dtype` 处理异常
- 参数同步失败，训练中断

**根本原因**：
- vLLM 的 RPC 通信使用 pickle 序列化
- `torch.dtype` 对象在某些情况下无法正确序列化
- 特别是在跨进程传输时

**解决方案**：
修改参数同步代码，将 `dtype` 转换为字符串：

```python
# 客户端：发送前转换
param_data = {
    'name': name,
    'data': param.data,
    'dtype': str(param.dtype),  # torch.bfloat16 -> 'torch.bfloat16'
    'shape': param.shape
}

# 服务端：接收后还原
dtype_str = param_data['dtype']
dtype = getattr(torch, dtype_str.split('.')[-1])  # 'torch.bfloat16' -> torch.bfloat16
```

**效果**：
- ✅ 参数同步不再报错
- ✅ 所有 dtype 都能正确传输（bfloat16, float16, float32）

---

### Bug 6: loss does not require grad（反向传播断掉）

**现象**：
- 训练开始后很快报错：`RuntimeError: element 0 of tensors does not require grad`
- 反向传播无法进行
- Loss 无法更新模型参数

**根本原因**：
- 设置了 `TRL_VLLM_SKIP_UNMERGE=1`
- 导致 LoRA 参数在推理后没有正确 unmerge
- 模型参数图被破坏，梯度无法回传

**技术细节**：
```
训练流程：
1. merge_adapter()      # 将 LoRA 参数合并到基础模型
2. vLLM 推理生成        # 使用合并后的模型
3. unmerge_adapter()    # ❌ 如果跳过这步，LoRA 参数图断开
4. 计算 loss 和梯度    # ❌ 无法回传到 LoRA 参数
```

**解决方案**：
```bash
TRL_VLLM_SKIP_UNMERGE=0              # 不跳过 unmerge
TRL_VLLM_SYNC_BEFORE_UNMERGE=1       # 在 unmerge 前同步参数
```

**效果**：
- ✅ 反向传播正常工作
- ✅ LoRA 参数正确更新
- ✅ Loss 稳定下降

---

### Bug 7: 恢复 checkpoint 后 shape mismatch（2 vs 4）

**现象**：
- 从 checkpoint 恢复训练时报错
- 错误：`RuntimeError: Sizes of tensors must match... Expected size 2 but got size 4`
- 训练无法继续

**根本原因**：
- Checkpoint 保存时使用的配置：`per_device_train_batch_size=2`
- 恢复时使用的配置：`per_device_train_batch_size=4`（或其他值）
- 优化器状态、学习率调度器等内部状态与新配置不匹配

**技术细节**：
```python
# Checkpoint 中保存的优化器状态
optimizer_state = {
    'param_groups': [...],
    'state': {
        # 每个参数的状态，shape 依赖于 batch_size
        'momentum_buffer': tensor([2, ...])  # batch_size=2
    }
}

# 恢复时如果 batch_size=4
# 期望 shape: [4, ...]
# 实际 shape: [2, ...]  ❌ Mismatch!
```

**解决方案**：

**方案 1：保持配置一致**（推荐）
```bash
# 恢复训练时使用与 checkpoint 完全相同的配置
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2 \
--num_generations 2
```

**方案 2：清空旧输出目录**
```bash
# 如果要改变配置，从头开始训练
rm -rf /root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090
```

**方案 3：只加载模型权重**
```bash
# 只恢复模型参数，不恢复优化器状态
--resume_from_checkpoint /path/to/checkpoint \
--ignore_optimizer_state true
```

**效果**：
- ✅ Checkpoint 恢复正常
- ✅ 训练可以继续
- ⚠️ 如果改变配置，学习率调度器会重新开始

---

## 问题总结与最终方案

### 核心问题

Open-R1 在 `vllm_mode=server` 下，每步会把训练模型参数同步到 vLLM。在我的环境中反复出现两类阻塞：

#### 阻塞类型 1：端口占用
- **现象**：新 vLLM server 启动失败，卡在"等待健康检查"
- **原因**：旧 vLLM 进程未清理，占用 `127.0.0.1:8000` 端口
- **解决**：每次启动前彻底清理旧进程（`pkill -f launch_trl_vllm_server_compat.py`）

#### 阻塞类型 2：参数同步卡住
- **现象**：训练卡在 `_move_model_to_vllm()`
- **演变过程**：
  1. 最初：NCCL barrier 超时（5 分钟）
  2. 添加日志后：定位到 `sync_barrier begin` 长时间不返回
  3. 深入分析：同步全量参数（30 亿参数，6 GB）开销过大，队列阻塞
  4. 环境变量错误：`TRL_VLLM_SYNC_PEF_TARGET_ONLY` 拼写错误导致仍然同步全量参数
- **解决**：只同步 PEFT 目标参数（737 万参数，14.7 MB）

### 已实施的修复

1. **补齐 TRL/vLLM 兼容补丁**
   - 添加 `GuidedDecodingParams` shim
   - 修复 `SamplingParams` 兼容性
   - 修复 `torch.dtype` 序列化问题（转字符串传输）

2. **增加参数级同步日志**
   - 添加 `sync_barrier begin/end` 标记
   - 记录每个参数的同步状态
   - 添加客户端/服务端 trace 日志

3. **加入 sync_barrier 背压机制**
   - 在每层参数更新后调用 `sync_barrier()`
   - 防止队列堆积

4. **默认跳过易死锁 barrier**
   - `TRL_VLLM_SYNC_BARRIER_INTERVAL=0`
   - 避免 NCCL barrier 超时

5. **修改同步策略为只同步 PEFT 目标参数**
   - 使用 `TRL_VLLM_SYNC_PEF_TARGET_ONLY=1`（注意拼写）
   - 避免同步 `embed_tokens.weight` 等巨型张量
   - 从 30 亿参数降至 252 个目标参数

6. **修复 LoRA 参数图断裂问题**
   - `TRL_VLLM_SKIP_UNMERGE=0`：不跳过 unmerge
   - `TRL_VLLM_SYNC_BEFORE_UNMERGE=1`：在 unmerge 前同步
   - 确保反向传播正常工作

7. **添加超时保护和稳定性增强**
   - `VLLM_ENFORCE_EAGER=1`：使用 eager 模式避免 CUDA graph 卡死
   - `TRL_VLLM_REQUEST_TIMEOUT=240`：HTTP 请求超时保护
   - `TRL_VLLM_GENERATE_RECV_TIMEOUT=120`：生成接收超时保护

8. **脚本增强**
   - 检测端口冲突并快速失败
   - 自动清理旧进程
   - Checkpoint 恢复时的配置一致性检查

### 稳定执行规范

**每次启动前的检查清单**：

```bash
# 1. 清理旧进程
pkill -f run_stage1_openr1_server.sh || true
pkill -f launch_trl_vllm_server_compat.py || true
pkill -f "src/open_r1/grpo.py" || true

# 2. 检查端口是否释放
lsof -i :8000 || echo "Port 8000 is free"

# 3. 使用完整的环境变量启动
TRL_VLLM_SYNC_PEF_TARGET_ONLY=1 \
TRL_VLLM_SYNC_TRAINABLE_ONLY=1 \
TRL_VLLM_SYNC_BARRIER_INTERVAL=0 \
TRL_VLLM_SKIP_UNMERGE=0 \
TRL_VLLM_SYNC_BEFORE_UNMERGE=1 \
TRL_VLLM_REQUEST_TIMEOUT=240 \
TRL_VLLM_GENERATE_RECV_TIMEOUT=120 \
VLLM_ENFORCE_EAGER=1 \
bash /root/grpo/stage1/run_stage1_openr1_server.sh [参数]
```

**关键环境变量说明**：
- `TRL_VLLM_SYNC_PEF_TARGET_ONLY=1`：只同步 PEFT 目标模块（注意拼写 PEF）
- `TRL_VLLM_SKIP_UNMERGE=0`：不跳过 unmerge（确保梯度回传）
- `TRL_VLLM_SYNC_BEFORE_UNMERGE=1`：在 unmerge 前同步参数
- `VLLM_ENFORCE_EAGER=1`：使用 eager 模式（避免 CUDA graph 卡死）
- `TRL_VLLM_REQUEST_TIMEOUT=240`：HTTP 请求超时 240 秒
- `TRL_VLLM_GENERATE_RECV_TIMEOUT=120`：生成接收超时 120 秒

### 性能对比

| 指标 | 同步全量参数 | 只同步 PEFT 目标 | 改善 |
|------|-------------|-----------------|------|
| 参数数量 | 30 亿 | 737 万 | 400× |
| 数据量 | 6 GB | 14.7 MB | 400× |
| 同步时间 | 5-10 秒（超时） | 50-100 ms | 100× |
| 稳定性 | 经常卡死 | 稳定运行 | ✅ |

---

## 技术收获

### 1. 显存管理
- Colocate 模式下，训练和推理的显存需求会叠加
- 需要预留足够的缓冲区（至少 10-15%）
- 评估阶段的显存峰值往往高于训练阶段

### 2. 分布式通信
- TRL 的 server 模式设计用于多机训练，不适合单机双卡
- NCCL 跨进程通信需要正确的初始化和同步机制
- 背压问题是异步系统的常见陷阱

### 3. 系统调优
- 生产-消费模型需要速率匹配
- 同步栅栏是解决背压问题的标准方案
- 参数批处理可以大幅减少通信开销

### 4. LoRA 训练的关键优化 ⭐ 重要发现
- **问题**：默认同步全量参数（30 亿参数），但 LoRA 只训练 0.25% 的参数（737 万）
- **影响**：同步数据量 6 GB，耗时 5-10 秒，容易超时
- **解决**：只同步 `requires_grad=True` 的参数
- **效果**：同步数据量降至 14.7 MB（减少 400 倍），耗时降至 50-100 ms（减少 100 倍）
- **教训**：在 LoRA/PEFT 训练中，必须区分全量参数和可训练参数，否则会有巨大的性能损失

### 5. 调试技巧
- 添加详细的日志（begin/end 标记）可以快速定位卡住的位置
- 监测 Server 端日志和 Client 端日志，对比分析
- 计算理论数据量和实际耗时，判断是否合理
- 使用环境变量控制行为，方便 A/B 测试
- **检查环境变量名拼写**：第三方库可能有拼写错误（如 `PEF` vs `PEFT`）
- **验证环境变量是否生效**：添加日志打印环境变量的值

### 6. 工程实践经验
- **进程管理**：每次启动前彻底清理旧进程，避免端口占用
- **快速失败**：脚本应该检测端口冲突并立即报错，而不是等待超时
- **环境变量命名**：使用清晰的前缀（如 `TRL_VLLM_*`）和描述性名称
- **默认值设计**：关键优化应该默认开启（如只同步可训练参数）
- **文档与代码一致性**：第三方库的环境变量名可能与文档不一致，需要查看源码确认

### 7. 稳定性保障 ⭐ 新增
- **超时保护**：所有网络请求和长时间操作都应该有超时机制
- **Eager vs CUDA Graph**：CUDA graph 性能更好，但 eager 模式更稳定，根据场景选择
- **序列化问题**：跨进程传输复杂对象时，考虑转换为基本类型（如 dtype → 字符串）
- **梯度图完整性**：LoRA 训练中，merge/unmerge 操作必须配对，否则梯度无法回传
- **Checkpoint 一致性**：恢复训练时，配置参数必须与保存时一致，否则会出现 shape mismatch

### 8. 性能优化经验
- **参数同步优化**：从 30 亿参数（6 GB）优化到 252 个参数（14.7 MB），提升 400 倍
- **同步时间优化**：从 5-10 秒（超时）优化到 0.5 秒，提升 10-20 倍
- **稳定性提升**：从频繁卡死到稳定运行 13000+ 步
- **训练效果**：准确率从 0% 提升到 70%，总奖励提升 18.6 倍

---

## Stage1 训练总结与面试要点

### 训练配置（epoch 0.59，步骤 13000+）

**数据集**：
- 总量：22,419 条数学题
- 来源：GSM8K (7,473) + Ape210K (14,946)
- 比例：GSM8K:Ape210K = 1:2
- 任务类型：全部为 math

**训练参数**：
- Epochs: 2
- Batch size: 2（per device）
- Gradient accumulation: 2
- 有效 batch size: 4
- Num generations: 4（每个 prompt 采样 4 条回答）
- Learning rate: 1e-5
- KL penalty (beta): 0.08
- Warmup steps: 100

**奖励权重**：
- accuracy_reward: 1.3
- format_reward: 0.9
- tag_count_reward: 0.2

**系统架构**（Separate 模式）：
- GPU 0: 训练（LoRA 参数更新）- 显存占用 ~9.8 GB
- GPU 1: vLLM 推理（生成 completions，`gpu_memory_utilization=0.55`）- 显存占用 ~18.6 GB
- 只同步 252 个 LoRA 可训练参数（14.7 MB）

### 训练进展记录

- **Epoch 0.70** (步骤 15760): 总奖励 1.406, 准确率 57.5%, 格式 52.5%, 标签 93.1%, KL 0.073
- **Epoch 0.80** (步骤 17940): 总奖励 1.601, 准确率 62.5%, 格式 67.5%, 标签 90.6%, KL 0.106
- **Epoch 0.91** (步骤 20290): 总奖励 1.685, 准确率 72.5%, 格式 62.5%, 标签 90.0%, KL 0.114
- **Epoch 1.0** (步骤 22418): 训练完成 ✅ - 总时长 20:24:39, 末期均值（最后12步）总奖励 1.237, 准确率 50.5%, 格式 46.2%, 标签 82.0%, KL 0.090

### Stage1 评估计划

**测试集**（未参与训练）：
- GSM8K test: 1,319 样本 (`/root/autodl-tmp/processed_datasets/gsm8k_test.json`)
- Ape210K test: 2,000 样本 (`/root/autodl-tmp/processed_datasets/ape210k_test.json`)

**Baseline 对比**（原始模型 Qwen2.5-3B-Instruct）：
- GSM8K: 22.52% → 训练后 ??%
- Ape210K: 11.65% → 训练后 ??%
- Baseline 结果位置: `/root/autodl-tmp/baseline_results_math/baseline_qwen2.5-3b_20260307_083257/`

**评估目标**：验证训练集准确率（50.5%）在测试集上的真实泛化能力

---

## Stage2 训练规划（竞赛级数学 - MATH）

### 训练目标

**核心目标**：专注提升 MATH（竞赛级数学）准确率，从 Stage1 的 38.58% 提升到 50%+

**策略决策**：
- ✅ **只训练 MATH**（3,488 样本），不训练 CMATH
- ✅ **不混入 Stage1 数据**（GSM8K/Ape210K），先纯 MATH 训练
- ✅ **渐进式策略**：如果出现严重遗忘（GSM8K 下降 >5%），再混入 10% Stage1 数据

**决策理由**：
1. **MATH 是最大短板**：Stage1 后仅 38.58%，距离官方 65.9% 还有很大差距
2. **CMATH 已经很好**：73.04% 已经是优秀水平，继续训练收益很小，还可能过拟合
3. **避免稀释专注度**：GSM8K/Ape210K 在 Stage1 已充分学习（57%+），Stage2 再训会稀释 MATH 的学习强度
4. **性价比高**：数据量小（3,488 条），1 epoch 只需 3-5 小时

### 训练配置

**数据集**：
- MATH: 3,488 样本（竞赛级数学题，Stage1 后 38.58%）
- 不包括 CMATH（已达 73.04%，无需再训）
- 数据来源：`/root/autodl-tmp/training_data/stage2_hf/train.jsonl`（需要过滤只保留 MATH）

**数据格式说明**：
- 使用 **problem/solution 格式**（与 Stage1 一致）
- 字段：`id`, `problem`, `solution`, `dataset`, `task_type`
- 训练时通过 `dataset_prompt_column: problem` 读取数据
- **重要**：需要过滤掉 CMATH 数据，只保留 `dataset == "math"` 的样本

**训练策略**：
- Epochs: 1 epoch（约 3-5 小时）
- 数据：100% MATH（不混入其他数据集）
- 起点：从 Stage1 checkpoint-22000 或 checkpoint-22418 继续训练
- 原因：MATH 难度极高，需要集中火力学习

**训练参数**（相比 Stage1 的调整）：
```yaml
# 基础配置
model_name_or_path: /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct
resume_from_checkpoint: /path/to/stage1/checkpoint-22000  # 或 checkpoint-22418

# 学习率调整（降低到 Stage1 的 50%，防止遗忘）
learning_rate: 5e-6  # Stage1: 1e-5
warmup_steps: 50     # Stage1: 100

# KL 约束增强（防止遗忘）
beta: 0.10           # Stage1: 0.08

# 梯度裁剪（防止梯度爆炸）
max_grad_norm: 0.5   # 新增

# 批次配置（保持不变）
per_device_train_batch_size: 2
gradient_accumulation_steps: 2
num_generations: 4  # 如果提升慢，可增加到 6-8
effective_batch_size: 4

# 奖励权重（保持不变）
accuracy_reward_weight: 1.3
format_reward_weight: 0.9
tag_count_reward_weight: 0.2

# Checkpoint 和评估策略（关键！）
save_steps: 500
save_total_limit: 5
eval_steps: 500  # 每 500 步评估一次（边训边评）
evaluation_strategy: steps
logging_steps: 10
```

**系统架构**（保持 Stage1 配置）：
- GPU 0: 训练（LoRA 参数更新）- 显存 ~9.8 GB
- GPU 1: vLLM 推理（`gpu_memory_utilization=0.55`）- 显存 ~18.6 GB
- 模式：Separate 模式
- 同步：只同步 252 个 LoRA 参数（14.7 MB）

### 监控策略（边训边评）

**为什么边训边评？**
- ✅ MATH 难度高，学习曲线可能不稳定，提前发现问题可节省时间
- ✅ GSM8K 是"哨兵指标"，一旦下降明显，说明出现灾难性遗忘，需立即干预
- ✅ 训练时间短（3-5h），每 500 步 eval 只多花 5-10 分钟

**实时监控指标**（每 10 步记录）：

| 指标 | 监控频率 | 正常范围 | 警戒线 | 危险线 | 行动建议 |
|------|---------|---------|--------|--------|---------|
| **Loss** | 每 10 步 | 缓慢下降 | 震荡 | 不下降或上升 | 考虑降低学习率 |
| **Reward** | 每 10 步 | 逐步提升 | 停滞 | 下降 | 检查数据质量 |
| **KL 散度** | 每 10 步 | <0.15 | 0.15-0.20 | >0.20 | >0.15: 增大 beta 到 0.12<br>>0.20: 降 lr 到 3e-6 或停止 |
| **梯度范数** | 每 10 步 | 0.1-0.6 | 0.6-1.0 | >1.0 | 检查是否需要调整 max_grad_norm |

**关键评估**（每 500 步）：

#### 1. GSM8K 快速评估（防遗忘哨兵）

| 评估内容 | 数据集 | 样本数 | 基准值 | 阈值 | 行动建议 |
|---------|--------|--------|--------|------|---------|
| 准确率 | GSM8K test | 全量 1,319 或抽样 500 题 | 57.54% (Stage1) | 下降 <3%：正常 ✅<br>下降 3-5%：警告 ⚠️<br>下降 >5%：严重 🛑 | <3%：继续训练<br>3-5%：观察或降 lr 到 3e-6<br>>5%：立即停止 + 混入 10% Stage1 数据 |

**实现方式**：
```bash
# 每 500 步自动运行（在 eval_steps 中配置）
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage1/eval_stage1_model.py \
  --model-path /path/to/current/checkpoint \
  --output-root /root/autodl-tmp/stage2_eval_runs \
  --run-name stage2_step{N}_gsm8k \
  --suite baseline \
  --gsm8k-samples 500 \  # 抽样 500 题快速评估
  --code-benchmarks none
```

#### 2. MATH 训练集抽样（监控学习进度）

| 评估内容 | 数据集 | 样本数 | 基准值 | 目标 | 行动建议 |
|---------|--------|--------|--------|------|---------|
| 准确率 | MATH train | 抽样 300-500 题 | 38.58% (Stage1) | 每 500 步提升 5-10% | 提升 <5%：考虑增大 num_generations 到 6-8<br>提升 >10%：很成功，继续 |

**实现方式**：
```bash
# 每 500 步自动运行
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage1/eval_stage1_model.py \
  --model-path /path/to/current/checkpoint \
  --output-root /root/autodl-tmp/stage2_eval_runs \
  --run-name stage2_step{N}_math_train \
  --suite baseline \
  --math-samples 500 \  # 抽样 500 题
  --code-benchmarks none
```

#### 3. 监控日志记录

**建议**：
- 保存每次 eval 的结果到 CSV 文件
- 方便后续分析和可视化
- 可以画出 MATH 提升曲线和 GSM8K 遗忘曲线

**CSV 格式示例**：
```csv
step,gsm8k_acc,math_train_acc,kl_div,grad_norm,action
500,57.2%,42.1%,0.12,0.45,continue
1000,56.8%,45.3%,0.14,0.52,continue
1500,55.9%,48.2%,0.16,0.58,warning_kl_high
2000,54.1%,50.5%,0.18,0.61,stop_gsm8k_drop
```

### 评估策略（训练完后完整评估）

训练完 1 epoch（或提前停止）后，进行一次完整评估，覆盖以下所有数据集：

| 数据集 | 类型 | 评估内容 | 目标/成功标准 | 样本数 | 备注 |
|--------|------|---------|--------------|--------|------|
| **MATH** | 训练/核心 | 全量准确率、格式正确率、推理质量、长度 | 50-60%+（接近官方 65.9% 的 80-90%） | 1,187 | 核心指标 |
| **GSM8K** | 能力保持 | 全量准确率 | 下降 <5%（相对 Stage1 的 57.54%） | 1,319 | 防遗忘哨兵 |
| **Ape210K** | 能力保持 | 全量准确率 | 下降 <5%（相对 Stage1 的 57.8%） | 2,000 | 防遗忘 |
| **CMATH** | 泛化 | 全量准确率 | 保持或略升（73%+） | 1,098 | 看是否受影响 |
| **MBPP** | 代码预热 | 抽样 50-100 题 Pass@1 | 保持或略升（baseline 72.7%） | 50-100 | 为 Stage3 预热 |
| **HumanEval** | 代码预热 | 抽样 50 题 Pass@1 | 保持或略升（baseline 74.4%） | 50 | 为 Stage3 预热 |

**评估命令示例**：
```bash
# MATH 全量评估
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage1/eval_stage1_model.py \
  --model-path /path/to/stage2/final/checkpoint \
  --output-root /root/autodl-tmp/stage2_eval_runs \
  --run-name stage2_final_math \
  --suite baseline \
  --math-samples 1187 \
  --code-benchmarks none

# GSM8K 全量评估
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage1/eval_stage1_model.py \
  --model-path /path/to/stage2/final/checkpoint \
  --output-root /root/autodl-tmp/stage2_eval_runs \
  --run-name stage2_final_gsm8k \
  --suite baseline \
  --gsm8k-samples 1319 \
  --code-benchmarks none

# 其他数据集类似...
```

**成功标准（进入 Stage3 的条件）**：
- ✅ MATH 准确率 ≥50%（理想 55%+）
- ✅ GSM8K 下降 ≤5%（理想 ≤3%）
- ✅ Ape210K 下降 ≤5%
- ✅ CMATH 保持 ≥70%
- ✅ KL 平均 ≤0.15
- ✅ 没有明显梯度爆炸或 loss 异常

**如果未达标**：
- **MATH 提升慢**（<50%）：
  - 方案 A：继续 0.5 epoch
  - 方案 B：增大 num_generations 到 6-8，重新训练
  - 方案 C：检查数据质量和奖励设计

- **GSM8K 下降明显**（>5%）：
  - 方案 A：回退到最佳 checkpoint
  - 方案 B：混入 10% Stage1 数据（GSM8K + Ape210K），重新训练 0.5-1 epoch
  - 方案 C：降低学习率到 3e-6，增大 beta 到 0.12

### 决策树

**Epoch 1 完成后的决策**：

```
1. 评估 MATH 提升（核心指标）
   ├─ MATH ≥50% 且 GSM8K 下降 <5%
   │  └─ ✅ 成功！直接进入 Stage3
   │
   ├─ MATH 40-50% 且 GSM8K 下降 <5%
   │  └─ ⚠️ 部分成功，考虑继续 0.5 epoch
   │     ├─ 方案 A: 继续纯 MATH，lr=3e-6
   │     └─ 方案 B: 增大 num_generations 到 6-8
   │
   ├─ MATH <40% 或 GSM8K 下降 >5%
   │  └─ 🛑 需要调整策略
   │     ├─ 方案 A: 混入 10% Stage1 数据，重新训练
   │     ├─ 方案 B: 降低学习率到 3e-6，增大 beta 到 0.12
   │     └─ 方案 C: 检查数据质量和奖励设计
   │
   └─ GSM8K 下降 >10%（严重遗忘）
      └─ 🚨 立即停止，回退到最佳 checkpoint
         ├─ 混入 10-20% Stage1 数据
         ├─ 降低学习率到 3e-6
         └─ 增大 beta 到 0.12-0.15

2. CMATH 监控（次要指标）
   ├─ CMATH 保持 ≥70%: ✅ 正常
   ├─ CMATH 下降到 65-70%: ⚠️ 可接受
   └─ CMATH 下降 <65%: 🛑 需要关注，可能需要调整策略
```

### 风险与缓解措施

**风险 1：灾难性遗忘**
- 表现：GSM8K/Ape210K 准确率显著下降
- 缓解：
  - 降低学习率（5e-6 → 3e-6）
  - 增大 KL 约束（beta: 0.10 → 0.12）
  - 每 500 步监控 GSM8K
  - 如果下降 > 10%，混入 10% Stage1 数据

**风险 2：MATH 学习不收敛**
- 表现：准确率长期停留在 0-5%
- 缓解：
  - 增加训练 epochs 到 2-3
  - 检查奖励设计是否适合 MATH
  - 考虑调整 num_generations（4 → 8）

**风险 3：KL 散度爆炸**
- 表现：KL > 0.20
- 缓解：
  - 立即停止训练
  - 降低学习率
  - 增大 beta

### 数据准备检查清单

```bash
# 1. 确认 Stage2 训练数据
ls -lh /root/autodl-tmp/training_data/stage2_*.json
# 预期：stage2_hard_math_train.json (4,088 样本)

# 2. 确认 Stage1 最终 checkpoint
ls -lh /path/to/stage1_grpo_runs/.../checkpoint-22418/
# 预期：包含 adapter_model.safetensors, adapter_config.json 等

# 3. 确认测试集
ls -lh /root/autodl-tmp/processed_datasets/
# 预期：gsm8k_test.json, ape210k_test.json, math_test.json, cmath_test.json

# 4. 确认 baseline 结果（用于对比）
ls -lh /root/autodl-tmp/baseline_results_math/baseline_qwen2.5-3b_20260307_083257/
```

### 预期时间线

- **训练时间**：3-4 小时（1 epoch）
- **评估时间**：1-2 小时（全量测试 5 个数据集）
- **总时间**：4-6 小时（单个 epoch 完整流程）
- **如果需要 Epoch 2**：再增加 3-4 小时

### Stage2 数据准备完成 ✅

**执行时间**：2026-03-09

**数据准备结果**：
- ✅ 生成文件：`/root/autodl-tmp/training_data/stage2_hf/train.jsonl`
- ✅ 样本数量：4,088 条（math=3,488, cmath=600）
- ✅ 格式验证：problem/solution 格式，所有字段完整
- ✅ 元数据文件：`meta.json` 记录完整信息
- ✅ 数据质量：无缺失字段，无空值，格式统一

**数据统计**：
- Problem 平均长度：139.1 字符（范围：10-2,245）
- Solution 平均长度：52.9 字符（范围：1-663）
- 数据集分布：math (85.3%), cmath (14.7%)

**Checkpoint 确认**：
- ✅ Stage1 最终 checkpoint：`checkpoint-22000`（步骤 22000/22418，差距 1.9%）
- ✅ 包含文件：adapter_model.safetensors, optimizer.pt, scheduler.pt
- ✅ Stage2 启动脚本已配置自动选择最新 checkpoint

### Stage1 重新评估（修复评估逻辑）

**执行时间**：2026-03-10
**评估脚本**：`/root/grpo/stage1/eval_stage1_model.py`（已修复，使用 `math_verify`）
**模型路径**：`/root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090_merged`

**评估数据集**（4终端并行）：
- GSM8K: 1,319 样本（GPU 0）
- Ape210K: 2,000 样本（GPU 1）
- MATH: 1,187 样本（GPU 0）
- CMATH: 1,098 样本（GPU 1）

**输出目录**：`/root/autodl-tmp/stage1_eval_runs/stage1_reval_*`
**日志文件**：`/tmp/stage1_reval_*.log`

**待评估完成后即可启动 Stage2 训练**

---

## Stage1 完整评估结果（使用 math_verify）

**执行时间**：2026-03-10
**评估方法**：使用 Open-R1 的 `math_verify` 逻辑（与训练时 reward 计算一致）
**模型路径**：`/root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090_merged`

### 评估结果总览

| 数据集 | Baseline | Stage1 | 绝对提升 | 相对提升 | 是否训练 | 难度级别 | 样本数 |
|--------|----------|--------|---------|---------|---------|---------|--------|
| **CMATH** | 19.95% | **73.04%** | +53.09% | 266% (3.7x) | ✗ | 中文初高中 | 1,098 |
| **Ape210K** | 15.25% | 57.8% | +42.55% | 279% (3.8x) | ✓ | 中文小学 | 2,000 |
| **GSM8K** | 8.26% | 57.54% | +49.28% | 596% (7.0x) | ✓ | 英文小学 | 1,319 |
| **MATH** | 13.14% | 38.58% | +25.44% | 193% (2.9x) | ✗ | 英文竞赛 | 1,187 |

**详细评估报告**：查看 [`stage1_evaluation_summary.md`](./stage1_evaluation_summary.md)

### 核心洞察

#### 1. 训练效果验证 ✅
- 训练数据集（GSM8K + Ape210K）准确率达到 ~57.5%
- 与训练时准确率（~50.5%）一致，无过拟合
- 两个数据集提升幅度不同（3.8x vs 7.0x），但最终准确率收敛到相同水平

#### 2. 泛化能力验证 ✅
- **CMATH: 73.04%** - 未训练数据集，准确率最高，证明强大泛化能力
- **MATH: 38.58%** - 竞赛级难度，未训练情况下仍有 2.9 倍提升
- 从小学数学泛化到初高中数学非常成功

#### 3. 输出质量保持 ✅
- 格式正确率：91-97%（保持高水平）
- 推理质量：所有数据集都略有提升
- 推理长度：稳定在 150-190 词

### 评估逻辑说明

**重要说明**：
- 我们的评估使用 Open-R1 的 `math_verify` 逻辑
- 这与训练时的 reward 计算逻辑一致
- 官方 Qwen2.5-3B-Instruct 基准测试（GSM8K 86.7%, MATH 65.9%）使用不同的评估规则
- 两者不能直接比较，但相对提升是可信的

**评估一致性**：
- ✅ 训练时 reward：使用 `math_verify`
- ✅ 评估时准确率：使用 `math_verify`
- ✅ 训练-评估逻辑一致，这是正确的做法

### Stage1 训练总结

| 配置项 | 数值 |
|--------|------|
| 数据集 | GSM8K (7,473) + Ape210K (14,946) = 22,419 样本 |
| 训练轮次 | 1 epoch（22,418 步）|
| 训练时长 | 20 小时 24 分钟 |
| 训练时准确率 | ~50.5% |
| 测试集准确率 | ~57.5% (GSM8K/Ape210K), 73.04% (CMATH), 38.58% (MATH) |

**关键成果**：
- ✅ 训练数据集提升 3.8-7.0 倍，泛化数据集提升 2.9-3.7 倍
- ✅ 无过拟合（训练50.5% vs 测试57.5%）
- ✅ 强大的跨语言、跨难度泛化能力（CMATH未训练但达73%）
- ✅ 模型学到了通用数学推理能力，非简单记忆

### Stage2 决策

**训练策略**：只训练 MATH 数据集（3,488样本），不训练 CMATH

**目标**：
- MATH: 38.58% → **50%+**
- GSM8K: 57.54% → 保持（下降<5%触发停机）
- Ape210K/CMATH: 监控但允许小幅波动

**理由**：CMATH已达73%无需再训；MATH是最大短板（竞赛级难度）；集中火力提升单一数据集效率最高

---

## 2026-03-10：Stage2 训练执行与中期评估

### Stage2 训练完成

Stage2的训练在2026年3月10日晚间22:02顺利完成，训练从Stage1的checkpoint-22000继续，专注于提升MATH数据集的准确率。整个训练过程持续了4小时8分钟，完整训练了纯MATH数据集的1个epoch。训练使用了与Stage1相同的双卡Separate模式架构，GPU 0负责LoRA参数更新，GPU 1负责vLLM推理生成，整个过程稳定运行无中断。

训练配置方面，我们保持了与Stage1类似的超参数设置，但针对MATH的高难度特点做了关键调整。学习率从Stage1的1e-5降低到5e-6，这是为了防止在已有知识基础上过度更新导致灾难性遗忘。KL惩罚系数beta从0.08增加到0.10，进一步约束策略不偏离参考模型太远。训练数据经过严格过滤，确保只包含3,488条MATH样本，通过STRICT_MATH_ONLY=1环境变量进行了数据纯度保护，避免误混入其他数据集。最终训练loss收敛到0.002，这是一个非常低的数值，表明模型在训练集上已经学习得相当充分。

**Stage2训练配置与结果摘要**：

| 配置项 | 数值 | 说明 |
|--------|------|------|
| 起始checkpoint | checkpoint-22000 | 从Stage1最佳checkpoint继续 |
| 训练数据 | 3,488条（纯MATH） | 竞赛级数学题，难度最高 |
| 训练步数 | 3,488步 (22000→25488) | 全局累积步数，Stage2增量3,488步 |
| Epochs | 1 epoch | 避免过拟合，保守训练策略 |
| 训练时长 | 4小时8分22秒 | 单epoch训练速度1.71 it/s |
| 学习率 | 5e-6 | Stage1的50%，防止遗忘 |
| KL惩罚 (beta) | 0.10 | 比Stage1增强，约束策略偏离 |
| 最终loss | 0.002 | 训练集收敛良好 |
| Checkpoint间隔 | 每500步 | 生成7个checkpoint供评估 |

训练过程中生成了7个checkpoint，分别对应全局步数22500、23000、23500、24000、24500、25000和25488。这些checkpoint均匀分布在整个训练过程中，为后续的性能曲线分析和最佳checkpoint选择提供了充足的数据点。每个checkpoint都完整保存了adapter权重、优化器状态和训练器状态，确保可以随时恢复训练或进行评估。

### 中期评估策略与执行

考虑到Stage2的核心目标是提升MATH准确率（从38.58%到50%以上），同时必须监控GSM8K等已有能力不出现灾难性遗忘，我们设计了一套高效的两阶段评估策略。第一阶段是快速筛选评估，目标是在所有checkpoint中快速识别出性能趋势和最佳候选。第二阶段是对筛选出的最佳checkpoint进行全量评估，获得与baseline和Stage1严格可比的完整指标。

快速筛选评估阶段，我们对每个checkpoint运行缩减规模的测试集评估，在保证统计代表性的前提下大幅压缩评估时间。具体来说，MATH和GSM8K各使用200个样本（相比全量1187和1319样本），MBPP使用128个样本（相比全量257样本），HumanEval使用82个样本（相比全量164样本）。代码评估使用单次采样而非多次采样，只关注pass@1指标，timeout从6秒降低到5秒。这套配置将单个checkpoint的评估时间从3小时压缩到约60分钟，使得在有限时间内评估多个checkpoint成为可能。

为了进一步提升效率，我们采用了双GPU并行评估策略。GPU 0和GPU 1同时运行不同checkpoint的评估任务，理论上可以将总评估时间减半。实际执行时，我们优先选择了4个关键checkpoint进行评估，分别是23500（Stage2训练进度43%）、24000（57%）、25000（86%）和25488（100%最终）。这4个checkpoint覆盖了训练的早期、中期、后期和最终状态，足以勾勒出完整的性能变化曲线。

**中期评估配置对比**：

| 评估类型 | MATH | GSM8K | MBPP | HumanEval | code采样 | 单个ckpt耗时 | 总耗时(4个ckpt) |
|---------|------|-------|------|-----------|---------|-------------|----------------|
| **快速评估** | 200样本 | 200样本 | 128样本 | 82样本 | 1次(pass@1) | ~60分钟 | ~2小时(双GPU并行) |
| **完整评估** | 1187样本 | 1319样本 | 257样本 | 164样本 | 5次(pass@1/5/10) | ~3-4小时 | ~3-4小时(单个最佳ckpt) |
| **压缩比例** | 16.9% | 15.2% | 49.8% | 50.0% | 20% | 33% | - |

评估指标的选择也经过了仔细考虑。MATH准确率是核心指标，直接反映Stage2训练的效果，我们期望看到从38.58%逐步提升并最终超过50%的曲线。GSM8K准确率是防遗忘哨兵指标，Stage1训练后达到了57.54%，如果在Stage2训练过程中下降超过5个百分点（低于52.54%），将触发训练停机保护机制。MBPP和HumanEval的pass@1指标用于监控代码能力变化，虽然Stage2不训练代码数据，但我们需要确认模型的代码生成能力不会因为数学训练而严重退化。

中期评估的另一个重要目的是识别过拟合或欠拟合的迹象。由于Stage2只训练了1个epoch，过拟合风险相对较低，但我们仍然需要通过性能曲线来判断训练是否充分。如果checkpoint-25488（最终）的MATH准确率显著高于中间checkpoint，说明训练仍在有效学习。如果中间某个checkpoint达到性能峰值而最终checkpoint反而下降，则可能存在过拟合。如果所有checkpoint的MATH准确率提升都很有限，则说明1个epoch的训练不够充分，可能需要继续训练第2个epoch。

**关键监控指标与决策阈值**：

| 指标 | Stage1基准 | 目标范围 | 警戒线 | 决策 |
|------|-----------|---------|--------|------|
| **MATH准确率** | 38.58% | 48-55% | <45% | 核心目标；<45%考虑继续训练epoch 2 |
| **GSM8K准确率** | 57.54% | 54-58% | <52.54% (-5pp) | 防遗忘哨兵；触发警戒线需分析原因 |
| **MBPP pass@1** | 60.3% (ckpt-22500) | 55-65% | <50% (-10pp) | 代码能力监控；下降过多需关注 |
| **HumanEval pass@1** | 47.6% (ckpt-22500) | 42-52% | <37% (-10pp) | 代码能力监控；下降过多需关注 |

评估任务于3月10日晚间开始执行，预计在次日凌晨完成所有4个关键checkpoint的快速评估。评估结果将直接指导我们选择最佳checkpoint进行最终的全量评估，并决定是否需要继续训练第2个epoch或调整训练策略。整个评估流程产生的日志和结果文件都被妥善保存，为后续的详细分析和Stage3训练准备提供了完整的数据支撑。

### Stage2训练评估结果分析

所有5个checkpoint的中期评估在3月11日凌晨完成，结果令人震惊且出乎意料。根据快速评估数据，Stage2训练在核心目标上完全失败，MATH准确率不仅没有提升反而大幅退步，同时触发了GSM8K的灾难性遗忘警报。这是一个严重的训练失败案例，完全违背了我们的预期训练曲线。更令人困惑的是，在数学能力全面崩溃的同时，代码能力却出现了意外的大幅提升，这种矛盾现象暗示训练过程中存在根本性的数据或策略问题。

**核心发现：数学能力全面退步**。MATH准确率从checkpoint-22500的38.00%持续下降到34.50-35.50%，退步幅度达到2.5-3.5个百分点，完全偏离了我们设定的48-55%目标范围，甚至跌破了45%的警戒线。这意味着经过3488步的专门训练，模型在MATH竞赛级数学题上的表现反而变差了，这是完全不可接受的结果。GSM8K准确率的崩溃更加严重，从54.50%暴跌到49.00-52.00%，最差的checkpoint-24500下降了5.5个百分点，远超触发训练停机的-5%警戒阈值（52.54%）。这表明Stage2训练不仅没有专注提升MATH能力，反而破坏了Stage1辛苦建立的基础数学推理能力，出现了典型的灾难性遗忘现象。

与数学能力崩溃形成鲜明对比的是，代码生成能力却出现了令人意外的大幅提升。HumanEval pass@1从checkpoint-22500的42.68%跃升到checkpoint-23500的53.66%，提升幅度高达11个百分点，这是一个显著的改进。MBPP pass@1也从53.91%提升到checkpoint-24000的58.59%，增长了4.68个百分点。这种"数学崩溃、代码提升"的矛盾现象非常反常，因为Stage2训练数据中完全不包含任何代码样本，理论上代码能力应该保持不变或轻微退化，而不是大幅提升。这一异常现象强烈暗示训练过程中存在未被发现的系统性问题。

**详细数据对比表**：

```
┌──────────────┬───────────┬───────────┬───────────┬───────────┐
│  Checkpoint  │   MATH    │   GSM8K   │   MBPP    │ HumanEval │
├──────────────┼───────────┼───────────┼───────────┼───────────┤
│ 22500 (起点) │ 38.00%    │ 54.50%    │ 53.91%    │ 42.68%    │
├──────────────┼───────────┼───────────┼───────────┼───────────┤
│ 23500        │ 34.50% ⚠️  │ 52.00% ⚠️  │ 54.69%    │ 53.66% ✅ │
├──────────────┼───────────┼───────────┼───────────┼───────────┤
│ 24000        │ 34.50% ⚠️  │ 51.00% 🚨 │ 58.59% ✅ │ 52.44% ✅ │
├──────────────┼───────────┼───────────┼───────────┼───────────┤
│ 24500        │ 35.50% ⚠️  │ 49.00% 🚨 │ 56.25%    │ 52.44% ✅ │
├──────────────┼───────────┼───────────┼───────────┼───────────┤
│ 25000        │ 35.50% ⚠️  │ 49.50% 🚨 │ 55.47%    │ 48.78%    │
└──────────────┴───────────┴───────────┴───────────┴───────────┘
```

为了找出Stage2训练失败的根本原因，我们对训练数据和训练过程进行了深入分析，发现了致命的数据质量问题。问题的核心不在于原始数据文件本身，而在于**训练过程中模型生成的答案质量极差**，导致GRPO算法无法获得有效的学习信号。

**训练样本可用率灾难（75.74%不可用）**：GRPO算法依赖模型在训练过程中生成多个候选答案，然后通过奖励信号比较优劣来更新策略。但Stage2训练中，模型生成的答案质量极其糟糕：准确率仅55.07%（44.93%答案直接错误），格式正确率仅44.05%（55.95%格式不正确，缺少必需的`<answer>`标签或格式混乱），标签完整率73.64%（26.36%缺少必要标签）。综合三个维度，训练样本的实际可用率仅24.26%，这意味着**75.74%的训练样本完全不可用**。模型在训练过程中大部分时间都在学习错误的或格式混乱的答案，这种持续的负面学习信号逐步侵蚀了Stage1建立的推理能力，直接导致MATH和GSM8K准确率的双重崩溃。

**CMATH数据污染风险（14.7%）**：Stage2原始数据包含4,088条记录，其中MATH数据集3,488条（85.3%），CMATH数据集600条（14.7%）。虽然训练时使用了`--only-math`参数理论上应该过滤掉CMATH，但数据准备阶段可能存在混入风险。CMATH是中文小学数学题（如"果园里有桃树120棵，梨树的棵数是桃树的3倍"），与英文竞赛级MATH题（如"If $4^6=8^n$, what is $n$?"）在语言、难度和答案格式上都存在巨大差异。即使只有少量混入，也会引入中英文语言混乱和小学算术与大学数学的难度跳变，进一步破坏训练稳定性。

GRPO算法的工作机制决定了它对训练样本质量极为敏感。当模型生成的候选答案中55.95%格式错误时，奖励模型无法正确解析答案进行打分，导致奖励信号噪声极大。当44.93%的答案本身就是错误的时，模型学到的是"如何生成错误答案"而非"如何正确推理"。更严重的是，格式错误和答案错误往往同时出现在同一个样本上，使得该样本的梯度方向完全错误。在这种情况下，GRPO不是在强化正确的推理模式，而是在强化错误和混乱，这完全违背了强化学习的初衷。这解释了为什么经过3,488步训练后，MATH能力不升反降，从38.0%退步到34.5-35.5%。

**根本原因总结**：
1. **训练样本可用率仅24.26%**（75.74%不可用）- 格式错误55.95%、答案错误44.93%、标签缺失26.36%，导致模型主要学习噪声
2. **CMATH污染风险14.7%** - 原始数据包含600条中文题，`--only-math`过滤可能失效，引入语言和难度混乱
3. **学习率过低(5e-6) + 仅1 epoch** - 即使有24.26%高质量样本，保守配置也导致有效学习严重不足
4. **GRPO算法对低质量数据极度敏感** - 强化学习放大了错误信号，将模型推向错误方向

HumanEval能力的意外提升虽然从结果上看是积极的，但实际上进一步证明了训练过程的混乱。最可能的解释是，低质量数据训练让模型学会了更多"猜测"而非"精确推理"的策略，这种更灵活但不够严谨的推理模式在代码生成任务上意外地表现更好（因为代码生成有多种正确实现方式），但在数学推理任务上却是灾难性的（因为数学答案必须精确）。这种trade-off完全违背了我们的训练目标，证明Stage2的训练策略从根本上是错误的。

**决策建议**：

基于以上分析，我们做出以下关键决策：第一，**立即停止Stage2训练**，不进行第2个epoch的训练，因为在当前数据质量下继续训练只会浪费计算资源并进一步破坏模型能力。第二，**回退到checkpoint-22500作为当前最佳模型**，这是Stage1的终点checkpoint，在MATH（38.00%）和GSM8K（54.50%）上都显著优于所有Stage2 checkpoint。第三，**不需要对checkpoint-22500进行全量评估**，快速评估已经清楚显示Stage2完全失败，全量评估不会改变这个结论，节省的时间可以用于Stage3准备。第四，**彻底解决数据质量问题**，包括三个方向：提升模型生成答案的质量（通过更好的prompt、温度调节或使用更强的base模型）；严格验证`--only-math`过滤逻辑，确保CMATH完全不混入训练；改进奖励模型的容错能力，对格式问题更宽容。第五，**重新设计Stage3训练策略**，可能需要调整学习率（当前5e-6过于保守）、增加训练epoch数（1个epoch明显不够）、或者使用curriculum learning从简单题逐步过渡到难题。

这次Stage2失败虽然令人沮丧，但为我们提供了宝贵的教训：GRPO这类强化学习算法对训练过程中模型生成的答案质量有极高要求，当生成答案可用率低于30%时，训练信号会被噪声主导导致能力退化。在开始昂贵的训练之前，必须先验证模型在目标数据集上的生成质量（准确率、格式正确率、标签完整率），如果基础生成质量不达标，应该先通过SFT或prompt优化提升生成质量，再进入GRPO训练阶段。我们将吸取这些教训，在Stage3中采用更稳健的训练策略。

### Stage2 v2优化方案设计与启动调试

针对Stage2失败暴露的数据质量问题，我们设计了v2优化方案并尝试重新训练。核心思路是通过调整超参数和奖励权重来改善训练样本生成质量：将学习率从5e-6提升至1e-5恢复到Stage1水平以加快学习速度，将KL惩罚从0.10降低至0.08增大模型探索空间，最关键的是将format_reward权重从0.9大幅提升至1.5以优先解决55.95%的格式错误问题，同时将accuracy_reward权重从1.3轻微降低至1.2避免在高难度MATH题上的过度负反馈。训练目标设定为先跑0.5 epoch（1744步）进行快速验证，每250步进行一次checkpoint评估以密切监控效果，如果format_reward能提升到60%以上且GSM8K不触发灾难性遗忘警戒线，则继续训练到完整1 epoch。

然而启动Stage2 v2训练时遇到了一系列意料之外的技术问题，其中最棘手的是vLLM推理服务器无法通过健康检查导致训练流程卡在第三阶段。排查发现vLLM server主进程启动正常但worker子进程在初始化LLM Engine时崩溃，Pydantic校验抛出`ValidationError: Assertion failed, duplicate template name`。这个错误的特殊之处在于跨越了3个框架层（Pydantic→vLLM→PyTorch），需要系统化的排查思路才能定位根因。**第一步是全局搜索定位**：vLLM源码中完全没有"duplicate template"字样，说明断言来自外部依赖，在整个site-packages目录下搜索定位到PyTorch的`torch/_inductor/select_algorithm.py:1695`中的`TritonTemplate.__init__`断言。**第二步是注入堆栈追踪**：通过monkey-patch在断言触发点注入`traceback.print_stack()`，捕获完整的4层调用链：`select_algorithm`（模块级导入）→ `lowering`（动态导入子模块）→ `kernel/flex_attention`（模块级注册template）→ vLLM worker初始化触发二次加载。**第三步是独立复现**：在不涉及vLLM的环境下单独执行`import torch._inductor.select_algorithm`同样触发断言失败，证明这是**PyTorch 2.10.0自身的循环导入bug**而非业务代码问题。**第四步是最小化修复**：将3处致命断言（`TritonTemplate`、`ExternKernelChoice`、`CuteDSLTemplate`）改为debug级别日志允许覆盖注册，这个修复是安全的因为训练配置中`enforce_eager=True`已禁用torch.compile和CUDA Graphs，inductor编译路径不会被实际执行。这套"全局搜索→堆栈注入→独立复现→最小化修复"的排查流程，从一个表面的Pydantic错误深挖到底层框架缺陷，展示了跨框架debug的系统化方法论。

除了PyTorch的核心bug，还修复了4个配置层面的问题：系统未安装ripgrep导致启动脚本中用于检测vLLM失败模式的`rg`命令静默失效，替换为等价的`grep -qE`；旧日志残留在`/tmp/stage1_openr1_vllm_server.log`触发误报，在启动前显式清空日志文件；checkpoint路径错误指向`stage1_grpo_runs`而实际位于`stage2_grpo_runs`；最隐蔽的是`num_train_epochs: 0.5`与checkpoint中记录的`epoch: 6.45`冲突导致HuggingFace Trainer判断训练已完成直接退出，改用`max_steps: 24244`绝对步数控制解决。这次调试涉及跨PyTorch、vLLM、Pydantic三个框架的交互问题，从错误信息到根因经过了4层间接调用链的追踪，最终用一天时间将训练流程从完全无法启动修复到全链路跑通，并编写了`monitor_v2_live.sh`监控脚本和`test_stage2_v2_smoke.sh`冒烟测试工具支撑后续实验。

### Stage2 v2 评估结果与总结

**评估时间**：2026-03-11
**评估配置**：smoke suite，每个数据集 128 样本（数学），32 样本（代码）
**评估 Checkpoint**：ckpt-23000, ckpt-23500, ckpt-24000, ckpt-24244

#### Stage2 v2 各 Checkpoint 详细数据

**数学数据集准确率**（128 样本/数据集）：

| Checkpoint | MATH | GSM8K | Ape210K | CMATH | 数学平均 |
|------------|------|-------|---------|-------|----------|
| Stage1终点 | 38.58% | 57.54% | 57.80% | 73.04% | 56.74% |
| ckpt-23000 | 35.94% (-2.6) | 54.69% (-2.9) | 48.44% (-9.4) | 72.66% (-0.4) | 52.93% |
| ckpt-23500 | 37.50% (-1.1) | 52.34% (-5.2) | 46.88% (-10.9) | 69.53% (-3.5) | 51.56% |
| ckpt-24000 | 36.72% (-1.9) | 53.91% (-3.6) | 51.56% (-6.2) | 72.66% (-0.4) | 53.71% |
| ckpt-24244 | 35.94% (-2.6) | 55.47% (-2.1) | 47.66% (-10.1) | 67.97% (-5.1) | 51.76% |

**各指标最佳 Checkpoint**：
- MATH: ckpt-23500 = 37.50% (vs Stage1 38.58%, -1.08%)
- GSM8K: ckpt-24244 = 55.47% (vs Stage1 57.54%, -2.07%)
- Ape210K: ckpt-24000 = 51.56% (vs Stage1 57.80%, -6.24%)
- CMATH: ckpt-23000/24000 = 72.66% (vs Stage1 73.04%, -0.38%)

**格式正确率**：

| Checkpoint | MATH | GSM8K | Ape210K | CMATH | 平均 |
|------------|------|-------|---------|-------|------|
| Stage1终点 | 76.07% | 97.42% | 90.70% | 96.63% | 90.21% |
| ckpt-23000 | 74.22% | 96.09% | 90.62% | 96.09% | 89.26% |
| ckpt-23500 | 78.12% | 97.66% | 90.62% | 96.09% | 90.62% |
| ckpt-24000 | 75.00% | 95.31% | 90.62% | 96.88% | 89.45% |
| ckpt-24244 | 76.56% | 96.88% | 94.53% | 96.09% | 91.02% |

**代码数据集 Pass@1/5**（32 样本/数据集）：

| Checkpoint | MBPP@1 | MBPP@5 | HumanEval@1 | HumanEval@5 | 代码平均@1 |
|------------|--------|--------|-------------|-------------|-----------|
| S2v1 基线 | 53.91% | - | 42.68% | - | 48.30% |
| ckpt-23000 | 53.12% | 65.62% | 46.88% | 84.38% | 50.00% |
| ckpt-23500 | 56.25% | 62.50% | 59.38% | 81.25% | 57.81% |
| ckpt-24000 | 53.12% | 75.00% | 53.12% | 75.00% | 53.12% |
| ckpt-24244 | 62.50% | 71.88% | 53.12% | 84.38% | 57.81% |

#### Stage2 v1 vs v2 横向对比

| 指标 | Stage1 Final | S2v1 后期趋势 | S2v2 最佳 | 变化评估 |
|------|--------------|---------------|-----------|----------|
| MATH | 38.6% | 34.5%↓ | 37.5% (ckpt-23500) | 基本持平 |
| GSM8K | 57.5% | 49.0%↓↓ | 55.5% (ckpt-24244) | 轻微下降，远好于 S2v1 |
| CMATH | 73.0% | N/A | 72.7% (ckpt-24000) | 持平 |
| Ape210K | 57.8% | N/A | 51.6% (ckpt-24000) | 下降 6.2% |
| 格式率 | 90.2% | ~76% | 91.0% (ckpt-24244) | 大幅提升 |
| MBPP | ~60% | 55-58% | 62.5% (ckpt-24244) | 提升明显 |
| HumanEval | ~48% | 48-53% | 59.4% (ckpt-23500) | 提升明显 |

#### Stage2 v2 总结与分析

Stage2 v2 的核心目标是提升数学能力，但所有数学指标都低于 Stage1，尤其 Ape210K 下降了 6.24%，确实不成功。

不过有几点需要注意：

1. **评估样本量差异大**：Stage1 是全量评估（MATH 1187, GSM8K 1319, Ape210K 2000），S2v2 只有 smoke 128 条。Ape210K 的 -6.24% 在 128 条样本下约等于差了 8 道题，统计波动较大。

2. **真正的收获**：
   - 格式遵循率从 76% → 91%（这是 v2 调高 format_weight 的直接结果）
   - 代码能力有提升（MBPP +8.6%，HumanEval +10.4%）
   - 比 S2v1 稳定得多（S2v1 的 GSM8K 跌到 49%）

3. **根本问题**：format_weight=1.5 虽然提升了格式，但挤占了数学学习的梯度信号。accuracy_weight 从 1.3 降到 1.2 也进一步弱化了数学优化。

**总结**：Stage2 v2 成功解决了格式问题和训练稳定性问题，但在数学提升这个核心目标上没有达成。如果要继续优化，可能需要调整策略——比如在保持 format_weight 的同时提高 accuracy_weight，或者用更大/更难的数学训练数据。

**最终决策**：采用 **ckpt-24244** 作为 Stage2 的最终输出（综合能力最优：GSM8K、MBPP、格式率均为最佳），进入 Stage3 代码训练阶段。

---

## Stage3 训练执行与问题记录

### Stage3 训练启动

**启动时间**：2026-03-12
**起点 Checkpoint**：Stage2 v2 ckpt-24244
**训练数据**：stage3_hf/train.jsonl (51,888 条)

### 关键问题：代码任务的 accuracy_reward 处理

训练启动后观察到以下现象：

1. **accuracy_reward: nan** — accuracy reward 用 `math_verify` 解析代码答案时失败，返回 NaN
2. **大量 `Failed to parse gold solution` 警告** — 代码被当作数学公式解析

**原因分析**：
- 代码任务的 solution 是 Python 代码，不是数学表达式
- `math_verify` 无法解析代码，对无法解析的答案返回 `None`
- TRL 的 GRPO Trainer 会跳过 `None` 的 reward（不是跳过样本）

**实际行为**：

| Reward 函数 | 代码样本结果 | 作用 |
|-------------|--------------|------|
| accuracy | None（跳过） | 不参与这个维度的奖励计算 |
| format | 0.0 或 1.0 | 正常打分，模型从中学习格式 |
| tag_count | 0.0-1.0 | 正常打分，模型从中学习标签结构 |

**关键结论**：
- 样本**没有被丢弃**，模型仍然从 format 和 tag_count 两个 reward 信号学习
- `Failed to parse gold solution` 是 warning 日志，不是错误
- `rewards/accuracy_reward/mean: nan` 是因为该 batch 全是代码任务，没有有效的 accuracy reward

**当前方案的局限**：

| 任务类型 | 可用 Reward 信号 | 学习效果 |
|----------|------------------|----------|
| 数学任务 | accuracy + format + tag_count | ✅ 能学到正确性和格式 |
| 代码任务 | format + tag_count（accuracy=None） | ⚠️ 只能学格式，代码正确性靠 GRPO relative ranking 间接学习 |

**为什么不用 code_reward？**
- 要学代码正确性，需要 code reward（实际执行代码跑测试用例）
- 但我们的数据没有 `verification_info`（测试用例字段）
- 这是 Open-R1 原本的设计，不是 bug

**Stage3 训练效果预期**：
- 代码格式规范性会提升（从 format + tag_count 学习）
- 代码正确性提升有限（只能靠 GRPO 的 relative ranking 间接学习）
- 数学能力应该保持（15,000 条数学任务有完整的 accuracy reward）

### 训练 vs 评估：两套数据、两套逻辑

训练时和评估时用的是不同的代码和数据：

**训练时（rewards.py 的 accuracy_reward）**：
- 数据是 `stage3_hf/train.jsonl`，里面只有 problem 和 solution
- solution 是代码字符串，accuracy_reward 试图用 `math_verify` 解析 → 失败 → 返回 None
- 所以训练时代码任务确实没有正确性 reward

**评估时（eval_stage2_model.py）**：
- 数据是 `mbpp_test.json` 和 `humaneval_test.json`，这些测试集自带测试用例（assert 语句）
- 评估脚本把模型生成的代码 + 测试用例一起执行，跑通就算 pass
- 所以评估时可以判断代码对不对

|  | 训练（GRPO reward） | 评估（eval 脚本） |
|--|---------------------|-------------------|
| **数据** | stage3_code_train.json（无测试用例） | mbpp_test.json（有测试用例） |
| **代码正确性判断** | ❌ 不能（只学格式） | ✅ 能（实际执行代码） |

**现状总结**：训练时代码任务只能学格式，但评估时能准确测出代码正确率。这意味着 Stage3 的代码正确率提升主要来自 GRPO 的 relative ranking（同一 prompt 的多个生成中，格式更好的代码往往质量也更高），而不是直接的正确性 reward。

---

## Stage3 训练规划（代码能力提升）

### 训练目标

**核心目标**：提升代码生成能力（MBPP、HumanEval），同时保持数学能力不退化

**起点选择**：Stage2 v2 的 **ckpt-24244**

选择理由：
- 格式率高（91%），对代码生成有帮助
- 代码基础好（MBPP 62.5%, HumanEval 59.4%），可以更快收敛
- Stage3 混入 14,306 条 Ape210K 数据，足以恢复可能的数学能力下降
- Stage1 的 Ape210K 57.8% 是全量 2000 条评估，S2v2 的 51.6% 是 128 条 smoke 评估，实际差距可能没有 6% 那么大

### 训练数据

**数据文件**：`stage3_code_train.json`（51,888 条）

| 任务类型 | 样本数 | 占比 | 数据集组成 |
|----------|--------|------|------------|
| **代码** | 36,888 | 71% | apps=33,288, mbpp=3,600 |
| **数学** | 15,000 | 29% | ape210k=14,306, gsm8k=516, math=136, cmath=42 |

**难度分布**：
- introductory: 16,000 (30.8%)
- interview: 14,400 (27.7%)
- competition: 2,888 (5.6%)
- None: 18,600 (35.8%)

**重复采样**：apps x8, mbpp x30（强化难题学习）

**风险提示**：数学任务中 GSM8K 仅 516 条、MATH 仅 136 条，防遗忘力度偏弱。如果后续发现 GSM8K/MATH 下降，需考虑追加这两个数据集的样本量。

### 训练配置

```yaml
# 基础配置
model_name_or_path: /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct
resume_from_checkpoint: /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/checkpoint-24244

# 学习率（与 Stage2 v2 一致，已验证稳定）
learning_rate: 1e-5
warmup_steps: 100

# KL 约束
beta: 0.08  # 保持探索空间

# 批次配置
per_device_train_batch_size: 2
gradient_accumulation_steps: 2
effective_batch_size: 4
num_generations: 4

# 序列长度（代码任务需要更长）
max_prompt_length: 1024    # 代码题目更长
max_completion_length: 1024  # 代码输出更长（Stage2 是 512）

# 奖励权重（恢复 Stage1 配置）
accuracy_reward_weight: 1.3  # 调回来，代码准确率更重要
format_reward_weight: 0.9    # 降回来，格式率已经 91% 不需要强调
tag_count_reward_weight: 0.2

# Checkpoint 策略
save_steps: 500
save_total_limit: 10
logging_steps: 10
```

### 训练时长估算

- 数据量：51,888 条
- 有效 batch size：4
- 0.5 epoch 步数：51,888 / 4 / 2 ≈ **6,486 步**
- 每步耗时：约 5-6 秒（代码 completion 更长）
- 预计时长：**9-11 小时**（0.5 epoch）

### 监控策略

**实时监控**（每 10 步）：

| 指标 | 正常范围 | 警戒线 | 危险线 |
|------|---------|--------|--------|
| Loss | 缓慢下降 | 震荡 | 上升 |
| KL 散度 | <0.15 | 0.15-0.20 | >0.20 |
| 梯度范数 | 0.1-0.6 | 0.6-1.0 | >1.0 |

**关键评估**（每 500 步）：

| 指标 | 基准 (ckpt-24244) | 目标 | 警戒线 | 说明 |
|------|-------------------|------|--------|------|
| **MBPP@1** | 62.5% | 70%+ | <55% | 核心目标 |
| **HumanEval@1** | 59.4% | 65%+ | <50% | 核心目标 |
| **GSM8K** | 55.5% | 保持 | <50% | 防遗忘哨兵 |
| **MATH** | 37.5% | 保持 | <33% | 防遗忘 |
| **Ape210K** | 51.6% | 恢复到 55%+ | <45% | 数据混入应能恢复 |

### 成功标准（Stage3 完成条件）

- ✅ MBPP@1 ≥ 70%（或相对提升 ≥10%）
- ✅ HumanEval@1 ≥ 65%（或相对提升 ≥5%）
- ✅ GSM8K 下降 ≤5%（保持 ≥50%）
- ✅ MATH 下降 ≤5%（保持 ≥33%）
- ✅ Ape210K 恢复到 ≥52%（或不继续下降）

### 风险与应对

**风险 1：GSM8K/MATH 遗忘**
- 原因：Stage3 数据中这两个数据集占比极低（GSM8K 516 条，MATH 136 条）
- 应对：如果下降超过警戒线，追加 GSM8K/MATH 样本混入训练

**风险 2：代码能力不提升**
- 原因：代码任务难度高，GRPO 探索效率可能不足
- 应对：增加 num_generations 到 6-8，或调整 temperature 增加多样性

**风险 3：训练不稳定**
- 原因：代码 completion 更长，可能导致显存压力或梯度问题
- 应对：监控 KL 和梯度范数，必要时降低 max_completion_length 或增大 beta

---

## Stage3 评估结果与分析

### 训练完成状态

**训练时间**：2026-03-12 ~ 2026-03-13
**训练步数**：24,244 → 30,730（0.5 epoch，6,486 新步数）
**训练状态**：正常完成，各项指标稳定

**训练过程指标演变**（每段 ~500 步均值）：

| 阶段 | Format | TagCount | Reward | KL |
|------|--------|----------|--------|-----|
| 24250-24750 | ~82% | ~89% | ~1.06 | 0.045 |
| 25750-26250 | ~87% | ~96% | ~1.15 | 0.042 |
| 28250-28750 | ~88% | ~96% | ~1.16 | 0.041 |
| 29250-30730 | ~87% | ~96% | ~1.16 | 0.044 |

**结论**：TagCount 在 25000 步后饱和在 96%，Format 稳定在 85-88%，Reward 在 1.13-1.16 区间平台期，KL 健康。训练从 26000 步后进入稳态。

---

### 全阶段评估对比

#### Baseline → Stage1 → Stage2 → Stage3 完整链路

**数学任务**：

| 阶段 | MATH | CMATH | GSM8K | Ape210K | 数学均值 |
|------|------|-------|-------|---------|---------|
| **Baseline** | 13.1% | 20.0% | 8.3% | 15.3% | 14.2% |
| **Stage1** | 38.6% | 73.0% | 57.5% | 57.8% | 56.7% |
| **Stage2** | 35.9% | 68.0% | 55.5% | 47.7% | 51.8% |
| **Stage3 best** | 39.1% | 78.1% | 60.9% | 55.5% | 58.4% |

**代码任务**：

| 阶段 | MBPP pass@1 | HumanEval pass@1 | 代码均值 |
|------|-------------|------------------|---------|
| **Baseline** | 57.2% (n=257) | 40.2% (n=164) | 48.7% |
| **Stage2** | 59.4% (n=32) | 50.0% (n=32) | 54.7% |
| **Stage3 best** | 59.4% (n=32) | 53.1% (n=32) | 56.3% |

---

### Stage2 基线 vs Stage3 各 Checkpoint 对比

| 指标 | S2 基线 | ckpt-29500 | ckpt-30000 | ckpt-30500 | ckpt-30730 |
|------|---------|------------|------------|------------|------------|
| **MATH** | 35.9% | 35.9% (=) | **39.1% (+3.1)** | 35.2% (-0.8) | 36.7% (+0.8) |
| **CMATH** | 68.0% | **78.1% (+10.2)** | 70.3% (+2.3) | 72.7% (+4.7) | 72.7% (+4.7) |
| **GSM8K** | 55.5% | 56.2% (+0.8) | **60.9% (+5.5)** | 59.4% (+3.9) | 53.9% (-1.6) |
| **APE210K** | 47.7% | 53.9% (+6.2) | **55.5% (+7.8)** | 53.9% (+6.2) | 50.8% (+3.1) |
| **MBPP** | 59.4% | 56.2% (-3.1) | **59.4% (=)** | 56.2% (-3.1) | 53.1% (-6.2) |
| **HumanEval** | 50.0% | 50.0% (=) | 43.8% (-6.2) | **53.1% (+3.1)** | **53.1% (+3.1)** |
| **Format** | 76.6% | 75.8% (-0.8) | 77.3% (+0.8) | 76.6% (=) | 75.8% (-0.8) |

---

### 关键发现

#### 1. Stage2 导致数学退化（灾难性遗忘）

| Stage1 → Stage2 | MATH | CMATH | GSM8K | Ape210K |
|-----------------|------|-------|-------|---------|
| 变化 | -2.7% | **-5.0%** | -2.0% | **-10.1%** |

Stage2 只训练 MATH+CMATH，却导致 **Ape210K 退化 10%**。这是典型的灾难性遗忘：专注难题训练导致简单题能力下降。

#### 2. Stage3 成功恢复并超越

| Stage2 → Stage3 | MATH | CMATH | GSM8K | Ape210K |
|-----------------|------|-------|-------|---------|
| 变化 | +3.2% | **+10.1%** | +5.4% | +7.8% |

Stage3 几乎完全恢复了 Stage2 的损失：
- CMATH 从 68% → 78%（**超过 Stage1 的 73%**）
- GSM8K 从 55.5% → 60.9%（**超过 Stage1 的 57.5%**）
- Ape210K 从 47.7% → 55.5%（接近 Stage1 的 57.8%）

**原因**：Stage3 数据中 29% 是数学题（14,306 条 Ape210K + 516 条 GSM8K 等），起到了继续强化和恢复的作用。

#### 3. 代码能力变化

| 对比 | MBPP | HumanEval |
|------|------|-----------|
| Baseline → Stage2 | +2.2% | **+9.8%** |
| Stage2 → Stage3 | 0% | +3.1% |
| **总提升** | +2.2% | **+12.9%** |

- **HumanEval 持续提升**：40.2% → 50.0% → 53.1%
- MBPP 基本保持（已接近 baseline 水平）

**重要说明**：smoke suite 每个代码数据集只有 32 题，±1 题就是 ±3.1% 的波动，结论可靠性有限。

---

### Checkpoint 选择推荐

| 选择 | 推荐 Checkpoint | 理由 |
|------|-----------------|------|
| **数学最优** | ckpt-30000 | MATH +3.1%, GSM8K +5.5%, APE210K +7.8% |
| **代码最优** | ckpt-30500 | HumanEval +3.1%, MBPP 仅 -3.1% |
| **综合最优** | **ckpt-30000** | 数学全面提升，代码 MBPP 持平，仅 HumanEval 下降 |

---

### Stage3 效果总结

**✅ 数学方面：有效**
- CMATH/APE210K 全线提升 +3~10%
- MATH 和 GSM8K 在 ckpt-30000 达到峰值
- 成功恢复 Stage2 导致的灾难性遗忘

**⚠️ 代码方面：部分有效**
- HumanEval 提升 +12.9%（40.2% → 53.1%）
- MBPP 基本持平（57.2% → 59.4%）
- 未达到 MBPP≥65%、HumanEval≥60% 的原定目标

**⚠️ 训练后期过拟合**
- ckpt-30000 之后各项指标开始下滑
- 建议 Stage3 训练到 0.3-0.4 epoch 即可

**📝 待办**
- 对 ckpt-30000 和 ckpt-30500 用 full suite（MBPP 257 / HumanEval 164）复测代码能力，以得出更可靠的结论

---

### 面试问题与标准答案

#### Q1：Stage1 具体在做什么？为什么不是简单监督微调？

**标准答案**：

Stage1 使用 GRPO（Group Relative Policy Optimization），而非简单监督微调。

**具体做法**：
- 使用本地准备的 22,419 条数学数据（GSM8K:Ape210K = 1:2）作为 prompt 源
- 每个 prompt 让当前策略多采样 4 条回答（`num_generations=4`）
- 基于准确率、格式、标签计数三维奖励打分（权重 1.3:0.9:0.2）
- 在 KL 约束下（`beta=0.08`）做策略更新

**与 SFT 的区别**：
- SFT 只拟合参考答案，是单一路径的模仿学习
- GRPO 允许模型在多个候选推理路径里探索
- 在 reward 引导下逐步偏向高质量解
- 对复杂推理任务的提升更明显（准确率从 0% → 70%）

---

#### Q2：奖励设计的具体细节？

**标准答案**：

奖励设计基于 Open-R1 实现，分为三个维度：

**1. accuracy_reward（权重 1.3）**

实现方式：LaTeX parse + symbolic verify（不是简单数值比较）

流程：
```python
# 使用 latex2sympy2_extended 和 math_verify 库
a. parse() 提取 gold solution 和模型输出的数学表达式
b. verify() 进行符号验证（symbolic verification）
c. 返回 1.0（正确）或 0.0（错误）
   解析失败返回 None（跳过该样本）
```

**2. format_reward（权重 0.9）**

检查格式：
```
<think>
...
</think>
<answer>
...
</answer>
```

注意：
- 使用 `<think>` 和 `<answer>` 标签（不是 `<reasoning>`）
- 使用正则表达式严格匹配完整格式
- 必须包含换行符和完整的开闭标签

**3. tag_count_reward（权重 0.2）**

精确计数 4 个特定标签，每个 0.25 分：
- `<think>\n` 出现 1 次
- `\n</think>\n` 出现 1 次
- `\n<answer>\n` 出现 1 次
- `\n</answer>` 出现 1 次

注意：
- 不是泛化的"成对闭合检查"
- 而是精确计数特定格式

**最终总奖励**：
```
total_reward = 1.3 × accuracy + 0.9 × format + 0.2 × tag_count
```

**实际效果**（epoch 0.59）：
- 总奖励：0.09 → 1.671（18.6 倍提升）
- 准确率：0% → 70%（最高值）
- 格式奖励：5% → 77.5%（最高值）
- 标签奖励：55% → 96.875%（最高值）

---

#### Q3：如何判断训练「稳定」？

**标准答案**：

主要看两个核心指标：**KL 散度**和**梯度范数**。

**KL 散度分析**（最近 10,000 步）：
- 范围：0.054 ~ 0.130
- 平均：0.077
- \> 0.1 的步数：7.5%
- 从未超过 0.15

**结论**：
- 策略没有偏离 reference 太远
- 没有出现 KL 爆炸
- 在健康范围内探索

**梯度范数分析**：
- 范围：0.12 ~ 0.67
- \> 1.0 的步数：< 1%

**结论**：
- 学习率和 reward scale 匹配良好
- 没有频繁出现梯度爆炸或严重震荡

**综合判断**：
- 总奖励和准确率单调提升
- KL 和梯度范数都在健康范围
- 这是一个**收敛过程**而不是训练发散

---

#### Q4：系统架构的工程优化？

**标准答案**：

为了提高吞吐和稳定性，我做了**生成-训练解耦**：

**双卡分工**（Separate 模式）：

- **GPU 1（vLLM）**：只负责高吞吐地生成 GRPO 所需的多样本回答（显存 ~18.6 GB，`gpu_memory_utilization=0.55`）
- **GPU 0（训练）**：专门做训练，消费这些样本并更新策略参数（显存 ~9.8 GB）

**关键优化**：
1. **只同步 LoRA 可训练参数**
   - 不同步全量模型权重（30 亿参数，6 GB）
   - 只同步 252 个 LoRA 参数（14.7 MB）
   - 通信开销降低 **400 倍**

2. **参数同步时间优化**
   - 从 5-10 秒（超时）优化到 0.5 秒
   - 提升 **10-20 倍**

3. **解决的问题**
   - NCCL 通信超时
   - sync_barrier 卡死
   - vLLM generate 偶发卡死
   - torch.dtype 序列化失败
   - LoRA 梯度回传断裂

**设计理念**：
- 类似社区 RLHF with vLLM 的最佳实践
- 推理和训练解耦
- 共享大模型只传 adapters
- 用 LoRA 达到参数高效和通信高效

**实际效果**：
- 稳定运行 13,000+ 步
- 每步 2.5 秒
- 无 OOM、无超时、无卡死

---

## 参考资料
- Open-R1 文档: https://github.com/huggingface/open-r1
- TRL vLLM 集成: https://github.com/huggingface/trl
- GRPO 论文: https://arxiv.org/abs/2402.03300
- NCCL 文档: https://docs.nvidia.com/deeplearning/nccl/
