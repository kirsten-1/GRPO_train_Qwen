# 开发日记：Stage2 v2 训练启动失败排查与修复

## 项目背景

在基于 Qwen2.5-3B-Instruct 模型的 GRPO（Group Relative Policy Optimization）强化学习训练项目中，Stage1 训练已顺利完成（MATH 38.0%、GSM8K 54.5%），需要启动 Stage2 v2 继续优化模型的格式遵循能力（format reward 从 44% 提升至 60%+）。

训练环境为双卡 RTX 5090（32GB×2），采用 GPU 0 训练 + GPU 1 运行 vLLM 推理服务的架构。技术栈：TRL 0.18.0 + vLLM 0.17.0 + PyTorch 2.10.0 + Pydantic 2.12。

启动 Stage2 v2 训练脚本时，vLLM 推理服务器始终无法通过健康检查，报 "Timed out waiting for vLLM server"，导致训练流程无法进入第三阶段。

## 主要职责

负责从零定位并修复 Stage2 v2 训练启动失败的全部问题，涉及跨框架版本兼容性分析、进程级调试、训练恢复逻辑修复等。具体包括：

### Bug 1：PyTorch 2.10.0 Inductor 模块重复注册断言失败（核心难点）

**现象**：vLLM server 主进程启动成功，但 worker 子进程（EngineCore Process-1）在初始化 LLM Engine 时崩溃，Pydantic 校验抛出 `ValidationError: Assertion failed, duplicate template name`。

**排查过程**：

1. **初步分析**：错误信息指向 `VllmConfig` 的 Pydantic 校验失败，但 vLLM 源码中搜索不到 "duplicate template" 字样，说明断言来自外部依赖。

2. **全局搜索定位**：在整个 site-packages 目录下搜索 "duplicate template"，定位到两处：
   - `torch/_inductor/select_algorithm.py:1695` — `TritonTemplate.__init__` 中的 `assert name not in self.all_templates`
   - `torch/_inductor/codegen/cutedsl/cutedsl_template.py:41` — `CuteDSLTemplate.__init__` 中的同类断言

3. **复现与根因分析**：通过 monkey-patch `TritonTemplate.__init__` 注入 `traceback.print_stack()`，捕获到完整调用链：
   ```
   torch._inductor.select_algorithm（模块级代码）
   → from . import lowering
   → lowering.py 的 import_submodule(kernel)
   → torch._inductor.kernel.flex_attention（模块级代码，第786行）
   → flex_attention_template = TritonTemplate(...)  ← 第二次注册，触发断言
   ```

   **根因**：PyTorch 2.10.0 的 `torch._inductor` 模块存在循环导入问题。`select_algorithm.py` 在模块加载时导入 `lowering`，`lowering` 又动态导入 `kernel/flex_attention.py`，后者在模块级代码中注册 `TritonTemplate`。当 vLLM 的 worker 子进程初始化引擎配置时，导入链触发了该模块的二次加载，导致同名 template 重复注册。

4. **验证关键发现**：即使不涉及 vLLM，单独执行 `import torch._inductor.select_algorithm` 就会触发断言失败——这是 **PyTorch 2.10.0 自身的 bug**。

5. **修复第一个断言后暴露第二个**：`duplicate extern kernel: _grouped_mm`，位于 `select_algorithm.py:2166`，同样是模块重复导入导致的 `ExternKernelChoice` 重复注册。

**修复方案**：将 3 处致命断言改为 debug 级别日志，允许覆盖已注册的模板/kernel：

```python
# 修复前（torch/_inductor/select_algorithm.py:1695）
assert name not in self.all_templates, "duplicate template name"

# 修复后
if name in self.all_templates:
    import logging as _logging
    _logging.getLogger(__name__).debug("duplicate template name: %s (overwriting)", name)
```

同样的修复应用于 `select_algorithm.py:2166`（ExternKernelChoice）和 `cutedsl_template.py:41`（CuteDSLTemplate）。

**为什么这个修复是安全的**：`enforce_eager=True` 已禁用 torch.compile 和 CUDA Graphs，inductor 编译路径不会被实际执行，template 注册仅是模块导入的副作用，覆盖写入不影响任何运行时行为。

### Bug 2：`rg`（ripgrep）未安装导致错误检测失效

**现象**：启动脚本第 83/89 行使用 `rg` 命令检测 vLLM 启动失败模式，但系统未安装 ripgrep，导致：
- Stage1 脚本中的失败检测（检查日志中的错误模式）静默失败，无法快速报错
- Stage2 脚本中的参数检测（`printf '%s\n' "$@" | rg -q`）直接报错

**修复**：将所有 `rg` 调用替换为等价的 `grep`：
- `rg -q "pattern"` → `grep -qE "pattern"`
- `rg -q -- '--flag'` → `grep -qF -- '--flag'`

### Bug 3：旧日志残留触发误报

**现象**：修复 Bug 1 后，脚本在第一次健康检查循环中就检测到 "vLLM startup failure"，实际是读取了上次失败遗留的旧日志 `/tmp/stage1_openr1_vllm_server.log`。

**修复**：在启动 vLLM server 前显式清空日志文件：
```bash
: > /tmp/stage1_openr1_vllm_server.log
```

### Bug 4：Checkpoint 路径错误

**现象**：`resume_from_checkpoint` 指向 `stage1_grpo_runs/.../checkpoint-22500`，但该 checkpoint 实际位于 `stage2_grpo_runs/` 目录下。

**修复**：通过 `find` 定位正确路径并更新配置。

### Bug 5：`num_train_epochs` 与 checkpoint epoch 冲突导致 0 步训练

**现象**：训练启动后立即完成，`train_runtime: 0.02s`，实际训练 0 步。

**根因**：checkpoint-22500 中记录的 `epoch: 6.45`，而 yaml 配置 `num_train_epochs: 0.5`。HuggingFace Trainer 判断 `6.45 > 0.5`，认为训练已完成。

**修复**：将 `num_train_epochs: 0.5` 改为 `max_steps: 24244`（22500 + 1744），用绝对步数控制训练长度，不受 epoch 计数影响。

## 项目成果

1. **定位并修复了 PyTorch 2.10.0 的 inductor 模块循环导入 bug**，涉及跨 3 个框架（PyTorch / vLLM / Pydantic）的交互问题，从错误信息到根因经过了 4 层间接调用链的追踪，最终确认是 PyTorch 自身的缺陷而非 vLLM 或业务代码问题。

2. **共修复 5 个独立 bug**，涵盖框架兼容性、系统依赖、文件状态管理、路径配置、训练恢复逻辑等不同层面，从完全无法启动到训练流程全链路跑通。

3. **Stage2 v2 训练成功启动并持续运行**，在 checkpoint-22500 基础上继续训练 1744 步，训练过程中 format reward 从基线 44% 波动上升，accuracy reward 维持在 45%-75% 区间，KL 散度稳定在 0.03-0.07，训练状态健康。

4. **编写了配套的监控工具** `monitor_v2_live.sh`，实时展示训练进度和关键指标（accuracy/format/tag_count reward、loss、KL），支持 `watch` 持续刷新。

5. **编写了 smoke test 脚本** `test_stage2_v2_smoke.sh`，支持在正式训练前用 3 步快速验证全链路（vLLM 启动 → 模型加载 → checkpoint 恢复 → GRPO 训练），避免长时间训练后才发现配置错误。
