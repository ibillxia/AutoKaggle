# surrogate 训练：本地调通配方，云端只换配置

比赛锁定的基模是 `gemma-4-31b-it-qat-w4a16-ct`。本地 RTX 4060 Ti 只有 **8GB 显存**
（桌面常驻占掉约 2.4GB，实际可用 ~5.7GB），而这份权重光加载就要约 16GB，
QLoRA 训练要 ≥24GB —— **本机既跑不了推理也训不了 31B**。

所以这里的定位不是"在本地训一个能提交的模型"，而是：

> **在本地用替身模型把整条训练流水线跑通、把坑踩完，再把配方原样搬到 Kaggle L4x4 上训真 31B。**
> 省下来的是宝贵的 L4 配额（免费档约 30 GPU 小时/周，L4x4 按 2 倍速率扣，**实际约 15 小时/周**）。

## 文件

| 文件 | 作用 |
|---|---|
| `surrogate_train.py` | 主入口。`smoke` / `features` / `train` / `export` 四个子命令 |
| `sft_data.py` | 轨迹 → 训练样本：chat 模板渲染、assistant 段定位、loss mask、collator |
| `adapter_export.py` | 把 adapter 打进 `adapters/<name>/`，并做提交前的扩展名/格式硬校验 |
| `configs/local_surrogate.yaml` | 本机 8GB 用的替身配置（Qwen2.5-Coder-1.5B + bnb 4bit） |
| `configs/l4x4_gemma31b.yaml` | Kaggle L4x4 的正式配置（31B + checkpoint_native） |
| `data/seed_trajectories.jsonl` | 轨迹格式示例（**人工构造的格式样例，不是真实训练数据**） |
| `tests/test_train_pipeline.py` | 10 个单元测试，离线可跑 |

## 快速开始

```bash
cd solutions/gemma_4_developer_agent/train
PY="C:/Users/deepg/.workbuddy/binaries/python/envs/default/Scripts/python.exe"

"$PY" surrogate_train.py smoke                                   # 零下载，验证整条流水线
"$PY" -m pytest tests -q                                        # 单元测试
"$PY" surrogate_train.py features --config configs/local_surrogate.yaml
"$PY" surrogate_train.py train --config configs/local_surrogate.yaml --max-steps 20
"$PY" surrogate_train.py train --config configs/local_surrogate.yaml
"$PY" surrogate_train.py export --adapter-dir runs/local_surrogate \
        --submission-root ../artifacts/best_submission --name main_lora
```

国内网络下 huggingface.co 直连不通，下载替身模型前先设镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 实测结果（RTX 4060 Ti 8GB）

环境：torch 2.14.1+cu126 / transformers 5.19.0 / peft 0.21.2 / bitsandbytes 0.50.2 / Python 3.13

| 检查 | 结果 |
|---|---|
| `smoke`（零下载 + 微型随机模型） | 4 步通过，loss 4.09 → 3.94，adapter 0.13MB，`.sh` 负例被拦 |
| 单元测试 `pytest tests -q` | **12 / 12 通过** |
| `features`（真实 Qwen2.5 分词器） | 3 条样本 / 1450 token，其中 583 被监督（40.2%），无 warning |
| `train`（1.5B + bnb 4bit，`--max-steps 5`） | **峰值显存 3.24 GB**，loss 1.91 → 1.05，约 1.2 s/step |
| `train`（完整配置 2 epochs） | adapter 70.49 MB，`adapter_check` 全绿 |
| `export` | 生成 `adapters/main_lora/`，扩展名扫描通过 |

**3.24GB / 8GB** —— 替身这一档余量充足，Qwen 1.5B 甚至还能再往上顶到 3B 左右。
对照 31B 需要的 ~16GB 权重，本地这一档与真机之间差的不只是参数，是数量级。

## 踩到的真实版本坑（已修，并有回归测试钉住）

这两个坑只有在真机上跑才会暴露，4.x 时代写的代码在 5.x 会直接崩：

1. **`TrainingArguments` 移除了 `warmup_ratio`**（只留 `warmup_steps`）。
   → `make_trainer` 现在按运行时签名探测，并自动把 ratio 换算成步数；
   所有不被接受的参数都会**打印出来**再丢弃，绝不静默忽略。
2. **`BatchEncoding` 不再是 `dict` 子类**。4.x 里 `isinstance(out, dict)` 能过，
   5.x 过不了，于是 `[int(x) for x in out]` 会遍历到键名并抛
   `invalid literal for int() with base 10: 'input_ids'`。
   → `_as_ids()` 改为优先取 `.input_ids` 属性，`tests` 里有专门的 `_FakeBatchEncoding` 回归用例。

## 本地 → Kaggle 的差异只有三行配置

| 项 | 本地 `local_surrogate.yaml` | Kaggle `l4x4_gemma31b.yaml` |
|---|---|---|
| `model.id` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `google/gemma-4-31b-it-qat-w4a16-ct` |
| `model.quantization` | `bnb_4bit` | `checkpoint_native` |
| batch | `1 × grad_accum 8` | `1 × grad_accum 16`（4 卡 DDP） |

`max_seq_len`、`lr`、`lora.r` 这类需要按显存和效果调的旋钮，也都写在 YAML 里。
**脚本本身一行都不用改。**

## 两个设计决策，以及为什么

**1. 只对 assistant 段算 loss（`train_on: assistant_only`）**
系统提示词是固定的，工具输出是环境给的。把它们的 token 也算进 loss，等于让模型花容量去
复述文件内容和日志 —— 那是纯粹浪费。`sft_data.py` 用「前缀一致」的方式定位每一轮 assistant 段：
对第 i 轮，先用 `apply_chat_template(msgs[:i], add_generation_prompt=True)` 拿到长度作为起点，
再用 `msgs[:i+1]` 的长度作为终点。标准 ChatML / Llama / Gemma 模板都满足这个前提，
代码里有断言保护；一旦某个模板不满足，会**显式报错**而不是静默算错。

**2. `checkpoint_native` 而不是二次量化**
31B 的官方权重**本身就是** w4a16 量化产物。再用 bitsandbytes 二次量化既不省显存
（约 16GB 已经是 4bit），又白白引入一层额外误差。所以真机配置直接按原样加载，
LoRA 适配器以 bf16 叠上去。

## 诚实边界

- **`smoke` 全部验证通过**，它证明的是「管道能通」，不是「模型会变好」。
  微型随机模型的 loss 下降说明梯度确实回流到了 LoRA 上。
- **本地跑出来的 adapter 不能提交。** `runs/local_surrogate/` 里那份是 Qwen 1.5B 的，
  `adapter_config.json` 的 `base_model_name_or_path` 写着 Qwen —— 比赛只认 gemma-4-31b 的适配器。
  这就是导出演示写在 `runs/_export_demo/` 而不是 `artifacts/best_submission.zip` 的原因
  （后者的时间戳自始未变，可自行核对）。
- **`checkpoint_native` 路径未经实机验证** —— 本地没有 31B 权重。首次上 Kaggle 时请先跑一次
  `--max-steps 2` 试加载，确认 (a) 权重自带的 `quantization_config` 被 transformers 正确识别，
  (b) 单卡 24GB 放得下。代码里对这种情况会打 warning，不会假装成功。
- **`data/seed_trajectories.jsonl` 是我手写的格式样例**，只有 3 条，用来固化 schema 和跑冒烟。
  真正训练需要真实轨迹数据（SWE-bench 风格 issue→patch 对，再改写成工具调用序列），
  这部分数据本仓库没有。**3 条样本训出来的 loss 曲线没有统计意义**，只说明流水线在动。
- **不做 RL**。YAML 里没有 RL/DPO 相关字段，因为本包只覆盖 SFT + LoRA 这一段确定性工作流。
