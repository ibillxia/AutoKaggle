# Jev 与 Qwen 系开源决策模型：调研、选型与 Kaggle 分类任务实测

> 环境：Windows / RTX 4060 Ti 8GB / i7-14700F (20C28T) / 32GB RAM / E: 剩余 ~175GB
> 全部推理在本地完成，未调用任何 API。日期：2026-10-13

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| Jev 能下载吗？ | **不能。** Jev 是 TypeSafe AI 的闭源商用模型，只有托管 API，权重从未发布，也没有自托管许可 |
| 有没有对应 Jev 的开源模型？ | **有，且是 Qwen 基座：`kev`**（Qwen3.5-0.8B/4B/9B + Qwen3.8-27B，LoRA + 指针头，Apache-2.0，兼容 Jev 的 `/v1/systemone` 接口） |
| 适合做 Kaggle 分类吗？ | **适合做"文本分类/打标/路由"这一类**，尤其是类别数固定、需要概率和置信度的场景。不适合需要生成、长链推理的任务 |
| 本机该选哪个？ | **kev-0.8B 作为主力**（1.6GB 显存，0.06s/条）；**kev-4B 作为质量上限**（8.2GB，贴死 8GB 显存，需共享内存回退） |
| 实测表现 | 域内：IMDB **96.0%**(4B) / 93.5%(0.8B)，AG News 两者均 90.75%；**零样本** SMS Spam：**4B 88.0%** vs 0.8B 56.5%。校准极好（ECE 0.017–0.041） |

一句话：**Jev 本体买不到，但它的开源等价物能下载、能在 8GB 卡上跑，分类任务上确实可用——0.8B 只适合域内/当冷启动工具，要泛化必须上 4B；但只要有标注数据，域内监督基线往往更快更准。**

---

## 1. Jev 到底是什么

### 1.1 定位

Jev 由 TypeSafe AI 于 2026-09-15 发布，创始人是前 OpenAI 研究员 Diogo Almeida（GPT-4 论文共同作者、RLHF 方向）。官方原话是 **"Jev is neither small nor an LLM"** —— 它不是一个"更小的聊天模型"，而是把大模型输出端整体替换掉的 **System One Model（系统 1 模型）**：

```
传统 LLM ：state → 自回归逐 token 生成文本 → 你的代码解析 → 校验 → 重试
Jev      ：state + 一组类型化问题 → 单次前向并行打分 → 类型化答案 + 校准概率
```

名字取自经济学家 William Stanley Jevons（杰文斯悖论）：单次决策变得又快又便宜，总决策量会暴涨。

### 1.2 三个决策原语

| 类型 | 输入 | 输出 |
|---|---|---|
| `choice` | 1–255 个具名选项（可带描述） | 每个选项的概率 + 最可能选项 + 置信度 |
| `score` | 1–255 个有序等级 | 每个等级的概率 + 期望等级 |
| `noul` | 是 / 否（可带描述） | "是"的概率 |

这正好覆盖了分类问题的全部形态：二分类 → `noul`，多分类 → `choice`，有序/评分 → `score`。

### 1.3 关键技术点

- **架构**：预训练骨干做 prefill（放弃自回归解码），输出头换成决策读出头。社区反向分析认为激活参数约 **4–9B**（MoE 总参数可能数倍于此），"a fine-tuned encoder with decision heads / frontier 级的零样本分类器"。
- **训练方法**：RLCD（Reinforcement Learning for Calibrated Decisions），优化目标是让"声明的置信度"与"真实正确率"对齐。
- **类型安全**：输出空间在请求发出时就被枚举确定，因此结构上不可能出现格式错误（0% type errors）。
- **定价**：**$0.042 / 百万输入 token，输出免费**。实测口径约 $0.064 / 600 次判断。

### 1.4 官方宣称 vs 第三方实测

| 指标 | 官方宣称 | 第三方独立复测（Jev Research，2026-10-04，16,379 次请求） |
|---|---|---|
| 速度 | 比 LLM 快 40–200×（自建 workflow 评测，193.6×） | 单次约 320–340ms，与最便宜的同等准确率 LLM 比快 4–8× |
| 成本 | 便宜 444.6× | 比 Sonnet 便宜约 90×（同一批 600 次判断：Jev $0.064 vs $5） |
| 能力 | "frontier intelligence" | MMLU-Pro 82.7%、GPQA Diamond 76.5%、**ARC-AGI-2 0.0%** —— 属于"优秀中小模型"而非前沿 |
| 幻觉 | "不能幻觉" | 只保证 **格式** 合规，不保证 **判断** 正确 |
| 校准 | 每个答案自带校准概率 | 置信度 >0.95 时准确率 96.5%；0.85–0.95 区间标称 0.90 实际 77.6% |

**社区共识（也是本调研的出发点）：Jev 是一个"小模型 + 概率读出头替换生成头"，这正是开源社区能在几天内复现它的原因。**

---

## 2. 开源世界里谁对应 Jev

Jev 发布的 72 小时内，社区出现了 50+ 个复现/替代项目。按基座分类：

### 2.1 Qwen 基座（重点）

| 项目 | 基座 | 规模 | 许可 | 备注 |
|---|---|---|---|---|
| **kev**（jaredpalmer） | **Qwen3.5-0.8B/4B/9B-Base、Qwen3.8-27B** | 0.8B–27B | Apache-2.0 | **最接近 Jev 的开源实现**：兼容 `/v1/systemone`，TypeSafe 官方 SDK 可不改代码指过来；带训练/基准/服务全套工具链 |
| Jeff（firelex / mstrasser） | Qwen3.5-0.8B/2B、Gemma-4-E2B | 0.8B–2B | 权重 Apache-2.0 / 代码 MIT | 家用单卡训出来的，~22–28ms/决策 |
| Strands Decider | Qwen3.5-2B-Base + 指针头 | ~1.9B | Apache-2.0 | 面向 agent 内部决策 |
| NanoJev | Qwen3-0.6B | 0.6B | MIT | 教学向的最小复刻 |
| SemIf | Qwen3-0.6B / Qwen3.5-4B | 0.6–4B | MIT | logit 读出路线 |
| QwenJev / openjev-sglang | Qwen3.5-4B VLM / Qwen3.6-35B-A3B | 4B–35B | 各异 | 能读图；后者靠 SGLang 前缀缓存提速 |

### 2.2 非 Qwen 基座（供对照）

| 项目 | 基座 | 规模 | 许可 |
|---|---|---|---|
| Laya | ModernBERT-large / mmBERT-base | 421M | Apache-2.0 |
| Julia 1 | mmBERT-small | 144.3M | Apache-2.0 |
| Von | ModernBERT | ~395M | Apache-2.0 |
| SelfJev-4B | 4B 多模态 | 4B | Apache-2.0 |
| Plumb-4B / imajev-4b / decider-4b / Mica-v0.1-4B | 4.2B 上下 | 4B | Apache-2.0 |

### 2.3 纯通用 Qwen 模型也能做分类

`Qwen3.5-4B`、`Qwen3.8-27B`（Apache-2.0，稠密 27.8B，262K 上下文，bf16 56GB / FP8 31GB）等通用指令模型，用「枚举选项 + 约束输出」的提示词同样能做分类，但：

- 要自回归生成，慢 1–2 个数量级；
- 需要解析 + 重试兜底；
- **没有校准概率**（这是 Jev/kev 最核心的差异点）。

### 2.4 官方公布的准确率对照（kev 项目自己的 transfer-v4 测试集）

| 模型 | transfer accuracy | Brier |
|---|---|---|
| Jev 1.13（闭源 API） | **0.857** | 0.211 |
| kev-27B | 0.851 | 0.225 |
| kev-9B | 0.820 | 0.289 |
| kev-4B | 0.817 | 0.269 |
| **kev-0.8B** | **0.648** | 0.481 |

> 注意：这是各项目自报数字。Jev 领先，但 kev-27B 已经咬得很近；而 0.8B 与 4B 之间有巨大断层。

---

## 3. 本机配置与选型

### 3.1 硬件

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti，**8 GB VRAM**（bf16 支持） |
| CPU | Intel i7-14700F，20 核 28 线程 |
| 内存 | 32 GB |
| 磁盘 | C: 110GB / E: ~175GB 可用 |
| 软件 | Python 3.13.14、torch 2.14.1+cu126、transformers 5.19.0 |

### 3.2 显存预算（bf16，含激活开销）

| 候选 | 权重体积 | 8GB 可行性 | 判定 |
|---|---|---|---|
| **kev-0.8B**（Qwen3.5-0.8B） | ~1.7 GB | 实测峰值 **1.58 GB** | ✅ **舒适** |
| kev-4B（Qwen3.5-4B） | ~8.7 GB；官卡称 adapter 加载路径峰值 15.9GB、合并后 8.4GB | 实测峰值 **8.24 GB**，靠 WDDM 共享内存回退 | ⚠️ **贴死上限** |
| kev-9B | ~18 GB | — | ❌ 不可行 |
| kev-27B | 51 GB（官卡要求 80GB 卡） | — | ❌ 不可行 |
| Qwen3.8-27B | bf16 56GB / FP8 31GB | — | ❌ 不可行 |
| 通用 Qwen3.5-4B（4bit） | ~2.5 GB | 可行 | ✅ 但无校准概率 |

### 3.3 选型结论

**主力：`jaredpalmer/kev-0.8b`**（Qwen3.5-0.8B-Base + rank-16 LoRA + 指针头）
- 显存只占 1.6GB，留足余量给长文本和批处理
- 0.06–0.07 s/条，400 条评测 20 秒跑完
- 校准优秀，能直接做"高置信自动通过 / 低置信转人工"
- 缺点：绝对准确率比 4B 低不少

**质量上限：`jaredpalmer/kev-4b`**（Qwen3.5-4B-Base 同构）
- 准确率明显更高，但 ~6.7 s/条，且显存贴着 8GB，务必关闭其他占显存程序
- 官卡建议 24GB 显卡；本机属于"能跑但不推荐"

---

## 4. 实测：两个 Kaggle 分类数据集 + 一个零样本探针

### 4.1 重要的诚实声明

**IMDB 和 AG News 都出现在 kev 官方模型卡的训练源列表里**（`stanfordnlp/imdb`、`fancyzhx/ag_news`）。因此这两个任务得到的是 **域内（in-domain）** 成绩，**不是零样本成绩**，不能拿去和 Jev 的零样本能力对比。

为了给出一个不含水分的泛化数据点，额外加了 **SMS Spam Collection**（Kaggle `uciml/sms-spam-collection-dataset`）——它**不在** kev 的任何训练源中，是真正的零样本探针。

| 数据集 | Kaggle 出处 | 题型 | 性质 |
|---|---|---|---|
| IMDB 50K 影评 | `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews` | 二分类 → `noul` | 域内 |
| AG News | `amananandrai/ag-news-classification-dataset` | 4 分类 → `choice` | 域内 |
| SMS Spam Collection | `uciml/sms-spam-collection-dataset` | 二分类 → `noul` | **零样本** |

### 4.2 方法与边界

- 数据：IMDB 400 条（正负各 200）、AG News 400 条（每类 100）、SMS Spam 200 条（各半），种子 `20261013`。
- 评测路径：直接调用 kev 官方的 `Checkpoint.load` → `LocalPredictor` → `evaluate_records`，即项目自己出报告用的同一条路径，**没有自己另写 prompt**。
- 解码：`bf16`（GPU），`torch` 后端；未安装 `flash-linear-attention` / `causal-conv1d`，Gated DeltaNet 走参考 PyTorch 实现（偏慢）。
- 基线：TF-IDF（1–2 gram，sublinear tf）+ 逻辑回归，用该数据集官方 train split 训练，在同一条测试样本上评测。这是"Kaggle 提交榜的真实对手"。
- 所有脚本、原始 `report.json`、逐条预测在 `validation/` 下可复现。

### 4.3 结果

#### IMDB 影评情感（二分类 · `noul` · 域内）

| 模型 | n | Accuracy | Macro-F1 | Brier | ECE | 平均置信度 | coverage@0.9 | acc@0.9 |
|---|---|---|---|---|---|---|---|---|
| **kev-4B** | 400 | **0.960** | 0.9606 | 0.0698 | 0.0249 | 0.9506 | 0.868 | **0.983** |
| kev-0.8B | 400 | 0.935 | 0.9356 | 0.0951 | **0.0174** | 0.9502 | 0.883 | 0.975 |
| TF-IDF + LogReg（监督） | 400 | 0.908 | 0.9075 | 0.1440 | 0.0626 | 0.8484 | 0.500 | 0.990 |

#### AG News 主题（4 分类 · `choice` · 域内）

| 模型 | n | Accuracy | Macro-F1 | Brier | ECE | 平均置信度 | coverage@0.9 | acc@0.9 |
|---|---|---|---|---|---|---|---|---|
| kev-4B | 400 | 0.9075 | 0.9340 | 0.1287 | 0.0410 | 0.9001 | 0.750 | 0.977 |
| kev-0.8B | 400 | 0.9075 | 0.9192 | 0.1400 | **0.0397** | 0.9338 | 0.828 | 0.964 |
| TF-IDF + LogReg（监督） | 400 | **0.9525** | 0.9527 | 0.0877 | 0.0449 | 0.9095 | 0.735 | 0.983 |

#### SMS Spam 垃圾短信（二分类 · `noul` · **零样本**）

| 模型 | n | Accuracy | Macro-F1 | Brier | ECE | coverage@0.9 | acc@0.9 | spam 类召回 |
|---|---|---|---|---|---|---|---|---|
| kev-4B | 200 | 0.880 | 0.8868 | 0.2322 | 0.1587 | 0.060 | 1.000 | 0.82 |
| **kev-0.8B** | 200 | **0.565** | 0.6766 | 0.4975 | 0.0890 | **0.000** | — | **0.22** |
| TF-IDF + LogReg（监督） | 200 | 0.920 | 0.9195 | 0.1217 | 0.0329 | 0.680 | 0.963 | 0.84 |

#### 硬件与速度（实测）

| 模型 / 任务 | 加载 (s) | 每条 (s) | 峰值显存 (GB) |
|---|---|---|---|
| kev-0.8B · IMDB | 4.1 | 0.073 | 1.58 |
| kev-0.8B · AG News | 4.8 | 0.056 | 1.52 |
| kev-0.8B · SMS | 3.8 | 0.047 | 1.52 |
| kev-4B · IMDB | 14.9 | **5.91** | 8.27 |
| kev-4B · AG News | 29.0 | 2.08 | 8.24 |
| kev-4B · SMS | 14.8 | 0.42 | 8.24 |

### 4.4 从数据里读出来的五件事

**1）0.8B 与 4B 的差距不在域内，在域外。**
IMDB 上 4B 领先 2.5 个点（96.0 vs 93.5）；AG News 上两者**完全打平**（都是 90.75%）。但到了没见过的 SMS Spam：**4B 88.0% vs 0.8B 56.5%**。0.8B 在这里基本退化成"永远猜不是垃圾"（spam 召回只有 22%）。**要泛化就必须上 4B**，这与 kev 官方 transfer-v4 的 0.817 vs 0.648 完全吻合。

**2）信号比分数更值钱：0.8B 知道自己在瞎猜。**
在 SMS 上，0.8B 有 **0%** 的样本置信度 ≥0.9，平均置信度只有 0.63 —— 它用置信度明确告诉你"这个领域我不行"。4B 则只有 6% 的样本 ≥0.9，但**这些样本准确率是 100%**。这正是 Jev 系模型的核心价值：**用置信度阈值决定哪些交给自动流程、哪些转人工/转强模型**，而不是只看一个孤立准确率。

**3）校准是真的好，不是营销。**
IMDB 上 0.8B 的 ECE = **0.0174**，4B 在 IMDB 上是 0.0249；作为对比 TF-IDF + LogReg 的 ECE 是 0.0626。域内任务上（这也是它被训练过的题型）校准远好于传统分类器。

**4）选项顺序完全稳。**
对 AG News 的 `choice` 题做 1/2/4 次循环轮转平均：准确率恒为 **0.9000**，**0/200 条答案发生改变**。这是指针头架构的天然性质，也是它相对"提示词 + 生成式 LLM 分类"的实打实优势 —— 后者经常因为选项顺序变化而翻答案。

**5）打不过域内监督学习，这很正常。**
AG News 上 TF-IDF + LogReg 95.25% 完胜两个 kev（90.75%）；SMS 上 92% vs 88%（零样本）。**但只要你有标注数据，微调一个小编码器/线性模型在成本上通常更划算。** kev 的价值在"零标注冷启动"和"自带可用概率"，不在刷分。

**6）精度选择不影响结论。**
SMS 任务在 CPU/fp32 与 GPU/bf16 下跑出完全一致的 56.5%，与 kev 文档"bf16 与 fp32 概率差 ~0.01、argmax 一致"的说法吻合 —— 说明上面这些结论不是数值精度带来的偶然。

### 4.5 边界与未做的事

- 未做零样本 → 微调对比（`kev.train --init_from`）；官方声称新领域 836 条标注可把准确率从 0.33 拉到 0.88，本机可复现但需要标注数据。
- 未安装 `flash-linear-attention` / `causal-conv1d`，Gated DeltaNet 走参考实现，4B 的 5.9 s/条**偏悲观**，装上预计快 2–4×。
- 4B 的 8.24GB 峰值依赖 WDDM 共享内存回退，属于"能跑但踩线"，长时间跑建议关闭其他占显存程序。
- 未评测 `score` 题型（无合适的 Kaggle 有序分类数据）。
- 以上所有数字均为本机本地推理结果，**没有任何 API 调用，也没有虚构任何分数**。

---

## 5. 对 Kaggle 分类任务的建议

**适合的场景**
1. **文本分类 / 打标 / 路由**：类别固定、类别数在几十以内（`choice` 上限 255）。
2. **需要概率而不只是标签**：ECE 在 0.02–0.04 量级，意味着 0.9 的置信度基本可信，可以做选择性预测（低置信转人工 / 转强模型）—— 这在 Kaggle 上是"用置信度阈值换准确率"的免费收益。
3. **不想做 prompt 工程**：state + 选项直接喂，不用调提示词、不用解析输出。

**不适合 / 需要警惕**
1. **它打不过领域内微调**。这张表里 TF-IDF + LogReg 就在 AG News 上赢了 kev-0.8B——只要你有标注数据，微调一个小编码器（DeBERTa/ModernBERT）通常更划算。
2. **绝对准确率受规模限制**。0.8B 的域外转移能力只有 ~0.65（官方数字），别指望它在陌生领域硬扛。
3. **长文本有代价**。IMDB 这类长 state 让 4B 掉到 6.7 s/条；0.8B 也明显变慢。
4. **"不会幻觉"是格式层面的**，选项给错它照样会自信地选错。

**推荐用法（本机）**
- 拿 kev-0.8B 做**零标注冷启动**：先跑一遍，用高置信样本自动打标，再拿去训练自己的小模型。
- 拿 kev-4B 做**教师模型**：对低置信样本复核，或蒸馏到 0.8B。
- 两者都当"免费基线"：如果它已经超过你的手工特征方案，说明任务确实可做；如果连它都做不动，先怀疑数据本身。

**下一步可做的优化**
- 装 `flash-linear-attention` + `causal-conv1d`，混合 Gated DeltaNet 骨架可望提速 2–4×。
- 用 `kev.train --init_from jaredpalmer/kev-0.8b --data <你的标注>` 在自己的 Kaggle 数据上微调（官方称新领域 836 条 support 决策从 0.33→0.88）。
- 用 `/v1/systemone/permute` 做选项顺序扰动，检查 `choice` 的稳定性。

---

## 6. 复现

```
models/                      # 下载的权重
  _download.py               # 从 hf-mirror 拉取（禁用 Xet）
  _fetch.sh                  # 大文件直连镜像（绕开沙箱临时文件清理）
  _localize.py               # 把 head.pt 的 base 改指本地底模
  kev-0.8b/{kev-0.8b-local}/ # adapter + head.pt + tokenizer
  kev-4b/{kev-4b-local}/
  Qwen3.5-0.8B-Base/  Qwen3.5-4B-Base/
third_party/kev/             # git clone jaredpalmer/kev（运行时装 -e 安装）

validation/
  _smoke.py                  # 单条冒烟推理
  prep_data.py               # 构造三个任务的 JSONL
  run_eval.py                # 本地评测（kev 官方路径）
  baseline.py                # TF-IDF + LogReg 基线
  collect.py                 # 汇总成表格
  make_report.py             # 生成 HTML 报告
  _wincompat/fcntl.py        # Windows 垫片（kev.suite 用了 fcntl 文件锁）
  runs/<model>__<task>/      # predictions.jsonl / rows.json / report.json / summary.json
```

常用命令：

```bash
PY=C:/Users/deepg/.workbuddy/binaries/python/envs/default/Scripts/python.exe

# 1) 下载（已禁用 Xet，走 hf-mirror）
$PY models/_download.py kev-0.8b Qwen3.5-0.8B-Base kev-4b Qwen3.5-4B-Base

# 2) 准备数据
$PY validation/prep_data.py --n 400

# 3) 评测
$PY validation/run_eval.py --run models/kev-0.8b-local --task imdb
$PY validation/run_eval.py --run models/kev-4b-local  --task imdb

# 4) 基线与汇总
$PY validation/baseline.py --task imdb
$PY validation/collect.py && $PY validation/make_report.py
```

---

## 7. 参考来源

- TypeSafe AI，*Introducing System One Models & Jev*（2026-09-15）；官方文档 Introduction / System One / Confidence
- Jev Research，第三方独立基准（2026-10-03/04，16,379 次请求）：MMLU-Pro 82.7%、GPQA Diamond 76.5%、ARC-AGI-2 0.0%
- Daniele Teti，*JevBench*（2026-09-22）：Jev 320–340ms、$0.064/600 次判断、置信度分档准确率
- `github.com/jaredpalmer/kev` 与 HF `jaredpalmer/kev-0.8b` / `kev-4b` 模型卡
- `github.com/firelex/jeff`（Jeff 系列）、`madewithjev.com/open-source-jev`（54 个开源替代目录）
- HF 模型卡：`Qwen/Qwen3.5-0.8B-Base`、`Qwen/Qwen3.5-4B-Base`、`Qwen/Qwen3.8-27B`
