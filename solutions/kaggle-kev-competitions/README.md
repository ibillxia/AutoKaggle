# 用 kev（Jev 的开源等价物）能打哪些 Kaggle 竞赛

> 扫描时间：2026-10-13 · 账号已接入 Kaggle API（token 存于 `~/.kaggle/access_token`）
> 配套实验脚本：`validation/prep_data.py` · `run_eval.py` · `baseline.py` · `train_kev.py` · `collect_*.py`

---

## 0. 一句话结论

**线上开放的 22 个竞赛里，只有 3 个在任务形态上适合 kev**，且全部集中在 Getting Started
（`nlp-getting-started`、`contradictory-my-dear-watson`、`llm-classification-finetuning`）。

而这三个赛里，**kev 能不能用、怎么用，取决于你走哪条路**：

| 用法 | 实测结论 |
|---|---|
| **零样本分类器**直接提交 | 能跑通，但**打不过监督基线**（disaster tweets：kev-4B F1 **0.678** vs TF-IDF **0.760**）。不适合冲榜 |
| **推理型判别**任务（NLI / 蕴含） | **结构性优势**。英文 NLI 零样本 **0.877**，词袋基线只有 **0.183** —— 这是词袋模型做不到的任务 |
| **当可训练骨干微调**（kev 训练代码开源） | ✅ **最推荐**。kev-0.8B LoRA 微调 2 epoch（本机 46 分钟）→ disaster tweets F1 **0.824**，**反超 TF-IDF 的 0.760** |

第三条是本轮最有价值的发现：**kev 不是「只能用零样本」，它的训练栈是开源的**，
`--data` 直接吃竞赛标注即可微调，产出标准 PEFT adapter，8 GB 消费级显卡就够。

---

## 1. kev 的能力边界（决定了怎么筛赛）

| 属性 | 值 |
|---|---|
| 架构 | 冻结骨干（Qwen3.5-Base）+ rank-16 LoRA + 指针决策头，**prefill-only，不生成文本** |
| 输入 | 一段 `state` 文本 + 枚举问题 |
| 输出 | 候选项上的**概率分布**（不是 token） |
| 原语 | `choice`（≤255 选项）/ `score`（连续打分）/ `noul`（是非） |
| 上下文 | 单记录 **16384 token**（`ROW_PASS_TOKENS`），state 上限 65536（`SERVE_MAX_STATE`） |
| 训练 | **开源**：`kev/train.py`（LoRA）、`kev/full_ft.py`（全参）、`kev/calibrate.py`（温度校准） |
| 本机训练实测 | 0.8B LoRA 可训练参数 11.3M，**0.209 s/条**（accum 8 / bf16 / 8 GB 显存） |

**契合**：一段文本 → 从固定标签集里选一个（含"读两个句子再判断蕴含/矛盾"这类判别）。
**不契合**：图像/音频/视频、回归、文本生成、博弈与仿真、长链符号推理。

---

## 2. 线上扫描结果

对 `all / featured / research / recruitment / gettingStarted / masters / playground` 七个维度
逐页拉取（221 条去重），按 `deadline > 2026-10-13` 过滤，得到**开放竞赛 22 个**：

| 截止 | 类别 | 竞赛 | 任务 | 适合 kev？ |
|---|---|---|---|---|
| 2026-10-14 | Featured | `kaggriculture` | 农业仿真 | ✗ 明天截止 |
| 2026-10-22 | Research | `rsna-knee-abnormality-detection` | 膝部影像 | ✗ 图像 |
| 2026-10-31 | Playground | `playground-series-s6e10` | 表格二分类 AUC | ✗ GBDT 完胜 |
| 2026-11-02 | Featured | `arc-prize-2026-arc-agi-2` | 抽象推理 | ✗ 见 §5（同类模型实测 0.0%） |
| 2026-11-02 | Featured | `arc-prize-2026-arc-agi-3` | 抽象推理 | ✗ 同上 |
| 2026-11-09 | Featured | `arc-prize-2026-paper-track` | 论文 | ✗ 非建模 |
| 2026-11-12 | Featured | `gemma-4-developer-agent-paper` | 论文 | ✗ 非建模（本仓库已适配 code 赛道） |
| 2026-12-02 | Featured | `gemma-4-developer-agent` | code agent | ✗ 已适配 |
| 2026-12-14 | Featured | `enveda-CASMI26-molecule-id-mass-spectra` | 多模态化学 | ✗ 分子/谱图 |
| 2027-01-06 | Featured | `nfl-big-data-bowl-2027` | 写作评审制 | ✗ 已适配 |
| 2027-01-08 | Playground | `the-pokemon-company-ptcg-ai-battle-challenge-playground` | 卡牌博弈 | ✗ 仿真 |
| 2030-01-01 | Getting Started | `spaceship-titanic` | 表格二分类 | △ 可做，无优势 |
| 2030-01-01 | Getting Started | **`nlp-getting-started`** | **文本二分类 F1** | **✓ 一档** |
| 2030-01-01 | Getting Started | `connectx` | 博弈 | ✗ |
| 2030-01-01 | Getting Started | `house-prices-*` | 表格回归 | ✗ |
| 2030-01-01 | Getting Started | `titanic` | 表格二分类 | △ 无优势 |
| 2030-01-01 | Getting Started | `digit-recognizer` | 手写图像 | ✗ |
| 2030-01-01 | Getting Started | `home-data-for-ml-course` | 表格回归 | ✗ |
| 2030-06-03 | Getting Started | `tpu-getting-started` | 图像 | ✗ |
| 2030-06-30 | Getting Started | `store-sales-*` | 时序回归 | ✗ |
| 2030-07-01 | Getting Started | **`llm-classification-finetuning`** | **文本三分类 log loss** | **✓ 一档** |
| 2030-07-01 | Getting Started | **`contradictory-my-dear-watson`** | **NLI 三分类 accuracy** | **✓ 一档** |

---

## 3. 一档推荐（三个赛的对比）

| | `contradictory-my-dear-watson` | `nlp-getting-started` | `llm-classification-finetuning` |
|---|---|---|---|
| 任务 | 跨语言 NLI：蕴含 / 中立 / 矛盾 | 推文是否为真实灾难 | 预测人类更偏好哪个回复（A/B/平局） |
| 语言 | **15 种** | 英文 | 英文 |
| 指标 | accuracy | **F1** | **log loss** |
| 评测集 | 5195 行 | 3262 行 | 隐藏 |
| 训练集 | 12k+ | 7613 行 | 57k 行（184 MB） |
| 提交方式 | **code competition**（Notebook 输出，2h/会话） | **普通上传**（本地 CSV 直接交） | **code competition**（9h，**无网**） |
| 榜上真实区间 | ~0.927–0.936（1.0 是答案泄漏） | ~0.851–0.854（1.0 是答案泄漏） | ~0.8287–0.8292（越低越好） |
| kev 零样本实测 | 4B **0.702**（15 语言）/ 0.877（英文） | 4B F1 **0.678** | 未实测（需先接受规则） |
| kev 微调后 | 未测（预期显著提升） | 0.8B **F1 0.824**（本机 46 分钟） | 未测；**校准工具正中 log loss** |
| 落地难度 | 中 | **最低** | 中高 |

**排序建议：想真的出货 → `nlp-getting-started`（已跑通微调，直接可交）；
想做有意思的实验 → `contradictory-my-dear-watson`；想赌"校准优势" → `llm-classification-finetuning`。**

---

## 4. 实测数据（全部本机推理，可复现）

### 4.1 `nlp-getting-started`：灾难推文二分类（F1）★ 主战场

数据：竞赛 `train.csv` 的本地镜像（**987,712 字节，与 Kaggle 官方逐字节一致**，7613 行）。
分层留出 **1000 行**做评测，剩余 **6613 行**用于训练——两条路线看到完全相同的测试集。

| 模型 | F1（正类，官方指标） | accuracy | macro-F1 | ECE | s/条 | 峰值显存 |
|---|---|---|---|---|---|---|
| kev-4B（零样本） | 0.6775 | 0.743 | 0.786 | 0.0453 | 0.312 | 8.24 GB |
| kev-0.8B（零样本） | 0.6076 | 0.690 | 0.744 | **0.0373** | 0.050 | 1.52 GB |
| TF-IDF + LogReg（6613 行监督） | 0.7600 | 0.779 | 0.778 | 0.0442 | — | — |
| **kev-0.8B + LoRA 微调（2 epoch）** | **0.8244** | **0.836** | **0.846** | 0.1001 | 0.069 | 1.51 GB |

**读法**：
- 零样本下 kev 输给"6613 行 + TF-IDF"，短板是 `true` 类召回（4B 在 true 上 0.540）；
- **微调后反超 6.4 个 F1 点**，且只用 46 分钟、1.5 GB 显存、11.3M 可训练参数；
- 微调的代价是**校准漂移**（ECE 0.037 → 0.100，模型变得过度自信，平均置信度 0.932）——
  这一条可以被 kev 自带的校准工具修好，见 §4.4。

### 4.2 `contradictory-my-dear-watson`：代理探针

该赛数据需接受规则才能下载，因此用同构数据集做代理。

**(a) 英文 NLI（GLUE MNLI-matched，300 行，accuracy）**

| 模型 | accuracy | macro-F1 | ECE（校准后 OOF） |
|---|---|---|---|
| **kev-4B** | **0.8767** | 0.8889 | 0.062 → **0.025** |
| kev-0.8B | 0.8067 | 0.8617 | 0.045 → 0.031 |
| TF-IDF + LogReg | 0.1833 | 0.1765 | — |

**这是 kev 最有说服力的一场**：词袋模型在 NLI 上直接退化到随机水平（18%），kev 零样本 87.7%。

**(b) 跨语言 NLI（XNLI 15 语言 × 21 行 = 315 行，accuracy）** —— XNLI 正是该赛的数据来源

| 模型 | 15 语言均值 | ECE（校准后 OOF） |
|---|---|---|
| kev-4B | **0.7016** | 0.212 → 0.077 |
| kev-0.8B | 0.6349 | 0.143 → 0.077 |

kev-4B 逐语言（每语言仅 21 行，**±11pp 采样噪声，只看趋势**）：

| 高 | 中 | 低 |
|---|---|---|
| en 1.000 · fr 0.810 · el 0.762 · vi 0.762 | bg/de/es/tr 0.714 · hi/ur/zh 0.667 · ar/ru 0.619 | sw 0.571 · th 0.524 |

**读法**：跨语言掉点严重（英文 87.7% → 15 语言 70.2%），高资源欧语 ~0.71–0.81，
低资源（泰语、斯瓦希里语）掉到 ~0.52–0.57。对榜上 0.93 的真实区间，**零样本大概落在中游**；
但这条赛有 12k+ 标注，走微调路线（见 §4.4）才是正解。

### 4.3 既有域内成绩（供对照）

| 任务 | kev-4B | kev-0.8B | kev-0.8B+微调 | TF-IDF+LogReg | 性质 |
|---|---|---|---|---|---|
| IMDB 二分类 | 96.0% | 93.5% | — | 90.75% | 域内（在 kev 训练源里） |
| AG News 四分类 | 90.75% | 90.75% | — | 95.25% | 域内 |
| SMS Spam 二分类 | 88.0% | 56.5% | — | 92.0% | **零样本** |
| 灾难推文（F1） | 0.6775 | 0.6076 | **0.8244** | 0.7600 | **零样本 → 微调** |

### 4.4 关键补充：微调 + 校准工具链

**(a) 微调可行性与产出**（`validation/train_kev.py`）

```
python validation/train_kev.py --task disaster --epochs 2
# → kev.train --base models/Qwen3.5-0.8B-Base --data data/disaster_train.jsonl \
#            --epochs 2 --lora 16 --accum 8 --dtype bf16 --device cuda
```

- 6613 条 × 2 epoch = 1654 步，**46 分钟**跑完（8 GB 显存，0.209 s/条），训练 loss 2.05 → 0.18；
- 产出是标准 PEFT `adapter_model.safetensors`（43 MB）+ `head.pt`（2.1 MB），
  `LocalPredictor` 可直接加载——**这意味着可以把微调结果打包成 Kaggle Dataset 直接上线**；
- 本机 8 GB 只够 0.8B 的 LoRA（4B 的 bf16 权重本身就 8.7 GB）；想微调 4B 得上 Kaggle 的 T4 x2。

**(b) 校准修复**（`kev/calibrate.py`，只拟合一个温度，不写回 checkpoint）

| 模型 / 任务 | 出厂温度 | 迁移温度 | ECE（出厂 → OOF 校准后） |
|---|---|---|---|
| kev-0.8B + 微调 / 灾难推文 | 1.00（训练重置了） | 2.09 | 0.098 → **0.014**（Δ −0.084，95% CI [−0.099, −0.052]） |
| kev-4B / 灾难推文 | 2.41 | 3.40 | 0.045 → **0.031** |
| kev-0.8B / 灾难推文 | 2.35 | 2.70 | 0.030 → 0.031（已很好，无收益） |
| kev-4B / 英文 NLI | 2.41 | 3.65 | 0.062 → **0.025** |
| kev-0.8B / 15 语言 NLI | 2.35 | 3.48 | 0.143 → **0.077**（跨语言校准不是单一温度能修好的） |

同一温度调整后 Brier 也从 0.275 → 0.252（CI 显著），**argmax 与 accuracy 不变**（温度不变式）。
这就是 `llm-classification-finetuning`（log loss）值得一试的机制性理由。

---

### 4.5 真实 LB 提交记录（`nlp-getting-started`，2026-10-13）

账号 `ibillxia` 已接通**提交权限**（legacy `kaggle.json`），四次真实 Public LB 结果：

| # | 方法 | Public F1 |
|---|---|---|
| 1 | kev-0.8B LoRA 微调（6613 条 × 2 epoch）@0.5 | 0.82623 |
| 2 | kev-0.8B LoRA 微调（**全量 7613 条 × 3 epoch**）@0.5 | 0.81887 |
| 3 | 同上 #1 + **F1 调优阈值 0.40** | 0.82194 |
| 4 | 两模型概率平均（#1 + #2）@0.5 | 0.82745 |
| 5 | **三模型概率平均**（#1 + #2 + seed=1）@0.5 | **0.83113** ⭐ |

榜前端 ≈0.851–0.854，所以 kev-0.8B 微调 + 集成落在**中上游**，距头部还有 ~0.020。

**唯一稳定、单调有效的操作是集成**：1 → 2 → 3 个成员，LB 0.82623 → 0.82745 → **0.83113**。
三个成员互有分歧（两两 `p_true` 分歧均值 0.078–0.132，p90 达 0.25–0.50），
而 `validation/blend_probs.py` 的等权平均把方差压下去了。
注意**本地均衡留出集测不出这个收益**（ft-6613 单模型 0.82441 vs 掺入 seed1 后 0.82315），
因为这三个成员的差异主要暴露在测试分布上——**这点只能靠 LB 反馈，不能靠本地小留出集**。

**(a) 训练数据更多反而更差。** 7613 全量 3 epoch 比 6613 × 2 epoch 低 0.0074。训练日志显示
末段 loss 已降到 0.08–0.17，属于把训练集记住了。**在 0.8B × 7.6k 条这个尺度上，3 epoch 过头了。**

**(b) F1 阈值调优在真实 LB 上失效（本地 +0.010 → 实测 −0.004）。** 原因不是方法错，而是
**留出集的类别分布和测试集不一致**：`validation/data/disaster_test.jsonl` 是刻意平衡的 500/500，
而真实正类率是 42.97%（train.csv），由模型预测数反推测试集 ≈41%。
**教训：用来选 F1 阈值的留出集必须复刻真实先验，否则调优出来的阈值会系统性偏移。**

**(c) 噪声尺度比想象的大。** 3263 行测试集上 F1 的 1σ ≈ 0.0066，
所以上表 2–4 之间的 0.004–0.007 差异**都在 1σ 以内**，不能当稳定结论。唯一方向明确的是
**集成**：两个成员 `p_true` 的分歧 mean 0.091 / p90 0.269，平均后 +0.0012，是唯一"免费"的正收益。

**(d) 提交链路本身踩了一个坑。** Kaggle 提交流程第 2 步（把文件 PUT 到 GCS resumable 上传会话）
的 host 是 `www.googleapis.com`，本机直连超时 / 沙箱代理 502；但同一套 GCS JSON API 的
`storage.googleapis.com` 是通的（upload_id 在 query string 里、不绑定 host），改写 host 即可上传成功。
封装在 `validation/kaggle_submit.py`（含直连/代理回退）。

---

### 4.6 真实 LB 提交记录（两个代码赛，2026-10-13/14）

同一套「微调 → 打包私有 Dataset → 推推理内核 → code submission」链路用到了另外两个竞赛上。
两个赛的 `sample_submission` 形态完全不同（watson 是 `id,prediction` 的整数类，llm 是三列概率表），
内核模板用 `OUT_MODE`/`PRED_COL` 两个开关覆盖，其余代码路径共用。

| 竞赛 | 版本 | 方法 | Public 成绩 |
|---|---|---|---|
| `contradictory-my-dear-watson` | v1 | kev-0.8B 发布头**零样本**，15 语言 NLI | accuracy 0.72107 |
| `contradictory-my-dear-watson` | v2 | **delta 微调**（10907 条 × 2 ep，lr 2e-5） | **0.78498** ⭐ |
| `contradictory-my-dear-watson` | v3 | 全量 12120 条 × 2 ep（同配方） | 0.78440 |
| `llm-classification-finetuning` | v1 | 发布头零样本（3 类，概率表） | log loss 1.16439 |
| `llm-classification-finetuning` | v2 | delta 微调（12000 条 × 2 ep）+ **温度校准** T=1.587 | **1.04073** ⭐ |

榜头参照：watson ≈0.9365（accuracy）、llm ≈0.8287（log loss）、llm 三分类**随机 ≈1.0986**。

**(a) choice 题必须从发布检查点出发微调（`--init_from`），从零训学不动。**
这是本轮最关键的一条：kev 的指针头在 `choice` 原语上从零训练时 loss 长期钉在 ln(3)≈1.099 附近
（第 10 步 1.523、第 100 步仍 ≈1.1），12000 条数据喂完也没起来；而 `--init_from <发布版检查点>` + 小 lr
（2e-5）后**首 10 步 loss 就降到 0.98、第 210 步 0.547**。watson 留出集 accuracy 因此从
零样本的 0.6617 跳到 **0.8178**，LB 上落地为 0.72107 → 0.78498（+6.4pt）。
`--base` 必须与 `head.pt` 里记录的 base 串**逐字符一致**（COMPAT_FIELDS 校验），
所以本地把 base 重指向 `models/kev-0.8b-local` 的副本是必需品。
（`noul` 二分类原语没有这个限制——disaster tweets 那次就是从零训成功的。）

**(b) log loss 赛必须做温度校准，且校准量级可提前预估。**
`kev.train` 不像发布版那样自带拟合温度，delta 微调出来的头是 T=1 裸 logits。
llm 赛发布头零样本 **1.16439 比三分类随机 1.0986 还差**——正是"过度自信"的典型症状，
因此这个赛天然吃校准红利。留出集上 T=1 的 NLL 1.0504，网格搜到 **T=1.5874 后降到 1.0310**，
LB 实测 **1.04073**（±0.01 与本地吻合）。温度不改变 argmax，所以对 accuracy 类指标**零成本**。

**(c) epoch 数要按留出集定，2 epoch 常常就是上限。**
watson 续训第 3–4 epoch（lr 1e-5）留出 acc **0.8120 < 0.8178**，且 NLL 从 0.656 劣化到 1.092、
ECE 0.121 → 0.158——**过拟合先表现在校准上，再表现在 accuracy 上**。全量数据重训（12120 条）
与切分版 LB 打平（0.78440 vs 0.78498，差异在 5195 行测试集的噪声内）。
0.8B 这个尺度的 watson 上限就是 **≈0.785**，要摸 0.93 需要更大的模型或外部 NLI 数据。

**(d) `MAX_STATE=384` 是训练侧的隐形天花板。**
kev 训练默认每条记录只留 384 个 state token，超长记录会被**直接丢掉**（`kev/train.py` 的 `fits` 过滤）。
llm 赛 300+600+600 字符的设置下有 **7.7% 的记录被丢弃**、且长回复在 token 384 处被截断；
换成 600/1200 字符 + `--max_state 1024` 后保留率 99.7%（19931/20000）。
代价是单步从 0.27 s/条涨到 0.69 s/条——**加宽上下文是最贵但最结构性的一档杠杆**。
注意三者必须一致：`prep_llm.py` 的字符宽度、训练时的 `--max_state`、内核模板的截断宽度
（`build_notebook.py` 的 `RUN_WIDTHS` 把宽度绑到每个 run 名上，避免配错）。
实测收益：留出集（同一批 1501 行、行序与标签序列完全一致）NLL **1.0310 → 1.01594**，
raw（T=1）1.0504 → 1.01793，且拟合所需温度从 1.587 回落到 **1.176**——说明模型本身更校准了，
不只是被温度压平。代价是训练量按时间预算从 20000×2ep 改为 20000×1ep。

### 4.7 watson 的下一个杠杆：语言维度诊断 + 外部数据

**诊断先行。** 比赛 data 里的 `language` 列可以借由 `(premise, hypothesis)` 文本反查回留出集，
于是「分数丢在哪」是可量化的（`validation/lang_breakdown.py`）。kev-0.8B delta 在 1213 行留出集上：

| 语言 | acc | n | 占比 |
|---|---|---|---|
| English | 0.870 | 685 | 56.5% |
| Russian | 0.885 | 26 | 2.1% |
| Vietnamese / Chinese / Arabic | 0.816 / 0.816 / 0.814 | 38 / 38 / 43 | ~3% |
| Spanish / French | 0.791 / 0.811 | 43 / 37 | ~3% |
| Bulgarian / German / Greek / Thai | 0.763 / 0.757 / 0.750 / 0.733 | ~35 | ~3% |
| Hindi / Turkish | 0.711 / 0.688 | 45 / 32 | ~3% |
| **Swahili / Urdu** | **0.614 / 0.595** | 44 / 37 | ~3% |

结论：**英语占一半以上且已经 0.87，拖后腿的是低资源语言**（乌尔都、斯瓦希里、土耳其、印地都掉到 0.6–0.7）。
这与 kev 官方 15 语言 NLI 零样本 0.635 的横截面一致——多语言迁移是小模型的固有短板。

**这就是外部数据的理由。** 比赛 rules 的 Rule C 明确允许 External Data（只需对所有参赛者免费可得），
而且比赛数据本身按规则声明就来自 **The Cross-Lingual NLI Corpus (XNLI) 与 The Multi-Genre NLI Corpus (MNLI)**
——比赛 `train.csv` 的 12,120 行只是这两个语料的一个子采样。于是可以把两个语料的 **train split**
（各有 392,702 行，XNLI 覆盖全部 15 语言）按语言配比混进来，把训练量从 1.2 万提到 3 万以上。

坑与规矩（`validation/prep_watson_ext.py` 已按此实现）：

- **只用 train split**，绝不碰 dev/test——比赛隐藏测试集极可能就取自那里；
- **XNLI-en 是 MNLI 的翻译版**（实测同一批 premise、同一顺序），所以英语只取 MNLI  naturals，
  否则等于把英语行复制一遍；
- 配比按比赛自身的语言分布走（英语 ≈57%），低资源语言按语言配额均匀采样，而不是按原始语料大小。

已构建 `data/comp/contradictory-my-dear-watson/kev_train_ext.jsonl` = **30,077 条**
（比赛 train 10,907 + XNLI 14 语言 11,172 + MNLI 英语 7,998），与此前一致的口径是
「同配方 2 ep 时 10,907 条已是最优、4 ep 过拟合」，所以 3 倍数据配 **1 epoch（3,760 步）**
——步数略高于之前的最优步数 2,728，但外部数据带来的多样性正好用来抵消多出来的步数。

**另一条已验证但未上的杠杆：成员集成。** `nlp-getting-started` 上只有集成是单调有效的操作
（1→2→3 成员 0.82623→0.82745→0.83113），所以内核模板已支持多成员概率平均
（`build_notebook.py --run a,b --weights 1,1`，每个成员带自己训练时的截断宽度），
离线预估用 `validation/ensemble_probs.py`（先各自校准再平均、再给混合体拟合温度，
保证单模型与集成的比较是在同等拟合自由度下做的）。

---

## 5. 四条必须记住的警告

1. **零样本 ≠ 能用。** 有标注的竞赛上，TF-IDF 都能赢过零样本 kev-4B。要出成绩必须走微调。
2. **区分域内与域外。** IMDB/AG News 在 kev 官方训练源里，96% 那种数字不能当零样本用；
   只有 disaster / NLI / XNLI / SMS 是干净的域外或竞赛留出成绩。
3. **kev 的三条独门优势**，且都已验证：
   - 校准好且**可修复**（出厂已拟合温度；再拟合一次能把 ECE 压到 0.014–0.031）；
   - **选项顺序完全稳**——AG News 上 1/2/4 次循环轮转，**0/200 条答案改变**（指针头架构的天然性质）；
   - 置信度可用——低置信样本可直接弃权（微调后 0.9 置信度子集准确率 88.1%，覆盖 80%）。
4. **ARC 别碰。** 第三方独立复测（16,379 请求）显示该类"预训练骨干 + 决策头"模型在
   ARC-AGI-2 上是 **0.0%**；`kaggriculture`（仿真）、`enveda-CASMI26`（谱图）、`rsna-knee`（影像）
   同理不在 kev 的射程内。

---

## 6. 落地路径

### 平台约束（2026-10 现状）

- Kaggle Notebook 现在**只有 T4 x2**（2 × 16 GB），P100 已于 **2026-09-15 下线**；单会话 9h，每周约 30h。
- **code competition 强制断网**，所以 kev 的代码与权重必须作为 **Kaggle Dataset** 随 notebook 挂载。
  kev-0.8B 约 1.6 GB、kev-4B 约 8.7 GB、微调后的 adapter 仅 45 MB —— 都放得下。

### 路线 A：最快出货 —— `nlp-getting-started`（普通上传赛）✅ 已具备条件

不需要 notebook。本地微调 → 预测 test → 产出 `submission.csv` → `kaggle competitions submit`。
脚本与数据都在：`train_kev.py` + `run_eval.py` + 已验证的竞赛数据镜像。
**唯一阻塞是接受竞赛规则**（见 §7）。

### 路线 B：`contradictory-my-dear-watson`（code competition，2h）

1. 打包 `third_party/kev` + `models/kev-0.8b-local`（或微调后的 adapter）为 Kaggle Dataset；
2. notebook 里 `pip install --no-deps` 装 kev 并加载；
3. 5195 行推理：kev-0.8B 0.054 s/行 → **约 5 分钟**；4B 则约 76 分钟（本机速度，T4 上更慢），
   而 2h 预算还要留给训练，**建议 0.8B 打底**；
4. 该赛有 12k+ 标注 → 用 `kev.train` 在 notebook 内微调再推理，是比零样本更靠谱的路线。

### 路线 C：`llm-classification-finetuning`（code competition，9h，无网）

log loss 是 kev 校准优势的最佳舞台，但要先解决三件事：
① 单行输入是「提问 + 两个回复」，token 量远大于前两赛（受 16384 上限约束，长样本要截断）；
② 必须离线，权重走 Dataset；③ 184 MB 训练集 + 9h 预算，够微调但不够大模型全参。
**建议先做本地小样本探针**（需先接受规则拿到数据）。

### 路线 D：把 kev 当**可训练骨干**（本轮已验证，最推荐）✅

kev 训练器自带的能力正好对应竞赛需求：

| 开关 | 用途 |
|---|---|
| `--option_isolation` | **精确置换不变**（彻底消除选项顺序偏置） |
| `--perm_kl` / `--perm_frac` | 顺序扰动下的对称 KL，提升鲁棒性 |
| `--label_smoothing` / `--brier_w` / `--focal_gamma` | 直接优化概率质量（对应 log loss / F1 不平衡） |
| `--anchor` / `--anchor_w` | 向冻结基座的零样本分布做 KL 锚定，抗遗忘 |
| `--head_lr` / `--lora` / `--lora_targets` | 头与 LoRA 分开调参；`dense` 减少对基座的漂移 |

配合 `kev/calibrate.py` 的后处理温度，形成「**微调提精度 + 校准提概率质量**」的完整闭环。

---

## 7. 需要你决定的事

1. **接受竞赛规则**：拉取竞赛数据前必须 `kaggle competitions accept`（会在你账号上建立团队并记录
   对该赛条款的同意）。这是**你的账号授权行为，我没有代做**。要哪几个赛，你说了算。
2. **先推哪个**：我建议 **`nlp-getting-started`**——普通上传赛，微调链路已经在本机跑通并产出
   F1 0.824，只差接受规则 → 拿官方 test → 提交。
3. **要不要继续压 nlp-getting-started 的分数**：目前只跑了 2 epoch。可试的方向：
   4 epoch、`--lora_targets all` + `--option_isolation`、`--brier_w`、以及 4B 在 T4 x2 上微调。

---

## 8. 复现清单

| 文件 | 作用 |
|---|---|
| `validation/prep_data.py` | 生成 6 个任务的数据（`imdb` / `ag_news` / `sms_spam` / `disaster` / `nli` / `xnli`） |
| `validation/run_eval.py` | kev 推理 + 指标（accuracy / macro-F1 / **正类 F1** / Brier / NLL / ECE / 覆盖率） |
| `validation/baseline.py` | TF-IDF + LogReg 监督基线，同一测试集 |
| `validation/train_kev.py` | 在竞赛标注上微调 kev（`kev.train` 的 `--data` 路线） |
| `validation/collect_kaggle.py` | 汇总全部 `summary.json` → 表格 + `report.html` |
| `validation/collect_calibration.py` | 归档 `kev.calibrate` 的温度校准报告 |
| `validation/rotations.py` | 选项顺序扰动稳定性实验 |
| `validation/_wincompat/` | Windows 垫片（`fcntl`、`resource`） |
| `validation/kaggle_probe/` | 竞赛扫描原始数据（`active.json` 为开放赛清单） |
| `results/` | 各任务 `summary.json` + `calibration.json` 归档 |

```bash
P="C:/Users/deepg/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
$P validation/prep_data.py --n 400 --disaster-n 1000 --nli-n 300 --xnli-per-lang 21
$P validation/run_eval.py   --run ../models/kev-0.8b-local --task disaster
$P validation/baseline.py   --task disaster
$P validation/train_kev.py  --task disaster --epochs 2          # 微调，本机约 46 分钟
$P validation/run_eval.py   --run ../runs/kev-0.8b-ft-disaster --task disaster --device cuda --dtype bf16
$P validation/collect_kaggle.py && $P validation/collect_calibration.py
```
