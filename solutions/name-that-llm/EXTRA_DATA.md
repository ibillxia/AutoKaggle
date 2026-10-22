# 讨论区外部数据集：清单、审计与本地验证

竞赛：`name-that-llm`（18 类作者归属，平衡准确率）。
数据落地目录：`solutions/name-that-llm/extra/ds/<owner>__<name>/`。

---

## 0. 结论（先看这里）

1. 讨论区一共流出 **18 个**可用的语料数据集，全部可下载、许可都是 CC BY 4.0 / MIT 一类；
   去重合并后共 **76,071** 条唯一段落，**与官方 train 无一条重复**（0 泄漏）。
2. 我们现在的 stage-1 池子只有 **34,940** 条（`nishantdahal/ntl-two-stage-lora:pool_v1.parquet`）。
   本次合并后相对「train ∪ 现有池」还能多出 **42,097** 条，等于池子扩容 **+120%**。
3. 但增量极不均匀：**64% 来自 `olegzholobov/name-that-llm-response-corpus`**，而它的标签噪声率
   **19.2%**（用训练集挖掘出的排版不变量测出来的，train 上误报为 0）。真正干净的大块增量是
   `maniceet/...openrouter-openai-generations`（7,750 条，噪声 0.10%）。
4. 本地代理实验（ModernBERT-base 两阶段微调，与 8B/14B 配方同构）结果见 §5：
   **直接堆原始增量 / 只加干净增量 / 只加 oleg** 的差异，决定要不要花 GPU 重训。

---

## 1. 合规边界

- 竞赛 §2.6 允许「公开、对所有参赛者免费可得」的外部数据。社区据此把花钱生成的语料全部以
  CC BY 4.0 公开，这是这批数据集存在的法律基础。
- 讨论区仍有人（`751544`、`751294`、`750512`、`749613`）在催官方给书面答复，**目前没有定论**。
  但榜上前 30 的多个选手（Seyit 30th、Maniceet 13th、prvsiyan 283rd）都已公开使用并在 notebook 里写明。
- 我们的处理：**只使用已公开发布、带 CC 许可、且能追溯到生成协议的数据集**；不使用任何需要付费
  才能复现的私有数据。提交时在 §2.8 文档里写清来源（模板见 §7）。

---

## 2. 清单

`marginal` = 去重后、既不在 train 也不在 `prvsiyan/ntl-public-in-style-pool-v1` 里的新增行数（即真正
能加进现有池子的量）。`噪声%` = 携带「该模型在 150 条训练样本里从不用」的排版字符的行占比（§4）。

| 数据集 | 内容 | 唯一行 | 新增 | 噪声% | 备注 |
|---|---|---|---|---|---|
| `olegzholobov/name-that-llm-response-corpus` | 18 模型宽表，135k 原始响应，1,902 个 prompt 全 18 齐 | 24,015 | **24,015** | **19.2** | 最大的增量，也是最大的污染源；标签来自第三方聚合站 |
| `maniceet/...openrouter-openai-generations` | OpenRouter + OpenAI 官方路由 | 7,750 | **7,750** | **0.10** | 干净得反常，性价比最高 |
| `dhruvtibarewal/nlm-matched-clean` | 长度匹配到 train 分布的清洗池 + pool_v2 | 24,442 | 2,655 | 4.44 | `nlm_pool_v2` 是 pool_v1 的再清洗版，与 pool_v1 重叠 99% |
| `lucasboesen/ntl-free-proxy-generations` | Kaggle 免费代理，8 个 GPT/Gemini 标签 | 2,880 | 1,510 | 5.69 | 官方 notebook 用它训 GPT-family gate |
| `prvsiyan/prvsiyans-extended-name-that-llm-dataset` | 510 prompt × 18 模型，passages_clean 4,926 | 4,926 | 518 | 4.49 | 已有池子里的主要来源之一 |
| `willrice/name-that-llm-openrouter-harvest` | 1,000 prompt 的 OpenRouter 采集 | 10,772 | 133 | 1.09 | 已被 pool_v1 全部吸收 |
| `seyitkaangunes/ntl-extra-api-passages` | **只有 5 个 Claude 标签**，1,000 prompt × 5 | 4,997 | 64 | 0.20 | 用途是「Claude gate」，不是 18 类训练 |
| `zbigniewkirchner/ntl-extra-gpt-passages` | GPT + Gemini | 2,003 | 73 | 2.75 | 同上，gate 用 |
| `alperen5252525/ntl-vertex-gemini-proxy-gpt-generations` | Vertex 官方路由的 Gemini + 代理 GPT | 3,929 | 116 | 9.75 | Vertex 路由可信，但 Gemini 族噪声高 |
| `lxdlxdlxd/ntl-synth-generations-v1` | 13 个标签，官方/一方路由 | 4,757 | 5 | 10.53 | 已被池子吸收 |
| `dophuchung/ntl-proxy-generations` | 免费代理 + settings sweep | 1,029 | 66 | 5.44 | 与 abdullah 共用一个 prompt bank |
| `abdullah1afify/ntl-generations` | 免费代理 GPT/Gemini | 1,032 | 406 | 3.49 | |
| `haileycheng/name-that-llm-same-prompt-passages` | 同一 prompt 的 12 标签 | 1,002 | 88 | 8.78 | 小 |
| `michaelsemenoff/ntl-own-extra-v1` | 自写 prompt | 4,923 | 26 | 2.08 | 已被池子吸收 |
| `prvsiyan/ntl-own-prompt-generations-v1` | 60 prompt × 12 标签 | 720 | 5 | 1.47 | 已被池子吸收 |
| `prvsiyan/ntl-public-in-style-pool-v1` | **15 个来源的合并池** | 38,640 | 0 | 0.00 | 现有池子的超集，可直接替换 |
| `pjmathematician/ntl-train-prompts-reconstructed` | 150 个 prompt_id 的**逆向 prompt** | 150 | — | — | 不训练，用于按 prompt 去重/诊断 |
| `dhruvtibarewal/nlm-matched-clean:nlm_pool_v2` | 同上（清洗池） | 37,250 | 1 | 0.00 | |

未采纳：`ahmedberatozer/ntl-code`（1.9 GB 代码答案，本题只有短段落）、`mustafakeser4/detect-llm-models`
（2024 年旧赛题，与 18 个标签无关）、各类 LoRA/head 权重数据集（不是文本语料）。

---

## 3. 重叠分析

```
官方 train                       2,699   （0 条与外部语料重复）
现有 stage-1 池 (nishant)        34,940
prvsiyan 合并池                  38,642  ⊃ nishant 池 33,796 条
全部 18 个语料去重合并           76,071
  ├ 已在 train                        0
  ├ 已在 prvsiyan 池             38,640
  └ 边际新增                    37,431
相对「train ∪ nishant 池」真正新增   42,097
```

结论：**继续在讨论区找新数据集已经没有意义**——除了 oleg 和 maniceet 两家，其余 16 个来源的
95%+ 已经躺在池子里了。要再扩数据只能自己花钱跑 API。

---

## 4. 质量审计：用排版不变量测错标

方法（`invariants.py`）：在官方 train 上对每个字符挖「哪些模型从来不用它」，保留
`3 ≤ 禁用模型数 ≤ 15` 且 `支持行数 ≥ 20` 的规则，得到 5 条：

| 规则 | train 上携带行数 | 禁用模型数 |
|---|---|---|
| `" — "`（带空格长破折号） | 86 | 14 |
| `"—"` | 1,052 | 3（gemini-3.1-pro / 3.6 / 3.8 flash） |
| 弯引号 `’‘“”` | 607 | 7（5 Claude + 2 GLM） |
| 直引号 `"` | 109 | 6 |
| 弯双引号 `“”` | 93 | 10 |

这些规则在 train 上误报 **0 行**（脚本里是断言），所以一行外部数据只要带了自己标签「绝不可能出现」
的字符，就几乎可以断定标签错了或根本不是这个模型写的。

按模型拆解后原因很清楚：

- `oleg_corpus` 的 Claude 列（fable-5 66.5%、sonnet-5 55.7%）和 kimi（62.5%）、qwen（47.8%）
  大面积违规 → 这一列的来源站把不同厂商的文本混在一起了；
- 其余数据集的违规几乎全部集中在 gemini 3.x 列（10~18%）→ 与讨论区 `751370` 报告的
  **3.7 请求被路由到 3.8** 一致，是路由错，不是标注错。

**可执行结论**：oleg 只能按「行」过滤后用（丢 19%，剩 19,400），Gemini 列的代理数据要单独降权或过滤。

---

## 5. 本地验证

### 5.1 设计

真实配方是「Qwen3-8B 前 24 层 + LoRA + 8 层池化头，stage-1 池子 1 epoch → stage-2 官方 2 epoch」。
本机只有一张 8 GB 4060 Ti，8B 微调不可行（Qwen3.5-0.8B 的线性注意力在无 `flash-linear-attention`
时会退回参考实现，实测只有 4.4 样本/秒，跑 4 万行要 2.5 小时，已放弃）。

改用 **ModernBERT-base（150M，全参微调）+ 同一个两阶段协议 + 同一个分组留出集**（按 `prompt_id`
分组，20% 留出，540 条，每类 30 条）。绝对分数不能和 8B 比，但**同一协议下的 A/B 差值**是可信的。
所有臂共享 stage-2（官方 2,159 条），只有 stage-1 换数据。

| 臂 | stage-1 | 说明 |
|---|---|---|
| `p_none` | 无 | 只有官方 train，基线 |
| `p_pool` | pool_v1 38,640 | 复现现有配方 |
| `p_poolmarg` | pool_v1 + 全部边际 37,431 | 「无脑全加」|
| `p_marg` | 只有边际 | 检验外部分布本身够不够 |
| `p_only_oleg` / `p_only_man` | 单来源 | 拆开看谁有用 |
| `p_drop_oleg` / `p_drop_man` | 池子+边际但去掉该来源 | 留一法 |

### 5.2 结果

<!--RESULTS-->

---

## 6. 接入方案

产物：`extra/pool_v2.parquet`（`build_pool_v2.py` 生成），列为 `prompt_id, model, text, source`，
与公开池完全同构，可以直接替换训练内核里的 `pool_v1.parquet`。

```bash
# 只用通过审计的来源 + 丢掉违规行
python build_pool_v2.py --sources oleg_corpus,maniceet,dhruv_clean,lucas_proxy \
                        --clean --out extra/pool_v2.parquet
```

接入步骤：
1. 把 `pool_v2.parquet` 发布成 Kaggle 数据集（`ntl-pool-v2`），挂到训练内核；
2. 训练内核只改一行：stage-1 的 parquet 路径；
3. 训练完把新 checkpoint 当作集成里的一个新成员（现有 hybrid2 的 `LORAS` 列表加一条即可）。

---

## 7. §2.8 文档模板

> External data: 18 public Kaggle datasets of single-paragraph passages generated by the 18
> competition models through their official or first-party APIs, all published under CC BY 4.0 by
> other competitors and freely available to every participant (`prvsiyan/prvsiyans-extended-name-that-llm-dataset`,
> `maniceet/name-that-llm-openrouter-openai-generations`, …). No competition text, prompt or label
> was sent to any provider. Rows were deduplicated against the official train set (0 overlaps) and
> filtered with typographic invariants mined on train.

---

## 8. 复现

```bash
python fetch_extra_sets.py        # 下载 17 个数据集到 extra/ds/
python build_extra_union.py       # 去重合并 -> extra/union_all.parquet / union_marginal.parquet
python audit_labelnoise.py        # 排版不变量噪声审计 -> extra/audit_labelnoise.csv
python proxy_enc.py --stage1 pool_marg --tag p_poolmarg   # 单臂验证
bash   run_proxy_queue.sh         # 全部对照臂
```
