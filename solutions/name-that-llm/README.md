# Name That LLM — 方案包

18 个前沿 LLM 的作者归因（40–250 词短文，balanced accuracy，Code Competition）。

## 数据事实
- `train.csv` 2699 行 / 150 个 prompt × 18 个模型（每 prompt 每模型 1 条，类别完全均衡）。
- 隐藏测试集约 4000 条，**prompt 与训练集不重叠**（150×18=2700 已占满训练量）。
- 因此本地验证必须用 `GroupShuffleSplit(prompt_id)`：随机分层切分会把分数打到 0.10–0.13（同 prompt 的主题词
  对 18 个类都是噪声，共享 prompt 反而**降低**准确率），分组切分才是真实水平。

## 结论（本地分组 holdout，18 类 × 30 条 = 540 行）
| 方案 | bal_acc | 备注 |
|---|---|---|
| 随机基线 | 0.056 | 1/18 |
| word+char TF-IDF + LogReg | 0.298–0.318 | 5 折 GroupKFold 与单次 holdout 一致 |
| Qwen3-Embedding-0.6B + LogReg | 0.066 | 语义嵌入被主题支配，无判别力 |
| kev-0.8B LoRA 2 ep | 0.065 | 塌缩（scratch pointer head 未热起来） |
| kev-0.8B LoRA 8 ep, head_lr 1e-3 | 0.230 | 训练 loss 1.48 但过拟合 |
| roberta-large 微调 (80% 数据, 4 ep) | 0.280 | ep0 0.15 → ep3 0.28 仍在升 |
| **prompt 内中心化 TF-IDF + LogReg C=30** | **0.387** | CV 虚高；**LB 只有 0.309**，见下 |
| TF-IDF + 手写风格特征块（标点/markdown 统计） | 0.357 | 与论坛 0.369 接近 |
| **Qwen3.5-4B 冻结嵌入**（7 层均值池化 + LogReg） | **0.415** | 复现论坛 750462 的 0.414 |
| Qwen3.5-9B 冻结嵌入（同法） | 0.372 | **更大模型反而更差** |
| TF-IDF/style + 4B 嵌入融合 (0.5/0.5) | 0.4296 | |
| + 9B 嵌入三项融合 (0.35/0.35/0.30) | 0.4315 | 5-seed 均值 0.4341 ± 0.0073 |
| **+ 3 风格簇先验软加权（geo α=0.5）** | **0.4411** | **5-seed 均值，+0.0070 且 5/5 非负**，见下 |
| TF-IDF + 24k 外部语料 | 0.256–0.276 | **额外数据变差**（见下） |

**前 10% 门槛 ≈ LB 0.58–0.60**（榜上 rank 100 约 0.601）。当前 0.43 差 ~0.15，**未提交**。

## 两级分类（厂商 → 版本）评估：否掉了

问题：先分厂商大类（openai/google/anthropic/…），再分版本，能否带来明显提升？
答：**不能。厂商分组是最差的分组方式**；换成数据驱动的最优分组后，收益也只有 +0.001（噪声内）。

| 一级分组 | k | 一级 bal_acc | 硬级联 18 类 | 软层级 | oracle 上界 |
|---|---|---|---|---|---|
| flat 18 类（无层级） | — | — | **0.4185** | — | — |
| 厂商（先验） | 6 | 0.574 | 0.3852（**−0.033**） | 0.3852 | 0.5852 |
| 混淆聚类 | 2 | 0.965 | 0.4148 | 0.4148 | 0.4370 |
| **混淆聚类** | 3 | 0.914 | **0.4259** | 0.4241 | 0.4667 |
| 混淆聚类 | 4 | 0.807 | 0.4167 | 0.4167 | 0.5056 |
| 混淆聚类 | 6 | 0.642 | 0.4019 | 0.4019 | 0.5852 |
| 混淆聚类 | 12 | 0.519 | 0.3981 | 0.3981 | 0.7685 |

5 个不同 holdout seed 的稳健性（4B 嵌入特征）：
```
flat       0.4230 ± 0.0138
k3_hard    0.4241 ± 0.0183      k3 − flat = +0.0011（逐 seed: +0.0074 +0.0037 −0.0056 −0.0167 +0.0167）
k3_geo     0.4319 ± 0.0161      （软加权，仍在 1σ 内）
vendor_hard 0.3989 ± 0.0104     （−0.024，五个 seed 全部劣化）
```

三点机制解释：
1. **厂商边界不是风格边界**。混淆聚类给出的真实结构是 `gemini 全族` / `gpt+qwen+kimi` / `claude+glm` 三簇：
   gemini 一族可分性极强（一级 0.91），但 claude 和 glm、gpt 和 qwen/kimi 在文风上本就混在一起。
   按厂商切会强行拆开同风格类、合并不同风格类 → 一级只有 0.57。
2. **一级错误不可恢复**：硬级联把一级的 43% 错误直接变成终局错误，这就是 −0.024 的来源。
   软层级（P(厂商)×P(版本|厂商)）保留了不确定性，但也只是回到 flat 水平。
3. **oracle 上界的升高是假象**：k=12 时 oracle 达 0.77，但那等于「已知正确厂商、候选集只剩 1–3 个」，
   是信息泄漏式的上界，不是可达性能。可达性能 = 一级准确率 × 二级条件准确率，一级不够准就没有收益。

理论层面：对**固定特征**，层级结构只是 flat 多项逻辑回归的一个受约束重参数化，不引入新信息；
只有当「每类样本极少、层级先验正确」时，"共享组内统计强度" 带来的方差下降才可能抵过一级误差传播。
这里每类 120 条 train 样本，还不够稀薄。

唯一可保留的用法：把一级后验当作 flat 概率的**软重加权**（`log P_flat + α·log P_group`，α≈0.3–0.5），
多种子 +0.009，勉强正向但仍在噪声内 —— 若要用，只在融合后的最终模型上挂这一项，不要做硬级联。

### 叠加到最优融合上的实测（5 个 holdout seed）

base = `TF-IDF/style 0.35 + 4B 0.35 + 9B 0.30` 的对数平均；geo(α) = `log P_base + α·log P_group(k=3)`。

| 组合 | 5-seed 均值 | 逐 seed 相对 base |
|---|---|---|
| base（原最佳 0.4315） | 0.4341 ± 0.0073 | — |
| **base + geo(α=0.5)** | **0.4411 ± 0.0054** | +0.0111 / +0.0074 / +0.0037 / +0.0130 / 0.0000（**5/5 非负**） |
| base + geo(α=0.3) | 0.4404 ± 0.0057 | 全非负，略低 |
| base + soft-cascade 混入 β=0.3 | 0.4393 ± 0.0145 | +0.0259 / −0.0111 / …（不稳，弃） |

- 增益 **+0.0070**，且**方差下降**（0.0073 → 0.0054）——不是靠单个 seed 的尖峰，是稳定的校准收益。
- α 在 0.2–0.7 是一片平坦平台（0.440–0.441），取 0.5 安全，1.0 反而回落。
- **纯 TF-IDF 也能建这个先验**：同一 3 簇一级模型在 TF-IDF 特征上准确率 0.875–0.908，
  单模 0.3457 → 0.3519（+0.0062，3/3 非负）。所以 CPU-only notebook 也能带上这项，不需要 4B 权重。

结论修正：**厂商分组的层级没用，但「3 个风格簇的先验」作为软校准项确实有稳定小收益**，
只是它补的是 0.007 级别的校准，不是 0.15 级别的判别力。

## 关键经验
1. **prompt 内中心化是 CV 幻觉**：本地 +0.09（0.298→0.387），但 LB 只有 0.309 ≈ 未中心化水平。
   官方明说测试集各模型条数不等、mix 与训练不同 → 隐藏测试集不满足「同一 prompt 18 个模型各一条」。
   论坛 750390 独立复现：聚类+匈牙利 0.344、按类 log-prob 中心化 0.329，都低于 plain 0.353。
   **教训：任何依赖「测试集 prompt 分组」的技巧都不可信，只能靠分组 CV 且不用测试布局技巧。**
2. **冻结嵌入要吃早期层**：layer3 单层 0.3685，7 层拼接 0.4148；深层编码「写什么」，浅层编码「怎么写」。
3. **9B 冻结嵌入 (0.372) 不如 4B (0.415)**：维度更高（4096）在 2159 样本上更易过拟合，4-bit 量化也更伤。
   0.600+ 的 notebook 用的是微调而不是纯嵌入。
4. **那份 24k 外部语料对我们有害**（TF-IDF 0.300→0.256，cap 到 800/类也只 0.276）：
   类别极度不均（4911 vs 71）且 prompt 分布不同，纯监督权重被带偏。
5. `predict_proba` 的列顺序是 `clf.classes_`（字母序），解码时不做映射会得到 0.13 的假分数（踩过一次）。

## 距离门槛还差什么
LB：0.42（当前）→ 0.58–0.60（前 10%）。缺的 0.15 不是调参能补的，需要**判别式微调**：
论坛实测 Qwen3-14B LoRA r16（全线性层，3 epoch，max_len 384）无额外数据 → 0.443 CV / **0.466 LB**；
0.600+ 的 notebook 用 Qwen3.5-9B。下一步计划：在 Kaggle 2×T4 上用官方 train + 该 24k 语料（按类重采样）
微调 Qwen3.5-9B（LoRA，必要时 4-bit QLoRA），用同一套 30-prompt grouped holdout 做闸门。


## 2026-10-22 更新：官方口径否掉「同组匈牙利」，改走公开 14B 五折

**官方 Evaluation 页原文**（决定了哪些解码技巧可用）：
> The number of test passages per model is **not** equal, and the mix of models differs
> between the public and private leaderboard.

—— 所以「每个 prompt 内 18 个模型各出现一次」**不成立**。`decode/`（ward 分组 + 同组匈牙利）
实测预期收益作废，虽然 `sim_decode2.py` 在合成数据上显示 +0.07，但那是建立在错误的
结构假设上（README「关键经验 1」里论坛 750390 的实测也是同向结论）。Sinkhorn 的
「类别均匀」同理不成立，但 twin+Sinkhorn 实测 0.576 > plain 0.520，故保留。

**新成员：`zhuyu2/name-that-llm-ts-lora`（5 个 Qwen3-14B two-stage LoRA 折）**
- 与 fork592 的 loader **完全兼容**：`ts14b_fN/adapter/adapter_model.safetensors` +
  `heads.safetensors`（norms.0–7 / plain / centered）+ `meta.json`；
  `keep=27, max_len=447, layers=[2,4,6,8,12,16,20,24]`，标签顺序与 `LABELS` 完全一致。
- 因此 `lora_features(ckpt, BASE_14B_DIR, texts, 8)` 直接可用，只需给 `model_sources` 挂 14B base。

**权重（无 OOF 时的稳妥做法）**：公开权重 8B .226 / 14B .429 / svc 1.021，且 twin 比恰为
0.35（.079/.226 = .150/.429）。新增五折每折 0.07（合计 0.35），twins 按同比例 0.0245；
prvsiyan 两 seed 各 0.11。所有成员 log-prob 分开存 `MEM` / `TWIN_MEM`，末单元一次
pass 产出 `submission_all`（提交用）/ `noprv` / `old`（复现 0.576 用）三个文件。

## 提交物
- `ntl_tfidf.ipynb`（方案 A，纯 sklearn，CPU，无需外部数据集）- `ntl_blend.ipynb`（方案 B，TF-IDF + roberta-large，需 `ibillxia/ntl-roberta` 数据集）
- **`ntl_final.ipynb`（方案 C，当前最佳）**：TF-IDF/style(0.4) + 4B 7 层嵌入(0.6) + 3 簇先验(α=0.6)。
  需私有数据集 `ibillxia/ntl-qwen4b`（8.96GB：4B bf16 权重 + 训练嵌入）。
  Kaggle 上 fp16 单卡 T4 跑得动（189 ms/text，7 层 mean-pool），5-seed CV **0.445**。
  生成器 `build_notebook_final.py`，kernel id `ibillxia/ntl-final`。
- `ntl_train4b.ipynb`（训练 kernel，4B fp16 LoRA + 18 类头，每 epoch 报 grouped holdout）

## 脚本
| 脚本 | 作用 |
|---|---|
| `baseline_tfidf.py` | 分层 vs 分组 CV 对照、多种线性模型 |
| `sweep_tfidf.py` | n-gram / C 网格 |
| `prep_kev.py` | 生成 kev 格式 JSONL（train/holdout/full） |
| `eval_kev.py` | kev holdout 评估（balanced acc + 混淆） |
| `embed_clf.py` / `diag_emb.py` | Qwen3-Embedding 路线（已否决） |
| `finetune_encoder.py` | roberta-large / ModernBERT 分类微调（`--full 1` 训全量） |
| `test_centering.py` | 中心化增益验证（oracle prompt_id） |
| `test_center_kmeans.py` | 聚类还原分组的可行性 |
| `test_multik.py` / `dryrun_tfidf.py` | 多 k 平均 + notebook 逻辑本地干跑 |
| `blend3.py` | 三模型权重网格 |
| `hier_clf.py` | 两级分类（厂商 → 版本）：hard/soft/oracle/geo 对照，`--feat tfidf|emb` |
| `hier2.py` | 公平 oracle（只算 16 个真有二级的类）、先验替换、免训练二级、融合空间层级 |
| `hier_groups.py` | 用 OOF 混淆矩阵做层次聚类，扫 k=2…12 的分组方式与组数 |
| `hier_seeds.py` | 5 个 holdout seed 的稳健性验证（k=3 级联 vs flat vs 厂商级联） |
| `hier_blend.py` | k=3 组先验叠加到三路融合（geo / soft-cascade），5 seed |
| `hier_blend_tfidf.py` | 纯 TF-IDF 上的 k=3 组先验（CPU-only 可行性） |
