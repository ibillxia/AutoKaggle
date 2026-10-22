# 把 0.601（ntl-ts）与 0.604（高斯栈）合起来 —— 融合方案

两套方案对同一批行输出同一组 18 类的概率矩阵，所以融合在原理上没有障碍，
差别只在**在哪一层融合**与**GPU 时间预算**。

| | ntl-ts（0.601） | 高斯栈 fork604（0.604） |
|---|---|---|
| 骨干 | svc + Qwen3-8B + Qwen3-14B + 五折 14B + prvsiyan×2 | 4 decoder 家族（GLM-4-9B / Qwen3-8B / Qwen2.5-7B / Qwen3-4B）+ 3 encoder + 4 个 LoRA run + 2 个 tokattn + LightGBM stylometry |
| 同组结构用法 | twin：组内**均值中心化**后的 centered 头 | gaussian：组内**联合高斯**（PCA + Ledoit-Wolf 收缩 + 精确边缘 + 温度），比中心化更 expressive |
| 分组来源 | tfidf 余弦 @0.85，average linkage | **内容 + GLM-4 L24 embedding** 混合距离 @0.60 |
| 均衡 | Sinkhorn（类均匀先验） | BALANCE_ALPHA=0.5 的部分均衡 + Full Hungarian |
| 权重来源 | 抄 fork592 配方（手工） | 0.563 notebook 在 prompt-grouped OOF 上 L-BFGS 拟合 |

## 三个融合点（按性价比排序）

### ① 成员级概率融合（最直接，可行性最高）
高斯栈里有现成的钩子：

```python
def add_member(key, prob, rows=None, weight=None):
    LOGIT[:] += weight * np.log(np.clip(prob, 1e-9, 1))
```

把 ts 独有的成员（Qwen3-14B 主成员、五折 14B、prvsiyan×2）作为额外 `add_member`
接进去即可，不需要改高斯栈任何建模代码。**8B/svc 两栈都有，不重复加。**

时间代价（ts 实测）：14B 1388s + 五折 479/100×4 + prv 两 seed ~600s ≈ **3000s**。
高斯栈自身耗时待 fork604b 跑完才能确定；若 ≤5h，总时长远小于 9h 上限。

### ② 换掉 ts 的分组来源（最便宜的高杠杆）
ts 的 twin 依赖同组划分，现在用 tfidf 余弦 @0.85（本地实测 pairwise F1 0.877）。
高斯栈用 GLM-4 L24 表示 + 内容距离（k 自由、阈值 0.60）。
把 GLM 分组喂给 ts 的 twin 头，属于「不改模型只换分组」的改动，预期收益正向且风险低。
注意：**不要**用 k=n/18 的固定簇数——官方口径说每模型条数不等，簇大小不该固定。

### ③ 把 gaussian 组头搬到 14B 特征上（理论上限最高）
高斯栈的 gaussian 头比 ts 的 centered twin 更强（联合分布 + 精确边缘），但只挂在
8B / GLM / 4B / deberta 上（TWIN_W 里 14B 根本没有）。
14B 是当前最强骨干 → 对 14B 的 L24/L28 池化特征拟合一个 gaussian 组头，
大概率是三者里增益最大的，但需要训练（用 train 的 prompt 分组做 CV 拟合）。

## 明确不做的事
- **不启用 Full Hungarian**：官方 Evaluation 页写明每模型测试条数不相等，
  「每组每个模型恰好一条」的假设不成立（论坛实测 0.344 < plain 0.353）。
- 不做纯消融提交（0.520 那次教训）。

## 执行顺序
1. 等 fork604b（0.604 的忠实复现）跑完 → 确认它自己能到 0.60x，并记录各段耗时
2. 做 ①（加成员）→ 提交，目标 ≥0.61
3. 若 ① 增益不足，再做 ②（换分组），最后才考虑 ③（14B gaussian，成本最高）
