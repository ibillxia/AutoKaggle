## 验证结果

### IMDB 影评情感（二分类 / noul，域内）

| 模型 | n | Accuracy | Macro-F1 | Brier | ECE | 平均置信度 | coverage@0.9 | acc@0.9 |
|---|---|---|---|---|---|---|---|---|
| kev-0.8b-local | 400 | 0.935 | 0.9356 | 0.0951 | 0.0174 | 0.9502 | 0.8825 | 0.9745 |
| kev-4b-local | 400 | 0.96 | 0.9606 | 0.0698 | 0.0249 | 0.9506 | 0.8675 | 0.9827 |
| tfidf-logreg | 400 | 0.9075 | 0.9075 | 0.144 | 0.0626 | 0.8484 | 0.5 | 0.99 |

- `kev-0.8b-local` per-class acc: {"false": 0.945, "true": 0.925}
- `kev-4b-local` per-class acc: {"false": 0.975, "true": 0.945}
- `tfidf-logreg` per-class acc: {"false": 0.925, "true": 0.89}

### AG News 主题（4 分类 / choice，域内）

| 模型 | n | Accuracy | Macro-F1 | Brier | ECE | 平均置信度 | coverage@0.9 | acc@0.9 |
|---|---|---|---|---|---|---|---|---|
| kev-0.8b-local | 400 | 0.9075 | 0.9192 | 0.14 | 0.0397 | 0.9338 | 0.8275 | 0.9637 |
| kev-4b-local | 400 | 0.9075 | 0.934 | 0.1287 | 0.041 | 0.9001 | 0.75 | 0.9767 |
| tfidf-logreg | 400 | 0.9525 | 0.9527 | 0.0877 | 0.0449 | 0.9095 | 0.735 | 0.983 |

- `kev-0.8b-local` per-class acc: {"Business": 0.84, "Sci/Tech": 0.88, "Sports": 1.0, "World": 0.91}
- `kev-4b-local` per-class acc: {"Business": 0.79, "Sci/Tech": 0.92, "Sports": 1.0, "World": 0.92}
- `tfidf-logreg` per-class acc: {"Business": 0.93, "Sci/Tech": 0.94, "Sports": 1.0, "World": 0.94}

### SMS Spam 垃圾短信（二分类 / noul，零样本探针）

| 模型 | n | Accuracy | Macro-F1 | Brier | ECE | 平均置信度 | coverage@0.9 | acc@0.9 |
|---|---|---|---|---|---|---|---|---|
| kev-0.8b-local | 200 | 0.565 | 0.6766 | 0.4975 | 0.089 | 0.6286 | 0.0 | None |
| kev-4b-local | 200 | 0.88 | 0.8868 | 0.2322 | 0.1587 | 0.7213 | 0.06 | 1.0 |
| tfidf-logreg | 200 | 0.92 | 0.9195 | 0.1217 | 0.0329 | 0.8976 | 0.68 | 0.9632 |

- `kev-0.8b-local` per-class acc: {"false": 0.91, "true": 0.22}
- `kev-4b-local` per-class acc: {"false": 0.94, "true": 0.82}
- `tfidf-logreg` per-class acc: {"false": 1.0, "true": 0.84}

### 硬件与速度

| 模型 | 任务 | 加载(s) | 每条(s) | 峰值显存(GB) | 底模 |
|---|---|---|---|---|---|
| kev-0.8b-local | imdb | 4.1 | 0.073 | 1.58 | E:\Bill\AI\AutoKaggle\models\Qwen3.5-0.8B-Base |
| kev-0.8b-local | ag_news | 4.8 | 0.056 | 1.52 | E:\Bill\AI\AutoKaggle\models\Qwen3.5-0.8B-Base |
| kev-0.8b-local | sms_spam | 3.8 | 0.047 | 1.52 | E:\Bill\AI\AutoKaggle\models\Qwen3.5-0.8B-Base |
| kev-4b-local | imdb | 14.9 | 5.912 | 8.27 | E:\Bill\AI\AutoKaggle\models\Qwen3.5-4B-Base |
| kev-4b-local | ag_news | 29.0 | 2.076 | 8.24 | E:\Bill\AI\AutoKaggle\models\Qwen3.5-4B-Base |
| kev-4b-local | sms_spam | 14.8 | 0.416 | 8.24 | E:\Bill\AI\AutoKaggle\models\Qwen3.5-4B-Base |
| tfidf-logreg | imdb | - | - | - | - |
| tfidf-logreg | ag_news | - | - | - | - |
| tfidf-logreg | sms_spam | - | - | - | - |