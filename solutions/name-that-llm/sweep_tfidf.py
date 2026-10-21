# Quick hyperparameter sweep for the TF-IDF baseline (grouped CV only).
import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score

df = pd.read_csv("data/train.csv")
X_text = df["text"].values
y = df["model"].values
groups = df["prompt_id"].values
cv = GroupKFold(n_splits=5)
splits = list(cv.split(X_text, y, groups))

def evaluate(vecs, clf_factory, name):
    y_pred = np.empty_like(y)
    t0 = time.time()
    for tr_idx, va_idx in splits:
        Ftr, Fva = [], []
        for cfg in vecs:
            vec = TfidfVectorizer(**cfg)
            Ftr.append(vec.fit_transform(X_text[tr_idx]))
            Fva.append(vec.transform(X_text[va_idx]))
        Ftr = hstack(Ftr).tocsr(); Fva = hstack(Fva).tocsr()
        clf = clf_factory()
        clf.fit(Ftr, y[tr_idx])
        y_pred[va_idx] = clf.predict(Fva)
    s = balanced_accuracy_score(y, y_pred)
    print(f"{name:60s} bal_acc={s:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return s

W12 = dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000)
W13 = dict(analyzer="word", ngram_range=(1, 3), sublinear_tf=True, min_df=2, max_features=300000)
C25 = dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=400000)
C26 = dict(analyzer="char_wb", ngram_range=(2, 6), sublinear_tf=True, min_df=2, max_features=500000)
C35 = dict(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_features=500000)

evaluate([W12, C25], lambda: LogisticRegression(C=10, max_iter=3000), "W12+C25 LogReg C=10 (ref)")
evaluate([W12, C26], lambda: LogisticRegression(C=10, max_iter=3000), "W12+C26 LogReg C=10")
evaluate([W12, C35], lambda: LogisticRegression(C=10, max_iter=3000), "W12+C35 LogReg C=10")
evaluate([W13, C25], lambda: LogisticRegression(C=10, max_iter=3000), "W13+C25 LogReg C=10")
evaluate([W12, C25], lambda: LogisticRegression(C=3, max_iter=3000), "W12+C25 LogReg C=3")
evaluate([W12, C25], lambda: LogisticRegression(C=30, max_iter=3000), "W12+C25 LogReg C=30")
evaluate([W12, C25], lambda: LogisticRegression(C=100, max_iter=3000), "W12+C25 LogReg C=100")
evaluate([W12, C25], lambda: LinearSVC(C=0.3), "W12+C25 LinearSVC C=0.3")
evaluate([W12, C25], lambda: LinearSVC(C=3), "W12+C25 LinearSVC C=3")
evaluate([W12, C25], lambda: LogisticRegression(C=10, max_iter=3000, class_weight="balanced"), "W12+C25 LogReg C=10 balanced")
