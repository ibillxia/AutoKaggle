# TF-IDF + linear baseline for Name That LLM (18-class authorship attribution)
# Compares StratifiedKFold vs GroupKFold(prompt_id) to expose topic leakage.
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
import time, sys

DATA = "data/train.csv"
SEED = 42

def run(name, vec_word_cfg, vec_char_cfg, clf_factory, splitter_name):
    df = pd.read_csv(DATA)
    X_text = df["text"].values
    y = df["model"].values
    groups = df["prompt_id"].values

    if splitter_name == "stratified":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = cv.split(X_text, y)
    else:
        cv = GroupKFold(n_splits=5)
        splits = cv.split(X_text, y, groups)

    y_pred = np.empty_like(y)
    t0 = time.time()
    for tr_idx, va_idx in splits:
        tr_texts = X_text[tr_idx]
        va_texts = X_text[va_idx]

        feats = []
        for cfg in (vec_word_cfg, vec_char_cfg):
            if cfg is None:
                continue
            vec = TfidfVectorizer(**cfg)
            Ftr = vec.fit_transform(tr_texts)
            Fva = vec.transform(va_texts)
            feats.append((Ftr, Fva))
        Ftr = hstack([f[0] for f in feats]).tocsr()
        Fva = hstack([f[1] for f in feats]).tocsr()

        clf = clf_factory()
        clf.fit(Ftr, y[tr_idx])
        y_pred[va_idx] = clf.predict(Fva)

    score = balanced_accuracy_score(y, y_pred)
    print(f"[{splitter_name:9s}] {name:40s} bal_acc={score:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return score

WORD = dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000)
CHAR = dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=400000)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    configs = [
        ("word(1-2)+char(2-5) LogReg C=10", WORD, CHAR, lambda: LogisticRegression(C=10, max_iter=2000, n_jobs=-1)),
        ("word(1-2)+char(2-5) LogReg C=30", WORD, CHAR, lambda: LogisticRegression(C=30, max_iter=2000, n_jobs=-1)),
        ("word(1-2)+char(2-5) LinearSVC C=1", WORD, CHAR, lambda: LinearSVC(C=1)),
        ("char(2-5) only LogReg C=10", None, CHAR, lambda: LogisticRegression(C=10, max_iter=2000, n_jobs=-1)),
        ("word(1-2) only LogReg C=10", WORD, None, lambda: LogisticRegression(C=10, max_iter=2000, n_jobs=-1)),
        ("word+char ComplementNB", WORD, CHAR, lambda: ComplementNB(alpha=0.3)),
    ]
    for splitter in ("stratified", "grouped"):
        for name, w, c, f in configs:
            if which == "all" or which in name:
                run(name, w, c, f, splitter)
