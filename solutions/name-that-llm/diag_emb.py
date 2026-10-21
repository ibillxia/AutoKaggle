# Diagnose why embeddings fail: try C grid, unnormalized, LDA, KNN, SVM on saved npz.
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

d = np.load("emb_train.npz", allow_pickle=True)
E, y, g = d["emb"], d["y"], d["groups"]
cv = list(GroupKFold(n_splits=5).split(E, y, g))

def ev(clf_factory, name, X=None):
    X = E if X is None else X
    yp = np.empty_like(y)
    for tr, va in cv:
        c = clf_factory(); c.fit(X[tr], y[tr]); yp[va] = c.predict(X[va])
    s = balanced_accuracy_score(y, yp)
    print(f"{name:46s} bal_acc={s:.4f}", flush=True)

for C in (0.1, 1, 10, 100):
    ev(lambda C=C: LogisticRegression(C=C, max_iter=2000), f"LogReg C={C} (normed)")
ev(lambda: LinearSVC(C=1), "LinearSVC (normed)")
ev(lambda: KNeighborsClassifier(10, metric="cosine"), "KNN k=10 cosine (normed)")
ev(lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), "LDA shrink (normed)")

# unnormalized: load raw last hidden? npz has normalized; re-derive is not possible -> skip
# PCA-whitened variant
from sklearn.decomposition import PCA
for k in (64, 128, 256):
    P = PCA(n_components=k, whiten=True).fit(E)
    ev(lambda: LogisticRegression(C=10, max_iter=2000), f"LogReg C=10 PCA-white {k}d", P.transform(E))
