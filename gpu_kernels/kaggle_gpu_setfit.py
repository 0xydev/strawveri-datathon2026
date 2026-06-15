# -*- coding: utf-8 -*-
"""Kaggle GPU'da kontrastif fine-tune (SetFit tarzı) metin skoru. T4, ~60-90 dk.
Aşama 1: BERTurk-base kontrastif. Aşama 2 (vakit kalırsa): cosmos-e5-large kontrastif.
Her aşama sonunda kaydeder -> /kaggle/working/setfit_results.npz"""
import os, time, gc, random
T0 = time.time()
def log(s): print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
import numpy as np, pandas as pd, torch
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
torch.manual_seed(42); random.seed(42); np.random.seed(42)
DATA = None
for cand in ["/kaggle/input/datathon-2026", "/kaggle/input/competitions/datathon-2026"]:
    if os.path.exists(f"{cand}/train.csv"): DATA = cand; break
if DATA is None:
    for r, _, fs in os.walk("/kaggle/input"):
        if "train.csv" in fs: DATA = r; break
train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test_x.csv")
y = train["career_success_score"].to_numpy(float)
ttr = train["mentor_feedback_text"].fillna("").tolist()
tte = test["mentor_feedback_text"].fillna("").tolist()
out = {}

def contrastive_score(model_name, key, n_pairs, bs, epochs=1):
    oof = np.zeros(len(y)); te_acc = np.zeros(len(tte))
    folds = list(KFold(5, shuffle=True, random_state=2024).split(ttr))
    for fi, (tr, va) in enumerate(folds):
        m = SentenceTransformer(model_name, device="cuda")
        m.max_seq_length = 192
        rng = np.random.default_rng(fi)
        pairs = []
        for _ in range(n_pairs):
            i, j = rng.choice(tr, 2, replace=False)
            sim = float(np.exp(-abs(y[i]-y[j])/15.0))
            pairs.append(InputExample(texts=[ttr[i], ttr[j]], label=sim))
        dl = DataLoader(pairs, shuffle=True, batch_size=bs)
        loss = losses.CosineSimilarityLoss(m)
        m.fit(train_objectives=[(dl, loss)], epochs=epochs, warmup_steps=100, show_progress_bar=False)
        E_tr = m.encode([ttr[i] for i in tr], batch_size=128, normalize_embeddings=True)
        E_va = m.encode([ttr[i] for i in va], batch_size=128, normalize_embeddings=True)
        E_te = m.encode(tte, batch_size=128, normalize_embeddings=True)
        r = Ridge(alpha=1.0).fit(E_tr, y[tr])
        oof[va] = np.clip(r.predict(E_va), 0, 100)
        te_acc += np.clip(r.predict(E_te), 0, 100) / len(folds)
        log(f"{key} fold {fi+1}/5 val-MSE={np.mean((oof[va]-y[va])**2):.2f}")
        del m, pairs, dl, loss; gc.collect(); torch.cuda.empty_cache()
    mse = np.mean((oof-y)**2)
    log(f"*** {key}: text-only OOF={mse:.2f}")
    out[f"{key}_oof"] = oof; out[f"{key}_te"] = te_acc
    np.savez("/kaggle/working/setfit_results.npz", **out)

contrastive_score("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", "sf_berturk", 16000, 32)
if time.time() - T0 < 7200:
    contrastive_score("ytu-ce-cosmos/turkish-e5-large", "sf_cosmos", 12000, 16)
log("BİTTİ -> setfit_results.npz")
