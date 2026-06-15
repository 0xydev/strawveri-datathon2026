# -*- coding: utf-8 -*-
"""Kaggle GPU'da embedding yarışı: 6 yeni aday tek koşuda. T4, ~25-35 dk.
Çıktı: /kaggle/working/embrace_results.npz (geçen adayların oof+te skorları)"""
import os, time
T0 = time.time()
def log(s): print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
import numpy as np, pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA:", torch.cuda.is_available())
DATA = None
for cand in ["/kaggle/input/datathon-2026", "/kaggle/input/competitions/datathon-2026"]:
    if os.path.exists(f"{cand}/train.csv"): DATA = cand; break
if DATA is None:
    for r, _, fs in os.walk("/kaggle/input"):
        if "train.csv" in fs: DATA = r; break
train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test_x.csv")
y = train["career_success_score"].to_numpy(float)
text_tr = train["mentor_feedback_text"].fillna("").tolist()
text_te = test["mentor_feedback_text"].fillna("").tolist()
from sentence_transformers import SentenceTransformer

CANDS = [
    ("gte_ml", "Alibaba-NLP/gte-multilingual-base", True, ""),
    ("jina3", "jinaai/jina-embeddings-v3", True, ""),
    ("arctic2", "Snowflake/snowflake-arctic-embed-l-v2.0", True, ""),
    ("e5li", "intfloat/multilingual-e5-large-instruct", False, "Instruct: Verilen mentör değerlendirmesinin olumluluk düzeyini belirle\nQuery: "),
    ("kalm", "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5", True, ""),
    ("atasoglu", "atasoglu/xlm-roberta-base-nli-stsb-tr", False, ""),
]
out = {}
results = []
for key, name, trust, prefix in CANDS:
    try:
        m = SentenceTransformer(name, device=DEV, trust_remote_code=trust)
        log(f"{key}: encode ({name})")
        E_tr = m.encode([prefix+t for t in text_tr], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        E_te = m.encode([prefix+t for t in text_te], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        oof = np.zeros(len(y))
        for tr, va in KFold(5, shuffle=True, random_state=2024).split(E_tr):
            oof[va] = np.clip(Ridge(alpha=1.0).fit(E_tr[tr], y[tr]).predict(E_tr[va]), 0, 100)
        mse = float(np.mean((oof-y)**2))
        log(f"  {key}: text-only OOF={mse:.2f}")
        results.append((mse, key))
        if mse < 143:
            te_s = np.clip(Ridge(alpha=1.0).fit(E_tr, y).predict(E_te), 0, 100)
            out[f"{key}_oof"] = oof; out[f"{key}_te"] = te_s
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        log(f"  {key}: HATA {str(e)[:150]}")
log("SIRALAMA: " + " | ".join(f"{k}={m:.1f}" for m, k in sorted(results)))
np.savez("/kaggle/working/embrace_results.npz", **out)
log("BİTTİ -> embrace_results.npz")
