# -*- coding: utf-8 -*-
"""Pseudo-labeling'in sızıntısız testi: fold içinde test tahminlerini (pseudo-etiket) düşük
ağırlıkla eğitime ekler; pseudo etiketler val satırlarını hiç görmez (OOF dürüst kalır).
Sonuç: base'i biraz iyileştirir ama meta çeşitliliğini öldürüyor, o yüzden almadık."""
import os, time
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool
import model_cv2 as M

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
train = M.train; y = M.y
test = pd.read_csv(os.path.join(D, "test_x.csv"))
T0 = time.time()
def log(s): print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)

art = np.load(os.path.join(D, "v3_artifacts.npz"), allow_pickle=True)
text_oof = art["text_oof"]; emb_oof = art["emb_oof"]
text_te = art["text_te"]; emb_te = art["emb_te"]

dtr = M.add_features(train); dtr["_text_score"] = text_oof; dtr["_emb_score"] = emb_oof
dte = M.add_features(test);  dte["_text_score"] = text_te;  dte["_emb_score"] = emb_te
feats = [c for c in dtr.columns if c not in M.DROP + [M.TEXTCOL]]
Xtr = dtr[feats].copy(); Xte = dte[feats].copy()
cat_cols = [c for c in M.CAT if c in Xtr.columns]
for c in cat_cols:
    Xtr[c] = Xtr[c].astype(str).fillna("nan"); Xte[c] = Xte[c].astype(str).fillna("nan")

CB = dict(loss_function="RMSE", depth=5, learning_rate=0.02, iterations=7000,
          l2_leaf_reg=3.0, random_seed=42, od_type="Iter", od_wait=150, verbose=0)
folds = list(KFold(5, shuffle=True, random_state=42).split(Xtr))

# pseudo ağırlığı: test satırlarına düşük ağırlık (0.3 ve 0.7 dene), train=1.0
print("=" * 66, flush=True)
log("baseline: pseudo'suz CB OOF=75.690")
for W in [0.3, 0.7]:
    oof = np.zeros(len(Xtr))
    for fi, (tr, va) in enumerate(folds):
        m1 = CatBoostRegressor(**CB)
        m1.fit(Pool(Xtr.iloc[tr], y[tr], cat_features=cat_cols),
               eval_set=Pool(Xtr.iloc[va], y[va], cat_features=cat_cols), use_best_model=True)
        pseudo = np.clip(m1.predict(Xte), 0, 100)
        X2 = pd.concat([Xtr.iloc[tr], Xte], axis=0, ignore_index=True)
        y2 = np.concatenate([y[tr], pseudo])
        w2 = np.concatenate([np.ones(len(tr)), np.full(len(Xte), W)])
        m2 = CatBoostRegressor(**CB)
        m2.fit(Pool(X2, y2, cat_features=cat_cols, weight=w2),
               eval_set=Pool(Xtr.iloc[va], y[va], cat_features=cat_cols), use_best_model=True)
        oof[va] = m2.predict(Xtr.iloc[va])
        log(f"  W={W} fold {fi+1}/5")
    mse = np.mean((np.clip(oof, 0, 100) - y) ** 2)
    log(f"[PSEUDO W={W}] CB(42) OOF={mse:.3f} (pseudo'suz 75.690, Δ={75.690-mse:+.3f})")
