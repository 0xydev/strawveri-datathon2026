# -*- coding: utf-8 -*-
"""Fold-içi (sızıntısız) target-encoding, yani kategorik etkileşim anahtarları (smoothing'li),
artı denetimsiz GMM cluster olasılıkları feature olarak. CB/LGB base'e etkisini ablation ile ölçtük.
Sonuç: base'i anlamlı iyileştirmedi."""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import model_cv2 as M

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
train = M.train.copy(); y = M.y; TEXTCOL = M.TEXTCOL; CAT = M.CAT; DROP = M.DROP
test = pd.read_csv(os.path.join(D, "test_x.csv"))
T0 = time.time()
def log(s): print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)

a5 = np.load(os.path.join(D, "v5_artifacts.npz"), allow_pickle=True)
dtr = M.add_features(train); dtr["_text_score"] = a5["text_oof"]; dtr["_emb_score"] = a5["emb_oof"]
dte = M.add_features(test);  dte["_text_score"] = a5["text_te"];  dte["_emb_score"] = a5["emb_te"]

# ---- TE anahtar kolonları (etkileşimler)
def add_keys(df):
    df = df.copy()
    df["_k_dept_role"] = df["department"].astype(str) + "|" + df["target_role"].astype(str)
    df["_k_tier_role"] = df["university_tier"].astype(str) + "|" + df["target_role"].astype(str)
    df["_k_tier_dept"] = df["university_tier"].astype(str) + "|" + df["department"].astype(str)
    return df
dtr = add_keys(dtr); dte = add_keys(dte)
TE_KEYS = ["_k_dept_role", "_k_tier_role", "_k_tier_dept", "target_role", "university_tier"]
K_SMOOTH = 20.0

def te_map(df_tr, ytr, key):
    g = pd.DataFrame({"k": df_tr[key].values, "y": ytr}).groupby("k")["y"].agg(["mean", "count"])
    gm = ytr.mean()
    return ((g["count"] * g["mean"] + K_SMOOTH * gm) / (g["count"] + K_SMOOTH)).to_dict(), gm

# ---- GMM (denetimsiz, train+test)
NUMS = [c for c in dtr.columns if dtr[c].dtype != object and c not in DROP and not c.startswith("_k_")]
gm_feats = ["project_quality_score", "technical_interview_score", "_skill_mean", "_soft_mean",
            "_exp_total", "_github_activity", "_text_score", "_emb_score"]
Xg = pd.concat([dtr[gm_feats], dte[gm_feats]], axis=0).fillna(0).to_numpy()
mu, sd = Xg.mean(0), Xg.std(0) + 1e-9
Xg = (Xg - mu) / sd
gmm = GaussianMixture(n_components=8, covariance_type="full", random_state=42, n_init=2).fit(Xg)
P = gmm.predict_proba(Xg)
for i in range(P.shape[1]):
    dtr[f"_gmm_{i}"] = P[:len(dtr), i]; dte[f"_gmm_{i}"] = P[len(dtr):, i]
log(f"GMM hazır (8 bileşen, {len(gm_feats)} feature)")

feats = [c for c in dtr.columns if c not in DROP + [TEXTCOL] + TE_KEYS[:3]]
cat_cols = [c for c in CAT if c in feats]
folds = list(KFold(5, shuffle=True, random_state=42).split(dtr))

def run(model_kind, use_te, use_gmm, label):
    oof = np.zeros(len(y)); t0 = time.time()
    fs = [f for f in feats if use_gmm or not f.startswith("_gmm_")]
    for tr, va in folds:
        Xa = dtr.iloc[tr][fs].copy(); Xv = dtr.iloc[va][fs].copy()
        if use_te:
            for key in TE_KEYS:
                m_, gm_ = te_map(dtr.iloc[tr], y[tr], key)
                Xa[f"_te_{key}"] = dtr.iloc[tr][key].map(m_).fillna(gm_)
                Xv[f"_te_{key}"] = dtr.iloc[va][key].map(m_).fillna(gm_)
        for c in cat_cols:
            Xa[c] = Xa[c].astype(str).fillna("nan"); Xv[c] = Xv[c].astype(str).fillna("nan")
        if model_kind == "cb":
            m = CatBoostRegressor(loss_function="RMSE", depth=5, learning_rate=0.02, iterations=7000,
                                  l2_leaf_reg=3.0, random_seed=42, od_type="Iter", od_wait=150, verbose=0)
            m.fit(Pool(Xa, y[tr], cat_features=cat_cols),
                  eval_set=Pool(Xv, y[va], cat_features=cat_cols), use_best_model=True)
        else:
            for c in cat_cols:
                cu = pd.api.types.union_categoricals([pd.Categorical(Xa[c]), pd.Categorical(Xv[c])]).categories
                Xa[c] = pd.Categorical(Xa[c], categories=cu); Xv[c] = pd.Categorical(Xv[c], categories=cu)
            m = lgb.LGBMRegressor(objective="regression", n_estimators=12000, learning_rate=0.015,
                num_leaves=7, min_child_samples=60, colsample_bytree=0.8, subsample=0.8,
                subsample_freq=1, reg_lambda=1.0, random_state=42, verbosity=-1)
            m.fit(Xa, y[tr], eval_set=[(Xv, y[va])], eval_metric="l2",
                  callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[va] = m.predict(Xv)
    mse = np.mean((np.clip(oof, 0, 100) - y) ** 2)
    log(f"[{label}] OOF={mse:.3f} ({time.time()-t0:.0f}s)")
    return oof, mse

print("=" * 66, flush=True)
log("baseline: CB=75.690  LGB=75.570")
run("cb", True, False,  "CB + TE")
run("cb", False, True,  "CB + GMM")
run("cb", True, True,   "CB + TE + GMM")
run("lgb", True, False, "LGB + TE")
run("lgb", False, True, "LGB + GMM")
run("lgb", True, True,  "LGB + TE + GMM")
