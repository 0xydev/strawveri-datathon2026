# -*- coding: utf-8 -*-
"""Hedef-dönüşümü çeşitliliği: aynı ağaçları 3 farklı hedefle eğitir. Hedefler: oran y/(pq+10),
lineer-artık (y - Ridge), oran-CB. Çeşitlilik hedeften gelir; her biri meta için bir OOF kolonu.
Çıktı: data/tgtvar.npz."""
import numpy as np, pandas as pd, time
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import model_cv2 as M
T0 = time.time()
def log(s): print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)

D = "data"
train = pd.read_csv(f"{D}/train.csv"); test = pd.read_csv(f"{D}/test_x.csv")
y = train["career_success_score"].to_numpy(float)
a5 = np.load(f"{D}/v5_artifacts.npz", allow_pickle=True)
dtr = M.add_features(train); dtr["_text_score"]=a5["text_oof"]; dtr["_emb_score"]=a5["emb_oof"]
dte = M.add_features(test);  dte["_text_score"]=a5["text_te"];  dte["_emb_score"]=a5["emb_te"]
feats = [c for c in dtr.columns if c not in M.DROP + [M.TEXTCOL]]
Xtr = dtr[feats].copy(); Xte = dte[feats].copy()
cat_cols = [c for c in M.CAT if c in Xtr.columns]
for c in cat_cols:
    cu = pd.api.types.union_categoricals([pd.Categorical(Xtr[c].astype(str)), pd.Categorical(Xte[c].astype(str))]).categories
    Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cu); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cu)
Xh = Xtr.copy(); Xht = Xte.copy()
for c in cat_cols: Xh[c] = Xh[c].cat.codes; Xht[c] = Xht[c].cat.codes
med = Xh.median(); Xh = Xh.fillna(med).astype(float); Xht = Xht.fillna(med).astype(float)
sc = StandardScaler().fit(pd.concat([Xh, Xht])); Zh = sc.transform(Xh); Zht = sc.transform(Xht)
folds = list(KFold(5, shuffle=True, random_state=42).split(Xtr))

def lgb_fit(Xa, ya, Xv, yv):
    m = lgb.LGBMRegressor(objective="regression", n_estimators=12000, learning_rate=0.015, num_leaves=7,
        min_child_samples=60, colsample_bytree=0.8, subsample=0.8, subsample_freq=1, reg_lambda=1.0,
        random_state=42, verbosity=-1)
    m.fit(Xa, ya, eval_set=[(Xv, yv)], eval_metric="l2", callbacks=[lgb.early_stopping(300, verbose=False)])
    return m

out = {}
# ---- 1) ORAN hedefi: y / (project_quality+10); pq en güçlü kolon (r=0.54)
den_tr = train["project_quality_score"].to_numpy(float) + 10
den_te = test["project_quality_score"].to_numpy(float) + 10
yr_t = y / den_tr
oof = np.zeros(len(y)); te = np.zeros(len(test))
for tr, va in folds:
    m = lgb_fit(Xtr.iloc[tr], yr_t[tr], Xtr.iloc[va], yr_t[va])
    oof[va] = m.predict(Xtr.iloc[va]) * den_tr[va]
    te += m.predict(Xte) * den_te / 5
log(f"ORAN-LGB: OOF={np.mean((np.clip(oof,0,100)-y)**2):.3f}")
out["ratio_oof"] = oof; out["ratio_te"] = te

# ---- 2) ARTIK hedefi: y - Ridge_lineer(X) (lineer kısım ayrı, LGB artığı öğrenir)
lin_oof = np.zeros(len(y)); lin_te = np.zeros(len(test))
for tr, va in folds:
    r = Ridge(alpha=10.0).fit(Zh[tr], y[tr])
    lin_oof[va] = r.predict(Zh[va]); lin_te += r.predict(Zht)/5
res_t = y - lin_oof
oof = np.zeros(len(y)); te = np.zeros(len(test))
for tr, va in folds:
    m = lgb_fit(Xtr.iloc[tr], res_t[tr], Xtr.iloc[va], res_t[va])
    oof[va] = m.predict(Xtr.iloc[va]) + lin_oof[va]
    te += (m.predict(Xte) + lin_te) / 5
log(f"ARTIK-LGB: OOF={np.mean((np.clip(oof,0,100)-y)**2):.3f}")
out["resid_oof"] = oof; out["resid_te"] = te

# ---- 3) ORAN hedefi CB ile (çeşitlilik)
oof = np.zeros(len(y)); te = np.zeros(len(test))
for tr, va in folds:
    m = CatBoostRegressor(loss_function="RMSE", depth=5, learning_rate=0.02, iterations=7000,
                          l2_leaf_reg=3.0, random_seed=42, od_type="Iter", od_wait=150, verbose=0)
    m.fit(Pool(Xtr.iloc[tr], yr_t[tr], cat_features=cat_cols),
          eval_set=Pool(Xtr.iloc[va], yr_t[va], cat_features=cat_cols), use_best_model=True)
    oof[va] = m.predict(Xtr.iloc[va]) * den_tr[va]
    te += m.predict(Xte) * den_te / 5
log(f"ORAN-CB: OOF={np.mean((np.clip(oof,0,100)-y)**2):.3f}")
out["ratio_cb_oof"] = oof; out["ratio_cb_te"] = te

np.savez(f"{D}/tgtvar.npz", **out)
log("kaydedildi: tgtvar.npz")
