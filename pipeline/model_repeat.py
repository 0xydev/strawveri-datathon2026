# -*- coding: utf-8 -*-
"""Tekrarlı K-fold: pipeline'ı verilen FOLD_SEED ile uçtan uca koşar (çoklu-seed ortalaması için).
Kullanım: FOLD_SEED=43 python model_repeat.py  ->  data/repeat_<seed>.npz (final_oof, final_te)."""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb
import model_cv2 as M

FOLD_SEED = int(os.environ.get("FOLD_SEED", "43"))
BASE_CAUCHY = os.environ.get("BASE_CAUCHY", "0") == "1"
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
train = M.train.copy(); y = M.y
test = pd.read_csv(os.path.join(D, "test_x.csv"))
yr = train["application_year"].to_numpy(); yr_te = test["application_year"].to_numpy()
T0 = time.time()
def log(s): print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
log(f"FOLD_SEED={FOLD_SEED}")

a5 = np.load(os.path.join(D, "v5_artifacts.npz"), allow_pickle=True)
text_oof, text_te = a5["text_oof"], a5["text_te"]   # metin skorları fold-bağımsız (seed 2024)
emb_oof, emb_te = a5["emb_oof"], a5["emb_te"]
e5_oof, e5_te = a5["e5_oof"], a5["e5_te"]; mini_oof, mini_te = a5["mini_oof"], a5["mini_te"]

dtr = M.add_features(train); dtr["_text_score"]=text_oof; dtr["_emb_score"]=emb_oof
dte = M.add_features(test);  dte["_text_score"]=text_te;  dte["_emb_score"]=emb_te
if os.environ.get("TEXTYR", "0") == "1":   # yıl x metin-skoru etkileşimleri
    for d_ in (dtr, dte):
        _yi = (d_["application_year"]-2018).astype(float)
        d_["_text_yr"] = d_["_text_score"]*_yi
        d_["_emb_yr"] = d_["_emb_score"]*_yi
    log("TEXTYR=1: _text_yr + _emb_yr eklendi")
feats = [c for c in dtr.columns if c not in M.DROP + [M.TEXTCOL]]
Xtr = dtr[feats].copy(); Xte = dte[feats].copy()
cat_cols = [c for c in M.CAT if c in Xtr.columns]
for c in cat_cols:
    Xtr[c] = Xtr[c].astype(str).fillna("nan"); Xte[c] = Xte[c].astype(str).fillna("nan")
Xtr_l = Xtr.copy(); Xte_l = Xte.copy()
for c in cat_cols:
    cu = pd.api.types.union_categoricals([pd.Categorical(Xtr_l[c]), pd.Categorical(Xte_l[c])]).categories
    Xtr_l[c] = pd.Categorical(Xtr_l[c], categories=cu); Xte_l[c] = pd.Categorical(Xte_l[c], categories=cu)
Xtr_h = Xtr_l.copy(); Xte_h = Xte_l.copy()
for c in cat_cols:
    Xtr_h[c] = Xtr_h[c].cat.codes; Xte_h[c] = Xte_h[c].cat.codes
Xtr_h = Xtr_h.fillna(-999); Xte_h = Xte_h.fillna(-999)
sc = StandardScaler().fit(pd.concat([Xtr_h, Xte_h]))
Ztr = sc.transform(Xtr_h.fillna(Xtr_h.median())); Zte = sc.transform(Xte_h.fillna(Xtr_h.median()))
folds = list(KFold(5, shuffle=True, random_state=FOLD_SEED).split(Xtr))

FOLD_W = {}
if BASE_CAUCHY:
    log("iç-CV Cauchy ağırlıkları üretiliyor")
    for fi, (tr, va) in enumerate(folds):
        inner = np.zeros(len(tr))
        for itr, iva in KFold(3, shuffle=True, random_state=7).split(tr):
            m = lgb.LGBMRegressor(objective="regression", n_estimators=3000, learning_rate=0.02,
                num_leaves=7, min_child_samples=60, colsample_bytree=0.8, subsample=0.8,
                subsample_freq=1, reg_lambda=1.0, random_state=42, verbosity=-1)
            m.fit(Xtr_l.iloc[tr[itr]], y[tr[itr]], eval_set=[(Xtr_l.iloc[tr[iva]], y[tr[iva]])],
                  eval_metric="l2", callbacks=[lgb.early_stopping(150, verbose=False)])
            inner[iva] = m.predict(Xtr_l.iloc[tr[iva]])
        rsd = y[tr] - inner
        zs = pd.Series(rsd).groupby(yr[tr]).transform("std").to_numpy()
        zz = np.abs(rsd)/zs
        FOLD_W[fi] = 1.0/(1.0+(zz/2.0)**2)
    log("ağırlıklar hazır")

def run_family(fit_one, Xa, Xe, seeds, label):
    oof = np.zeros(len(y)); te = np.zeros(len(test)); t0 = time.time()
    for fi, (tr, va) in enumerate(folds):
        wts = FOLD_W.get(fi) if BASE_CAUCHY else None
        for s in seeds:
            m = fit_one(s, Xa.iloc[tr] if hasattr(Xa,"iloc") else Xa[tr], y[tr],
                        Xa.iloc[va] if hasattr(Xa,"iloc") else Xa[va], y[va], wts)
            oof[va] += m.predict(Xa.iloc[va] if hasattr(Xa,"iloc") else Xa[va]) / len(seeds)
            te += m.predict(Xe) / (len(seeds)*len(folds))
    log(f"  {label} OOF={np.mean((np.clip(oof,0,100)-y)**2):.3f} ({time.time()-t0:.0f}s)")
    return oof, te

_CBHPO = os.environ.get("HPOCB", "")   # HPO'lu CB parametreleri
_cbp = dict(depth=6, learning_rate=0.0175, iterations=6956, l2_leaf_reg=5.48, random_strength=1.65) if _CBHPO=="1" else \
       dict(depth=5, learning_rate=0.02, iterations=7000, l2_leaf_reg=3.0)
def fit_cb(s, Xa, ya, Xv, yv, w=None):
    m = CatBoostRegressor(loss_function="RMSE", **_cbp,
                          random_seed=s, od_type="Iter", od_wait=150, verbose=0)
    m.fit(Pool(Xa, ya, cat_features=cat_cols, weight=w), eval_set=Pool(Xv, yv, cat_features=cat_cols), use_best_model=True)
    return m
def fit_lgb(s, Xa, ya, Xv, yv, w=None):
    m = lgb.LGBMRegressor(objective="regression", n_estimators=12000, learning_rate=0.015, num_leaves=7,
        min_child_samples=60, colsample_bytree=0.8, subsample=0.8, subsample_freq=1, reg_lambda=1.0,
        random_state=s, verbosity=-1)
    m.fit(Xa, ya, sample_weight=w, eval_set=[(Xv, yv)], eval_metric="l2", callbacks=[lgb.early_stopping(300, verbose=False)])
    return m
def fit_xgb(s, Xa, ya, Xv, yv, w=None):
    m = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=12000, enable_categorical=True,
        tree_method="hist", early_stopping_rounds=300, random_state=s, verbosity=0,
        max_depth=3, learning_rate=0.015, min_child_weight=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0)
    m.fit(Xa, ya, sample_weight=w, eval_set=[(Xv, yv)], verbose=False)
    return m
def fit_et(s, Xa, ya, Xv, yv, w=None):
    return ExtraTreesRegressor(n_estimators=1246, max_depth=20, min_samples_leaf=3, max_features=0.88,
                               n_jobs=-1, random_state=s).fit(Xa, ya, sample_weight=w)
def fit_hgb(s, Xa, ya, Xv, yv, w=None):
    return HistGradientBoostingRegressor(max_iter=1610, learning_rate=0.0114, max_leaf_nodes=22,
        min_samples_leaf=94, l2_regularization=18.5, early_stopping=True, validation_fraction=0.15,
        random_state=s).fit(Xa, ya, sample_weight=w)
def fit_lasso(s, Xa, ya, Xv, yv, w=None): return Lasso(alpha=0.05, max_iter=5000).fit(Xa, ya, sample_weight=w)
def fit_svr(s, Xa, ya, Xv, yv, w=None): return SVR(C=10, epsilon=1.0, gamma="scale").fit(Xa, ya, sample_weight=w)
def fit_dart(s, Xa, ya, Xv, yv, w=None):
    m = lgb.LGBMRegressor(objective="regression", boosting_type="dart", n_estimators=1948, learning_rate=0.0614,
        num_leaves=7, min_child_samples=39, drop_rate=0.2485, colsample_bytree=0.537, reg_lambda=2.37,
        random_state=s, verbosity=-1)
    m.fit(Xa, ya, sample_weight=w)
    return m
def fit_rf(s, Xa, ya, Xv, yv, w=None):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=494, max_depth=30, min_samples_leaf=3, max_features=0.65,
                                 n_jobs=-1, random_state=s).fit(Xa, ya, sample_weight=w)

log("base aileleri")
cb_o, cb_t = run_family(fit_cb, Xtr, Xte, [42,7,2025,1,3], "CBx5")
lg_o, lg_t = run_family(fit_lgb, Xtr_l, Xte_l, [42,7,2025,1,3], "LGBx5")
xg_o, xg_t = run_family(fit_xgb, Xtr_l, Xte_l, [42,7,2025], "XGBx3")
et_o, et_t = run_family(fit_et, Xtr_h, Xte_h, [42], "ET")
hg_o, hg_t = run_family(fit_hgb, Xtr_h, Xte_h, [42], "HGB")
la_o, la_t = run_family(fit_lasso, Ztr, Zte, [42], "Lasso")
sv_o, sv_t = run_family(fit_svr, Ztr, Zte, [42], "SVR")
da_o, da_t = run_family(fit_dart, Xtr_l, Xte_l, [42], "DART")
rf_o, rf_t = run_family(fit_rf, Xtr_h, Xte_h, [42], "RF")

log("eşik classifier'ları")
def tail_clf(target):
    po = np.zeros(len(y)); pt = np.zeros(len(test))
    for tr, va in folds:
        mc = CatBoostClassifier(loss_function="Logloss", depth=5, learning_rate=0.03, iterations=2000,
                                l2_leaf_reg=8.0, random_seed=42, od_type="Iter", od_wait=150, verbose=0)
        mc.fit(Pool(Xtr.iloc[tr], target[tr], cat_features=cat_cols),
               eval_set=Pool(Xtr.iloc[va], target[va], cat_features=cat_cols), use_best_model=True)
        po[va] = mc.predict_proba(Xtr.iloc[va])[:,1]; pt += mc.predict_proba(Xte)[:,1] / len(folds)
    return po, pt
p_oof, p_te = tail_clf((y==100).astype(int))
p95_o, p95_t = tail_clf((y>=95).astype(int))
p90_o, p90_t = tail_clf((y>=90).astype(int))
plo_o, plo_t = tail_clf((y<=45).astype(int))

log("meta + çok-eşikli final")
cols = [cb_o,lg_o,xg_o,et_o,hg_o,p_oof,text_oof,emb_oof,e5_oof,mini_oof,
        train["project_quality_score"].to_numpy(),train["technical_interview_score"].to_numpy(),yr.astype(float),la_o,sv_o,da_o,rf_o]
cols_t = [cb_t,lg_t,xg_t,et_t,hg_t,p_te,text_te,emb_te,e5_te,mini_te,
        test["project_quality_score"].to_numpy(),test["technical_interview_score"].to_numpy(),yr_te.astype(float),la_t,sv_t,da_t,rf_t]
# varsa TabPFN/TabM (bu fold seed'iyle üretilmiş)
for fam in ["tabpfn", "tabm"]:
    p = os.path.join(D, f"{fam}_full_{FOLD_SEED}.npz" if FOLD_SEED != 42 else f"{fam}_full.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        cols.append(z["oof"]); cols_t.append(z["te"])
        log(f"  + {fam} dahil ({os.path.basename(p)})")
X = np.column_stack(cols); Xt = np.column_stack(cols_t)
mask = y < 100
ptr = pd.Series(yr).value_counts(normalize=True); pte_s = pd.Series(yr_te).value_counts(normalize=True)
w = np.array([(pte_s/ptr).to_dict()[v] for v in yr]); td = pte_s.sort_index()
def ywmse(f):
    per = {Y: np.mean((f[yr==Y]-y[yr==Y])**2) for Y in np.unique(yr)}
    return sum(fr*per[Y] for Y,fr in td.items())
mo = np.zeros(len(y))
for tr, va in KFold(5, shuffle=True, random_state=321).split(X):
    tr2 = tr[mask[tr]]
    mo[va] = Ridge(alpha=1.0).fit(X[tr2], y[tr2], sample_weight=w[tr2]).predict(X[va])
mo = np.clip(mo, 0, 100)
def transform(params, mo_, p_, p95_, p90_, plo_):
    a1,a2,a3,a4 = params
    return np.clip(p_*100 + (1-p_)*mo_ + a1*p95_*(100-mo_)*(1-p_) + a2*p90_*(100-mo_)*(1-p_) - a3*plo_*mo_, 0, 100)
res = minimize(lambda p: ywmse(transform(p, mo, p_oof, p95_o, p90_o, plo_o)),
               x0=[0.05,0.02,0.02,0.0], method="Nelder-Mead", options={"maxiter":800})
foof = transform(res.x, mo, p_oof, p95_o, p90_o, plo_o)
mt = np.clip(Ridge(alpha=1.0).fit(X[mask], y[mask], sample_weight=w[mask]).predict(Xt), 0, 100)
fte = transform(res.x, mt, p_te, p95_t, p90_t, plo_t)
print("="*66, flush=True)
print(f"[REPEAT seed={FOLD_SEED}] uniform={np.mean((foof-y)**2):.3f} YIL-AĞIRLIKLI={ywmse(foof):.3f} params={np.round(res.x,3)}", flush=True)
np.savez(os.path.join(D, f"repeat_{FOLD_SEED}{'_rw' if BASE_CAUCHY else ''}{os.environ.get('OUT_TAG','')}.npz"), final_oof=foof, final_te=fte, y=y, params=res.x,
         cols=X, cols_t=Xt, p_oof=p_oof, p_te=p_te,
         p95_o=p95_o, p95_t=p95_t, p90_o=p90_o, p90_t=p90_t, plo_o=plo_o, plo_t=plo_t)
log(f"repeat_{FOLD_SEED}{'_rw' if BASE_CAUCHY else ''}.npz kaydedildi")
