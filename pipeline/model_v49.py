# -*- coding: utf-8 -*-
"""v47 pipeline + gereksiz alt-kolonların budanması (DROPCOLS ile leave-one-out denetimi). Çıktı: submission_v49.csv"""
import os
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize

D = "data"
train = pd.read_csv(f"{D}/train.csv"); test = pd.read_csv(f"{D}/test_x.csv")
y = train["career_success_score"].to_numpy(float)
yr = train["application_year"].to_numpy(); yr_te = test["application_year"].to_numpy()
bg_o = np.load(f"{D}/embscore_bgem3_oof.npy"); bg_t = np.load(f"{D}/embscore_bgem3_te.npy")
ce_o = np.load(f"{D}/embscore_cosmos_e5_oof.npy"); ce_t = np.load(f"{D}/embscore_cosmos_e5_te.npy")
gpu = np.load(f"{D}/tabpfn_gpu_results.npz", allow_pickle=True)
r2k = np.load(f"{D}/round2_results.npz", allow_pickle=True)
ag = np.load(f"{D}/ag_results.npz", allow_pickle=True); ag_o = np.clip(ag["oof"],0,100); ag_t = np.clip(ag["te"],0,100)
ar = np.load(f"{D}/embrace_arctic2.npz", allow_pickle=True)
e7 = np.load(f"{D}/emb7b_results.npz", allow_pickle=True)
g7 = np.load(f"{D}/emb7b_gtequen.npz", allow_pickle=True)
af = np.load(f"{D}/rp_out/agfeat_rp_results.npz", allow_pickle=True)
af_o = np.clip(af["oof"],0,100); af_t = np.clip(af["te"],0,100)
v14 = np.load(f"{D}/v14_artifacts.npz", allow_pickle=True)["final_oof"]
v19 = np.load(f"{D}/v19_artifacts.npz", allow_pickle=True)["final_oof"]
v39 = np.load(f"{D}/v39_artifacts.npz", allow_pickle=True)["final_oof"]

mask = y < 100
ptr = pd.Series(yr).value_counts(normalize=True); pte_s = pd.Series(yr_te).value_counts(normalize=True)
w_yr = np.array([(pte_s/ptr).to_dict()[v] for v in yr]); td = pte_s.sort_index()
resid = y - v14; zstd = np.array([resid[yr==Y].std() for Y in yr]); z = np.abs(resid)/zstd
W = w_yr * (1.0/(1.0+(z/3.0)**2))
def ywmse(f):
    per = {Y: np.mean((f[yr==Y]-y[yr==Y])**2) for Y in np.unique(yr)}
    return sum(fr*per[Y] for Y,fr in td.items())

LEGS = [42, 43, 44]
_uyr = np.unique(yr)
if os.environ.get("PERYIL", "1") == "1":   # per-yıl kuyruk kalibrasyonu
    YGRP = np.searchsorted(_uyr, yr); YGRP_TE = np.searchsorted(_uyr, yr_te); NGRP = len(_uyr)
else:
    YGRP = np.where(yr<=2022, 0, np.where(yr<=2024, 1, 2)); YGRP_TE = np.where(yr_te<=2022, 0, np.where(yr_te<=2024, 1, 2)); NGRP = 3

REPEAT = {FS: f"repeat_{FS}_txy{os.environ.get('LEGSRC','')}.npz" for FS in LEGS}
NE8 = {42: (gpu["reg42_oof"], gpu["reg42_te"]), 43: (gpu["reg43_oof"], gpu["reg43_te"]),
       44: (r2k["reg44_oof"], r2k["reg44_te"])}
NEWFAM = {FS: f"rp_out/newfam2_{FS}_txy.npz" for FS in LEGS}
_t1 = np.load(f"{D}/rp_out/thresh_txy.npz")
P55 = {FS: (_t1[f"p55lo_o_{FS}"], _t1[f"p55lo_t_{FS}"]) for FS in LEGS}

ISO_P = {}
for FS in LEGS:
    _r = np.load(f"{D}/{REPEAT[FS]}", allow_pickle=True)
    p_o, p_t = _r["p_oof"], _r["p_te"]
    tgt = (y == 100).astype(float)
    p_cal = np.zeros(len(p_o))
    for tr, va in KFold(5, shuffle=True, random_state=7).split(p_o):
        ir = IsotonicRegression(out_of_bounds="clip").fit(p_o[tr], tgt[tr])
        p_cal[va] = ir.predict(p_o[va])
    ir_full = IsotonicRegression(out_of_bounds="clip").fit(p_o, tgt)
    ISO_P[FS] = (np.clip(p_cal, 0, 1), np.clip(ir_full.predict(p_t), 0, 1))

def tf(a, mo_, p_, p95_, p90_, plo_):
    return np.clip(p_*100 + (1-p_)*mo_ + a[0]*p95_*(100-mo_)*(1-p_) + a[1]*p90_*(100-mo_)*(1-p_) - a[2]*plo_*mo_, 0, 100)

TF_CACHE = {}   # (FS, grup) -> kuyruk-a; bir kez fit edilir
def build_leg(X, Xt, p_o, p_t, p95o, p95t, p90o, p90t, ploo, plot, cv_seed, leg_key):
    mo = np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=cv_seed).split(X):
        tr2 = tr[mask[tr]]
        mo[va] = Ridge(alpha=1.0).fit(X[tr2], y[tr2], sample_weight=W[tr2]).predict(X[va])
    mo = np.clip(mo, 0, 100)
    foof = np.zeros(len(y))
    mt = np.clip(Ridge(alpha=1.0).fit(X[mask], y[mask], sample_weight=W[mask]).predict(Xt), 0, 100)
    fte = np.zeros(len(Xt))
    for gi in range(NGRP):   # yıl-grubu kuyruk fiti
        m2 = YGRP == gi; m2t = YGRP_TE == gi
        if m2.sum() == 0: continue
        key = (leg_key, gi)
        if key in TF_CACHE:
            a = TF_CACHE[key]
        else:
            r = minimize(lambda p: np.mean((tf(p, mo[m2], p_o[m2], p95o[m2], p90o[m2], ploo[m2])-y[m2])**2),
                         x0=[0.05,0.02,0.02], method="Nelder-Mead", options={"maxiter":800})
            a = r.x; TF_CACHE[key] = a
        foof[m2] = tf(a, mo[m2], p_o[m2], p95o[m2], p90o[m2], ploo[m2])
        fte[m2t] = tf(a, mt[m2t], p_t[m2t], p95t[m2t], p90t[m2t], plot[m2t])
    return foof, fte

if os.environ.get("PRUNE", "1") == "1":   # greedy-backward: gereksiz base-kolonları düşür
    NS_O = [g7["gtequen7b_oof"], e7["e5mistral_oof"], ar["oof"], af_o]
    NS_T = [g7["gtequen7b_te"], e7["e5mistral_te"], ar["te"], af_t]
else:
    NS_O = [ag_o, ce_o, g7["gtequen7b_oof"], e7["e5mistral_oof"], ar["oof"], af_o]
    NS_T = [ag_t, ce_t, g7["gtequen7b_te"], e7["e5mistral_te"], ar["te"], af_t]
_PRUNE = os.environ.get("PRUNE", "1") == "1"

LEG_CANDS = []
_rm = np.load(f"{D}/rp_out/realmlp.npz")
LEG_CANDS.append(("realmlp", {FS: (np.clip(_rm[f"oof_{FS}"],0,100), np.clip(_rm[f"te_{FS}"],0,100)) for FS in LEGS}))
if os.path.exists(f"{D}/rp_out/tabpfn_interact4.npz"):
    _tpi = np.load(f"{D}/rp_out/tabpfn_interact4.npz")
    LEG_CANDS.append(("tabpfn_int4", {FS: (np.clip(_tpi[f"oof_{FS}"],0,100), np.clip(_tpi[f"te_{FS}"],0,100)) for FS in LEGS}))
if os.path.exists(f"{D}/rp_out/realmlp_interact.npz"):
    _rmi = np.load(f"{D}/rp_out/realmlp_interact.npz")
    LEG_CANDS.append(("realmlp_int", {FS: (np.clip(_rmi[f"oof_{FS}"],0,100), np.clip(_rmi[f"te_{FS}"],0,100)) for FS in LEGS}))
LEG_ACTIVE = []

def legs_for(cv_seed):
    legs_o, legs_t = [], []
    for FS in LEGS:
        r = np.load(f"{D}/{REPEAT[FS]}", allow_pickle=True)
        nf = np.load(f"{D}/{NEWFAM[FS]}", allow_pickle=True)
        cols = r["cols"].copy(); cols_t = r["cols_t"].copy()
        cols[:, -2] = NE8[FS][0]; cols_t[:, -2] = NE8[FS][1]   # kolon[-2] = TabPFN slotu
        _drop = [int(x) for x in os.environ.get("DROPCOLS","").split(",") if x!=""]
        if _drop:   # verilen base-kolonlarını leg'den çıkar
            _keep = [i for i in range(cols.shape[1]) if i not in _drop]
            cols = cols[:, _keep]; cols_t = cols_t[:, _keep]
        spo = np.clip(nf["q65_o"]-nf["q35_o"],0,None); s2o = spo*nf["q50_o"]/100
        spt = np.clip(nf["q65_t"]-nf["q35_t"],0,None); s2t = spt*nf["q50_t"]/100
        bg2 = np.load(f"{D}/rp_out/cfgbag_{FS}_txy.npz")
        ex_o = [bg2["lgb_o"], bg2["xgb_o"]]; ex_t = [bg2["lgb_t"], bg2["xgb_t"]]
        lc_o = [lc[1][FS][0] for lc in LEG_ACTIVE]; lc_t = [lc[1][FS][1] for lc in LEG_ACTIVE]
        X = np.column_stack([cols, bg_o, nf["q35_o"], nf["q50_o"], nf["q65_o"], nf["rg_o"], spo, s2o] + NS_O + ex_o + lc_o)
        Xt = np.column_stack([cols_t, bg_t, nf["q35_t"], nf["q50_t"], nf["q65_t"], nf["rg_t"], spt, s2t] + NS_T + ex_t + lc_t)
        fo, ft = build_leg(X, Xt, ISO_P[FS][0], ISO_P[FS][1], r["p95_o"], r["p95_t"], r["p90_o"], r["p90_t"], P55[FS][0], P55[FS][1], cv_seed, FS)
        legs_o.append(fo); legs_t.append(ft)
    return legs_o, legs_t

CANDS = []
sf1 = np.load(f"{D}/setfit_out/setfit_results.npz"); sf2 = np.load(f"{D}/rp_out/setfit_v2.npz")
if not _PRUNE: CANDS.append(("setfit_avg", np.clip((sf1["sf_berturk_oof"]+sf2["oof"])/2,0,100), np.clip((sf1["sf_berturk_te"]+sf2["te"])/2,0,100)))
lr = np.load(f"{D}/rp_out/lora_full.npz"); _lo = [lr["oof"]]; _lt = [lr["te"]]
for p in [f"{D}/rp_out/lora_s1042.npz", f"{D}/rp_out/lora_s2042.npz", f"{D}/rp_out/lora2_full.npz", f"{D}/rp_out/lora5_full.npz", f"{D}/rp_out/lora6_full.npz"]:
    if os.path.exists(p):
        zz = np.load(p); _lo.append(zz["oof"]); _lt.append(zz["te"])
if not _PRUNE: CANDS.append(("lora", np.clip(np.mean(_lo,0),0,100), np.clip(np.mean(_lt,0),0,100)))
rc3 = np.load(f"{D}/rp_out/ratio_cb3_txy.npz")
CANDS.append(("ratio_cb3", np.clip(rc3["oof"],0,100), np.clip(rc3["te"],0,100)))
n3 = np.load(f"{D}/rp_out/notop_cb3_txy.npz"); ns2 = np.load(f"{D}/rp_out/notop_cb_s2.npz")
no5 = (3*n3["oof"]+2*ns2["oof"])/5; nt5 = (3*n3["te"]+2*ns2["te"])/5
CANDS.append(("notop_cb3", np.clip(no5,0,100), np.clip(nt5,0,100)))
_eo, _et = [], []
for p in [f"{D}/rp_out/eurobert.npz", f"{D}/rp_out/eurobert2.npz", f"{D}/rp_out/eurobert3.npz"]:
    if os.path.exists(p):
        zz = np.load(p); _eo.append(zz["oof"]); _et.append(zz["te"])
print(f"eurobert: {len(_eo)} seed")
CANDS.append(("eurobert", np.clip(np.mean(_eo,0),0,100), np.clip(np.mean(_et,0),0,100)))
_ho, _ht = [], []
for s in ["", "3", "4", "5", "7"]:
    p = f"{D}/rp_out/lorahyb{s}_full.npz"
    if os.path.exists(p):
        zz = np.load(p); _ho.append(zz["oof"]); _ht.append(zz["te"])
print(f"lorahyb: {len(_ho)} seed")
CANDS.append(("lorahyb", np.clip(np.mean(_ho,0),0,100), np.clip(np.mean(_ht,0),0,100)))
_fo, _ft = [], []
for p in [f"{D}/rp_out/lorahybF_full.npz", f"{D}/rp_out/lorahybG_full.npz"]:
    if os.path.exists(p):
        zz = np.load(p); _fo.append(zz["oof"]); _ft.append(zz["te"])
print(f"lorahybF (tam-satır): {len(_fo)} seed")
CANDS.append(("lorahybF", np.clip(np.mean(_fo,0),0,100), np.clip(np.mean(_ft,0),0,100)))
_qo, _qt = [], []
for p in [f"{D}/rp_out/lorahybQ1_full.npz", f"{D}/rp_out/lorahybQ2_full.npz", f"{D}/rp_out/lorahybQ3_full.npz"]:
    if os.path.exists(p):
        zz = np.load(p); _qo.append(zz["oof"]); _qt.append(zz["te"])
if _qo:
    print(f"lorahybQ (Qwen3): {len(_qo)} seed")
    CANDS.append(("lorahybQ", np.clip(np.mean(_qo,0),0,100), np.clip(np.mean(_qt,0),0,100)))
rcb = np.load(f"{D}/rp_out/resid_cb.npz")
CANDS.append(("resid_cb", np.clip(rcb["oof"],0,100), np.clip(rcb["te"],0,100)))
_n = [np.load(f"{D}/rp_out/lorahybN_full.npz"), np.load(f"{D}/rp_out/lorahybN2_full.npz")]
CANDS.append(("nemotron", np.clip(np.mean([z["oof"] for z in _n],0),0,100), np.clip(np.mean([z["te"] for z in _n],0),0,100)))
_q3 = [np.load(f"{D}/rp_out/lorahybQa_full.npz"), np.load(f"{D}/rp_out/lorahybQb_full.npz")]
CANDS.append(("qwen3lo", np.clip(np.mean([z["oof"] for z in _q3],0),0,100), np.clip(np.mean([z["te"] for z in _q3],0),0,100)))
if os.path.exists(f"{D}/rp_out/svr_emb.npz"):
    sv_ = np.load(f"{D}/rp_out/svr_emb.npz")
    if "avg_oof" in sv_:
        CANDS.append(("svr_emb", sv_["avg_oof"], sv_["avg_te"]))
al = np.load(f"{D}/aglong_out/ag_long_results.npz")
CANDS.append(("aglong", np.clip(al["oof"],0,100), np.clip(al["te"],0,100)))
tp = np.load(f"{D}/rp_out/tapt_bert.npz"); t2 = np.load(f"{D}/rp_out/tapt_bert2.npz")
CANDS.append(("tapt", np.clip((tp["oof"]+t2["oof"])/2,0,100), np.clip((tp["te"]+t2["te"])/2,0,100)))
print("adaylar:", [c[0] for c in CANDS], flush=True)

def yws_for():
    out = {}
    for cs in [321, 777]:
        lo, lt = legs_for(cs)
        out[cs] = ywmse(np.clip(np.mean(lo,0),0,100))
    return out

ref = yws_for()
print(f"iskelet: yw321={ref[321]:.3f} yw777={ref[777]:.3f}", flush=True)
kept = []
for name, co, ct in CANDS:
    NS_O.append(co); NS_T.append(ct)
    ev = yws_for()
    d321, d777 = ev[321]-ref[321], ev[777]-ref[777]
    ok = d321 < 0 and d777 < 0
    print(f"{name}: Δ321={d321:+.3f} Δ777={d777:+.3f} -> {'KABUL' if ok else 'RED'}", flush=True)
    if ok:
        kept.append(name); ref = ev
    else:
        NS_O.pop(); NS_T.pop()
for lc in LEG_CANDS:
    LEG_ACTIVE.append(lc)
    ev = yws_for()
    d321, d777 = ev[321]-ref[321], ev[777]-ref[777]
    ok = d321 < 0 and d777 < 0
    print(f"{lc[0]} (bacak-hizalı): Δ321={d321:+.3f} Δ777={d777:+.3f} -> {'KABUL' if ok else 'RED'}", flush=True)
    if ok:
        kept.append(lc[0]); ref = ev
    else:
        LEG_ACTIVE.pop()
print(f"kabul edilenler: {kept}", flush=True)

TF_CACHE.clear()   # kuyruk-a'yı final kabul-edilmiş kolon setinde yeniden fit et

SEEDS = [321, 777, 555, 999, 123, 2024, 7, 42, 1001]  # 9 meta-seed
L = {cs: legs_for(cs) for cs in SEEDS}
def wblend(legs, w): w = np.abs(w); w = w/w.sum(); return np.clip(sum(wi*l for wi,l in zip(w,legs)),0,100)
FIT, VAL = [321, 555, 123], [777, 999]
r = minimize(lambda w: np.mean([ywmse(wblend(L[cs][0], w)) for cs in FIT]), x0=[1]*len(LEGS), method="Nelder-Mead")
w_opt = np.abs(r.x)/np.abs(r.x).sum()
dval = np.mean([ywmse(wblend(L[cs][0], w_opt)) - ywmse(np.clip(np.mean(L[cs][0],0),0,100)) for cs in VAL])
use_w = w_opt if dval < 0 else np.full(len(LEGS), 1/len(LEGS))
print(f"bacak ağırlıkları {np.round(use_w,3)} (doğrulama Δ={dval:+.3f})", flush=True)
f = np.clip(np.mean([wblend(L[cs][0], use_w) for cs in SEEDS],0), 0, 100)
t = np.clip(np.mean([wblend(L[cs][1], use_w) for cs in SEEDS],0), 0, 100)
# metin kuralı: bu ifade neredeyse her zaman tam başarı (99.5)
phr = "mükemmel bir başarı"
m_tr = train["mentor_feedback_text"].fillna("").str.lower().str.contains(phr, regex=False).to_numpy()
m_te = test["mentor_feedback_text"].fillna("").str.lower().str.contains(phr, regex=False).to_numpy()
print(f"metin kuralı: train {m_tr.sum()} satır (yw önce={ywmse(f):.3f})", flush=True)
f[m_tr] = 99.5; t[m_te] = 99.5
phr2 = "mükemmel bir kariyer"   # ikinci sabit-ifade kuralı
m2_tr = train["mentor_feedback_text"].fillna("").str.lower().str.contains(phr2, regex=False).to_numpy()
m2_te = test["mentor_feedback_text"].fillna("").str.lower().str.contains(phr2, regex=False).to_numpy()
f[m2_tr] = 99.5; t[m2_te] = 99.5
print(f"2. metin kuralı: train {m2_tr.sum()} / test {m2_te.sum()} satir", flush=True)
print(f"V49: yw={ywmse(f):.3f} uniform={np.mean((f-y)**2):.3f}")

rng = np.random.default_rng(51); yi = {Y: np.where(yr==Y)[0] for Y in td.index}
for ref_name, refv in [("v19", v19), ("v39", v39)]:
    dd = (refv-y)**2 - (f-y)**2
    counts = {Y: max(1, int(round(3000*fr))) for Y, fr in td.items()}
    counts[td.idxmax()] += 3000 - sum(counts.values())
    sims = np.array([dd[np.concatenate([rng.choice(yi[Y], size=counts[Y], replace=True) for Y in td.index])].mean() for _ in range(2000)])
    print(f"KAPI: P(v49 {ref_name}'ü yener) = {(sims>0).mean()*100:.1f}% (ort {sims.mean():+.3f})")
print(f"LB tahmini: {ywmse(f)+0.25:.2f} ± 0.2 (yıl-ağırlıklı tahmin + kalibrasyon payı)")
np.savez(f"{D}/v49_artifacts.npz", final_oof=f, final_te=t, y=y, w=use_w)
pd.DataFrame({"student_id": test["student_id"], "career_success_score": t}).to_csv(f"{D}/submission_v49.csv", index=False)
print("kaydedildi: v49_artifacts.npz + submission_v49.csv")
