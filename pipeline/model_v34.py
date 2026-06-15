# -*- coding: utf-8 -*-
"""7 metin-sinyalli stacking varyantı: etkileşim alt-modelleri + ileriye-dönük seçimli kolonlar,
5-seed Ridge meta, yıl-bazlı uç-değer kalibrasyonu. Çıktı: submission_v34.csv"""
import os
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
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
_afp = f"{D}/rp_out/agfeat_rp_results.npz"
af = np.load(_afp, allow_pickle=True)
af_o = np.clip(af["oof"],0,100); af_t = np.clip(af["te"],0,100)
v14 = np.load(f"{D}/v14_artifacts.npz", allow_pickle=True)["final_oof"]
v19 = np.load(f"{D}/v19_artifacts.npz", allow_pickle=True)["final_oof"]
v23 = np.load(f"{D}/v23_artifacts.npz", allow_pickle=True)["final_oof"]

mask = y < 100
ptr = pd.Series(yr).value_counts(normalize=True); pte_s = pd.Series(yr_te).value_counts(normalize=True)
w_yr = np.array([(pte_s/ptr).to_dict()[v] for v in yr]); td = pte_s.sort_index()
resid = y - v14; zstd = np.array([resid[yr==Y].std() for Y in yr]); z = np.abs(resid)/zstd
W = w_yr * (1.0/(1.0+(z/2.0)**2))
def ywmse(f):
    per = {Y: np.mean((f[yr==Y]-y[yr==Y])**2) for Y in np.unique(yr)}
    return sum(fr*per[Y] for Y,fr in td.items())

TF_CACHE = {}   # kuyruk parametreleri bacak başına bir kez fit edilir
def build_leg(X, Xt, p_o, p_t, p95o, p95t, p90o, p90t, ploo, plot, cv_seed, leg_key=None):
    def tf(params, mo_, p_, p95_, p90_, plo_):
        a1,a2,a3,a4 = params
        return np.clip(p_*100 + (1-p_)*mo_ + a1*p95_*(100-mo_)*(1-p_) + a2*p90_*(100-mo_)*(1-p_) - a3*plo_*mo_, 0, 100)
    mo = np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=cv_seed).split(X):
        tr2 = tr[mask[tr]]
        mo[va] = Ridge(alpha=1.0).fit(X[tr2], y[tr2], sample_weight=W[tr2]).predict(X[va])
    mo = np.clip(mo, 0, 100)
    if leg_key is not None and leg_key in TF_CACHE:
        px = TF_CACHE[leg_key]
    else:
        r = minimize(lambda p: ywmse(tf(p, mo, p_o, p95o, p90o, ploo)), x0=[0.05,0.02,0.02,0.0],
                     method="Nelder-Mead", options={"maxiter":800})
        px = r.x
        if leg_key is not None: TF_CACHE[leg_key] = px
    foof = tf(px, mo, p_o, p95o, p90o, ploo)
    mt = np.clip(Ridge(alpha=1.0).fit(X[mask], y[mask], sample_weight=W[mask]).predict(Xt), 0, 100)
    fte = tf(px, mt, p_t, p95t, p90t, plot)
    return foof, fte

NE8 = {42: (gpu["reg42_oof"], gpu["reg42_te"]), 43: (gpu["reg43_oof"], gpu["reg43_te"]),
       44: (r2k["reg44_oof"], r2k["reg44_te"])}
NEWFAM = {42: "rp_out/newfam2_42.npz", 43: "rp_out/newfam2_43.npz", 44: "rp_out/newfam2_44.npz"}

NS_O = [ag_o, ce_o, g7["gtequen7b_oof"], e7["e5mistral_oof"], ar["oof"], af_o]
NS_T = [ag_t, ce_t, g7["gtequen7b_te"], e7["e5mistral_te"], ar["te"], af_t]

def legs_for(cv_seed):
    legs_o, legs_t = [], []
    for FS in [42, 43, 44]:
        r = np.load(f"{D}/repeat_{FS}_pq2.npz", allow_pickle=True)
        nf = np.load(f"{D}/{NEWFAM[FS]}", allow_pickle=True)
        cols = r["cols"].copy(); cols_t = r["cols_t"].copy()
        cols[:, -2] = NE8[FS][0]; cols_t[:, -2] = NE8[FS][1]
        spo = np.clip(nf["q65_o"]-nf["q35_o"],0,None); s2o = spo*nf["q50_o"]/100
        spt = np.clip(nf["q65_t"]-nf["q35_t"],0,None); s2t = spt*nf["q50_t"]/100
        X = np.column_stack([cols, bg_o, nf["q35_o"], nf["q50_o"], nf["q65_o"], nf["rg_o"], spo, s2o] + NS_O)
        Xt = np.column_stack([cols_t, bg_t, nf["q35_t"], nf["q50_t"], nf["q65_t"], nf["rg_t"], spt, s2t] + NS_T)
        fo, ft = build_leg(X, Xt, r["p_oof"], r["p_te"], r["p95_o"], r["p95_t"], r["p90_o"], r["p90_t"], r["plo_o"], r["plo_t"], cv_seed, leg_key=FS)
        legs_o.append(fo); legs_t.append(ft)
    return legs_o, legs_t

CANDS = []
lr = np.load(f"{D}/rp_out/lora_full.npz")
sf1 = np.load(f"{D}/setfit_out/setfit_results.npz"); sf2 = np.load(f"{D}/rp_out/setfit_v2.npz")
sfa_o = np.clip((sf1["sf_berturk_oof"]+sf2["oof"])/2,0,100); sfa_t = np.clip((sf1["sf_berturk_te"]+sf2["te"])/2,0,100)
CANDS.append(("setfit_avg", sfa_o, sfa_t))
_lo = [lr["oof"]]; _lt = [lr["te"]]
for p in [f"{D}/rp_out/lora_s1042.npz", f"{D}/rp_out/lora_s2042.npz", f"{D}/rp_out/lora2_full.npz", f"{D}/rp_out/lora5_full.npz", f"{D}/rp_out/lora6_full.npz"]:
    if os.path.exists(p):
        z = np.load(p); _lo.append(z["oof"]); _lt.append(z["te"])
print(f"lora ortalaması: {len(_lo)} seed")
CANDS.append(("lora", np.clip(np.mean(_lo,0),0,100), np.clip(np.mean(_lt,0),0,100)))
tv = np.load(f"{D}/tgtvar.npz")
if os.path.exists(f"{D}/rp_out/ratio_cb3.npz"):
    rc3 = np.load(f"{D}/rp_out/ratio_cb3.npz")
    CANDS.append(("ratio_cb3", np.clip(rc3["oof"],0,100), np.clip(rc3["te"],0,100)))
else:
    CANDS.append(("ratio_cb", np.clip(tv["ratio_cb_oof"],0,100), np.clip(tv["ratio_cb_te"],0,100)))
fp = np.load(f"{D}/rp_out/fepolicy.npz")
CANDS.append(("notop_cb", np.clip(fp["notop_cb_o"],0,100), np.clip(fp["notop_cb_t"],0,100)))
CANDS.append(("notext_cb", np.clip(fp["notext_cb_o"],0,100), np.clip(fp["notext_cb_t"],0,100)))
for nm, fp_, keys in [("eurobert", f"{D}/rp_out/eurobert.npz", ("oof","te")),
                       ("tapt", f"{D}/rp_out/tapt_bert.npz", ("oof","te")),
                       ("lama", f"{D}/rp_out/lama.npz", ("oof","te"))]:
    if os.path.exists(fp_):
        z = np.load(fp_)
        CANDS.append((nm, np.clip(z[keys[0]],0,100), np.clip(z[keys[1]],0,100)))
if os.path.exists(f"{D}/rp_out/lorahyb_full.npz"):
    z = np.load(f"{D}/rp_out/lorahyb_full.npz")
    CANDS.append(("lorahyb", np.clip(z["oof"],0,100), np.clip(z["te"],0,100)))
al = np.load(f"{D}/aglong_out/ag_long_results.npz")
CANDS.append(("aglong", np.clip(al["oof"],0,100), np.clip(al["te"],0,100)))
print("adaylar:", [c[0] for c in CANDS], flush=True)

def yws_for():
    out = {}
    for cs in [321, 777]:
        lo, lt = legs_for(cs)
        out[cs] = ywmse(np.clip(np.mean(lo,0),0,100))
    return out

ref = yws_for()
print(f"v34 temel yapı: yw321={ref[321]:.3f} yw777={ref[777]:.3f}", flush=True)
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
print(f"kabul edilenler: {kept}", flush=True)

SEEDS = [321, 777, 555, 999, 123]
L = {cs: legs_for(cs) for cs in SEEDS}
def wblend(legs, w): w = np.abs(w); w = w/w.sum(); return np.clip(sum(wi*l for wi,l in zip(w,legs)),0,100)
FIT, VAL = [321, 555, 123], [777, 999]
r = minimize(lambda w: np.mean([ywmse(wblend(L[cs][0], w)) for cs in FIT]), x0=[1,1,1], method="Nelder-Mead")
w_opt = np.abs(r.x)/np.abs(r.x).sum()
dval = np.mean([ywmse(wblend(L[cs][0], w_opt)) - ywmse(np.clip(np.mean(L[cs][0],0),0,100)) for cs in VAL])
use_w = (0.5*w_opt + 0.5*np.array([1/3,1/3,1/3])) if dval < 0 else np.array([1/3,1/3,1/3])  # uniform'a büzme
print(f"bacak ağırlıkları {np.round(use_w,3)} (doğrulama Δ={dval:+.3f})", flush=True)
f = np.clip(np.mean([wblend(L[cs][0], use_w) for cs in SEEDS],0), 0, 100)
t = np.clip(np.mean([wblend(L[cs][1], use_w) for cs in SEEDS],0), 0, 100)
print(f"V34: yw={ywmse(f):.3f} uniform={np.mean((f-y)**2):.3f}")

rng = np.random.default_rng(51); yi = {Y: np.where(yr==Y)[0] for Y in td.index}
for ref_name, refv in [("v19", v19), ("v23", v23)]:
    dd = (refv-y)**2 - (f-y)**2
    counts = {Y: max(1, int(round(3000*fr))) for Y, fr in td.items()}
    counts[td.idxmax()] += 3000 - sum(counts.values())
    sims = np.array([dd[np.concatenate([rng.choice(yi[Y], size=counts[Y], replace=True) for Y in td.index])].mean() for _ in range(2000)])
    print(f"KAPI: P(v34 {ref_name}'ü yener) = {(sims>0).mean()*100:.1f}% (ort {sims.mean():+.3f})")
print(f"LB tahmini: {ywmse(f)+0.25:.2f} ± 0.2 (yıl-ağırlıklı tahmin + kalibrasyon payı)")
np.savez(f"{D}/v34_artifacts.npz", final_oof=f, final_te=t, y=y, w=use_w)
pd.DataFrame({"student_id": test["student_id"], "career_success_score": t}).to_csv(f"{D}/submission_v34.csv", index=False)
print("kaydedildi: v34_artifacts.npz + submission_v34.csv")
