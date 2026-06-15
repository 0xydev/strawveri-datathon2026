# -*- coding: utf-8 -*-
"""TabPFN-v2 regresyon (n_est=8, fold 42 & 43) + kuyruk-eşik TabPFNClassifier'ları (p100/p95/p90/plo).
Kaggle GPU T4, internet açık. TABPFN_TOKEN + HF_TOKEN değişkenlerine kendi token'ını gir.
Çıktı: /kaggle/working/tabpfn_gpu_results.npz."""
import os
TABPFN_TOKEN = "BURAYA_TOKEN"   # <-- tabpfn_sk_... token'ını yapıştır
os.environ["TABPFN_TOKEN"] = TABPFN_TOKEN
os.environ["HF_TOKEN"] = "BURAYA_HF_TOKEN"  # <-- hf_... token (hızlı model indirme)

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn", "sentence-transformers"], check=True)

import re, time
import numpy as np, pandas as pd
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | CUDA: {torch.cuda.is_available()}" + ("" if torch.cuda.is_available() else "\n*** UYARI: GPU YOK! Settings -> Accelerator -> GPU T4 sec, oturum yeniden baslar ***"))
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

T0 = time.time()
def log(s): print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
DATA = None
for cand in ["/kaggle/input/datathon-2026", "/kaggle/input/competitions/datathon-2026"]:
    if os.path.exists(f"{cand}/train.csv"): DATA = cand; break
if DATA is None:
    for r, _, fs in os.walk("/kaggle/input"):
        if "train.csv" in fs: DATA = r; break
print("DATA =", DATA)
train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test_x.csv")
TGT = "career_success_score"; y = train[TGT].to_numpy(float)
# add_features: ana FE modülünün kopyası (Kaggle kernel'i harici modül import edemez)
CAT = ["department","university_tier","target_role","hobby","preferred_social_media_platform"]
TEXTCOL = "mentor_feedback_text"; DROP = [TGT, "student_id"]
SKILLS = ["coding_score","problem_solving_score","data_structures_score","sql_score",
          "machine_learning_score","backend_score","frontend_score","cloud_score","devops_score","project_quality_score"]
ROLE_SKILL = {"Cloud Engineer":"cloud_score","DevOps Engineer":"devops_score","MLOps Engineer":"devops_score",
    "Backend Developer":"backend_score","Frontend Developer":"frontend_score","Data Scientist":"machine_learning_score",
    "AI Engineer":"machine_learning_score","Data Analyst":"sql_score","Software Developer":"coding_score",
    "Product Analyst":"problem_solving_score","Cybersecurity Analyst":"cloud_score"}
POS = ["mükemmel","olağanüstü","başarı","kaliteli","etkileyici","üstün","harika","güçlü","yüksek","aranan","derin","liderlik","sektör","derece"]
NEG = ["eksiklikler","henüz","gerekiyor","sınırlı","zorluklar","zayıf","geliştirmesi","artırılması","ihtiyaç","umut","gelişmekte","becerilerde","aşamasında"]
ROLE_KW = {"Cloud Engineer":r"cloud|bulut","DevOps Engineer":r"devops","MLOps Engineer":r"mlops|devops",
    "Backend Developer":r"backend|arka uç","Frontend Developer":r"frontend|ön ?yüz","Data Scientist":r"veri bilim|makine öğren",
    "AI Engineer":r"yapay zeka|makine öğren","Data Analyst":r"veri analiz|sql","Software Developer":r"yazılım|kodlama",
    "Product Analyst":r"ürün|product","Cybersecurity Analyst":r"güvenlik|siber"}

def add_features(df):
    df = df.copy()
    df["_skill_mean"]=df[SKILLS].mean(axis=1); df["_skill_std"]=df[SKILLS].std(axis=1)
    df["_skill_min"]=df[SKILLS].min(axis=1);  df["_skill_max"]=df[SKILLS].max(axis=1)
    df["_interview_mean"]=df[["technical_interview_score","hr_interview_score"]].mean(axis=1)
    df["_soft_mean"]=df[["communication_score","teamwork_score","leadership_score","presentation_score"]].mean(axis=1)
    df["_grad_minus_app"]=df["graduation_year"]-df["application_year"]
    df["_github_activity"]=df[["github_repo_count","github_avg_stars","open_source_contribution_count"]].fillna(0).sum(axis=1)
    df["_exp_total"]=df[["real_client_project_count","internship_count","freelance_project_count","hackathon_count"]].sum(axis=1)
    df["_role_skill"]=np.nan
    for role,sc in ROLE_SKILL.items():
        m=df["target_role"]==role; df.loc[m,"_role_skill"]=df.loc[m,sc]
    df["_role_skill"]=df["_role_skill"].fillna(df["_skill_mean"])
    for c in ["internship_duration_months","english_exam_score","github_avg_stars",
              "open_source_contribution_count","hr_interview_score","linkedin_profile_score","portfolio_score"]:
        df[f"_miss_{c}"]=df[c].isna().astype(int)
    low=df[TEXTCOL].fillna("").str.lower()
    df["_txt_charlen"]=low.str.len(); df["_txt_words"]=low.str.split().str.len()
    df["_txt_sents"]=df[TEXTCOL].fillna("").str.count(r"\.")
    df["_has_contrast"]=low.str.contains(r"\b(?:ancak|ama|fakat)\b").astype(int)
    def sl(s):
        p=re.split(r"\b(ancak|ama|fakat)\b",s,maxsplit=1)
        return (len(p[0].split()),len(p[2].split())) if len(p)>=3 else (len(s.split()),0)
    pl,cl=zip(*low.map(sl)); df["_praise_len"]=pl; df["_crit_len"]=cl
    df["_praise_ratio"]=df["_praise_len"]/(df["_praise_len"]+df["_crit_len"]+1)
    df["_pos_cnt"]=sum(low.str.count(re.escape(wd)) for wd in POS)
    df["_neg_cnt"]=sum(low.str.count(re.escape(wd)) for wd in NEG)
    df["_polarity"]=df["_pos_cnt"]-df["_neg_cnt"]
    df["_text_mentions_role"]=[int(bool(re.search(ROLE_KW.get(r,"$^"),t))) for r,t in zip(df["target_role"],low)]
    return df

# ---- 1) metin skorları (TF-IDF + BERTurk)
log("TF-IDF text skoru")
text_tr = train[TEXTCOL].fillna("").to_numpy(); text_te = test[TEXTCOL].fillna("").to_numpy()
WKW = dict(ngram_range=(1,3), min_df=2, max_features=100000, sublinear_tf=True)
CKW = dict(analyzer="char_wb", ngram_range=(2,7), min_df=2, max_features=100000, sublinear_tf=True)
text_oof = np.zeros(len(text_tr))
for tr, va in KFold(5, shuffle=True, random_state=2024).split(text_tr):
    wv = TfidfVectorizer(**WKW); cv = TfidfVectorizer(**CKW)
    Xtr = hstack([wv.fit_transform(text_tr[tr]), cv.fit_transform(text_tr[tr])]).tocsr()
    Xva = hstack([wv.transform(text_tr[va]), cv.transform(text_tr[va])]).tocsr()
    text_oof[va] = np.clip(Ridge(alpha=4.0).fit(Xtr, y[tr]).predict(Xva), 0, 100)
wv = TfidfVectorizer(**WKW); cv = TfidfVectorizer(**CKW)
allt = np.concatenate([text_tr, text_te]); wv.fit(allt); cv.fit(allt)
text_te_score = np.clip(Ridge(alpha=4.0).fit(
    hstack([wv.transform(text_tr), cv.transform(text_tr)]).tocsr(), y).predict(
    hstack([wv.transform(text_te), cv.transform(text_te)]).tocsr()), 0, 100)

log("BERTurk embedding skoru (GPU)")
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", device=DEV)
E_tr = st.encode(list(text_tr), batch_size=256, normalize_embeddings=True, show_progress_bar=False)
E_te = st.encode(list(text_te), batch_size=256, normalize_embeddings=True, show_progress_bar=False)
emb_oof = np.zeros(len(y))
for tr, va in KFold(5, shuffle=True, random_state=2024).split(E_tr):
    emb_oof[va] = np.clip(Ridge(alpha=1.0).fit(E_tr[tr], y[tr]).predict(E_tr[va]), 0, 100)
emb_te = np.clip(Ridge(alpha=1.0).fit(E_tr, y).predict(E_te), 0, 100)
del st

# ---- 2) feature matrisi
dtr = add_features(train); dtr["_text_score"]=text_oof; dtr["_emb_score"]=emb_oof
dte = add_features(test);  dte["_text_score"]=text_te_score; dte["_emb_score"]=emb_te
feats = [c for c in dtr.columns if c not in DROP + [TEXTCOL]]
Xa = dtr[feats].copy(); Xe = dte[feats].copy()
for c in [c for c in CAT if c in feats]:
    cu = pd.api.types.union_categoricals([pd.Categorical(Xa[c].astype(str)), pd.Categorical(Xe[c].astype(str))]).categories
    Xa[c] = pd.Categorical(Xa[c].astype(str), categories=cu).codes.astype(np.float32)
    Xe[c] = pd.Categorical(Xe[c].astype(str), categories=cu).codes.astype(np.float32)
Xa = Xa.astype(np.float32).to_numpy(); Xe = Xe.astype(np.float32).to_numpy()

# ---- 3) TabPFN n_est=8: fold seed 42 ve 43 regresyon + seed 42 kuyruk classifier'ları
from tabpfn import TabPFNRegressor, TabPFNClassifier
out = {}
for FS in [42, 43]:
    folds = list(KFold(5, shuffle=True, random_state=FS).split(Xa))
    oof = np.zeros(len(y)); te = np.zeros(len(Xe))
    for fi, (tr, va) in enumerate(folds):
        m = TabPFNRegressor(device=DEV, n_estimators=8, random_state=42, ignore_pretraining_limits=True)
        m.fit(Xa[tr], y[tr])
        oof[va] = m.predict(Xa[va]); te += m.predict(Xe) / len(folds)
        log(f"reg fs={FS} fold {fi+1}/5 MSE={np.mean((np.clip(oof[va],0,100)-y[va])**2):.2f}")
        del m
    log(f"[TabPFN ne8 fs={FS}] OOF={np.mean((np.clip(oof,0,100)-y)**2):.3f}")
    out[f"reg{FS}_oof"] = oof; out[f"reg{FS}_te"] = te

folds = list(KFold(5, shuffle=True, random_state=42).split(Xa))
for nm, target in [("p100", (y==100)), ("p95", (y>=95)), ("p90", (y>=90)), ("plo", (y<=45))]:
    t = target.astype(int)
    po = np.zeros(len(y)); pt = np.zeros(len(Xe))
    for fi, (tr, va) in enumerate(folds):
        m = TabPFNClassifier(device=DEV, n_estimators=4, random_state=42, ignore_pretraining_limits=True)
        m.fit(Xa[tr], t[tr])
        po[va] = m.predict_proba(Xa[va])[:,1]; pt += m.predict_proba(Xe)[:,1] / len(folds)
        del m
    from sklearn.metrics import roc_auc_score
    log(f"[TabPFNClf {nm}] AUC={roc_auc_score(t, po):.4f}")
    out[f"{nm}_oof"] = po; out[f"{nm}_te"] = pt

np.savez("/kaggle/working/tabpfn_gpu_results.npz", **out)
log("BİTTİ -> Output sekmesinden tabpfn_gpu_results.npz dosyasını indir")
