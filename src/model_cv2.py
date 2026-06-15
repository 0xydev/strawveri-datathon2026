# -*- coding: utf-8 -*-
"""Çekirdek modül: feature engineering + sızıntısız metin-OOF.
add_features (skill-istatistik, etkileşimler, MNAR eksiklik, Türkçe NLP sinyalleri) ve
build_text_oof (TF-IDF + Ridge). Tüm pipeline bu modülü import eder."""
import os
import re, time
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack
from catboost import CatBoostRegressor, Pool

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
train = pd.read_csv(os.path.join(D, "train.csv"))
TGT = "career_success_score"; y = train[TGT].to_numpy(float)
CAT = ["department","university_tier","target_role","hobby","preferred_social_media_platform"]
if os.environ.get("YRROLE", "0") == "1":   # yıl x rol çaprazı etkileşimi
    CAT = CAT + ["_yr_role"]
if os.environ.get("INTERACT","0")=="1":
    CAT = CAT + ["_miss_pattern"]
TEXTCOL = "mentor_feedback_text"; DROP = [TGT,"student_id"]
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
    if os.environ.get("YRROLE", "0") == "1":
        df["_yr_role"] = df["application_year"].astype(str) + "|" + df["target_role"].astype(str)
    df["_skill_mean"]=df[SKILLS].mean(axis=1); df["_skill_std"]=df[SKILLS].std(axis=1)
    df["_skill_min"]=df[SKILLS].min(axis=1);  df["_skill_max"]=df[SKILLS].max(axis=1)
    df["_interview_mean"]=df[["technical_interview_score","hr_interview_score"]].mean(axis=1)
    df["_soft_mean"]=df[["communication_score","teamwork_score","leadership_score","presentation_score"]].mean(axis=1)
    df["_grad_minus_app"]=df["graduation_year"]-df["application_year"]
    df["_pq_ti"]=df["project_quality_score"]*df["technical_interview_score"]/100
    df["_x_ti_ps"]=df["technical_interview_score"]*df["problem_solving_score"]/100
    df["_x_code_ml"]=df["coding_score"]*df["machine_learning_score"]/100
    df["_x_ti_soft"]=df["technical_interview_score"]*df["_soft_mean"]/100
    if os.environ.get("INTERACT","0")=="1":   # ti*iletişim + eksiklik-pattern etkileşimleri
        df["_x_ti_comm"]=df["technical_interview_score"]*df["communication_score"]/100
        _mc=["internship_duration_months","english_exam_score","github_avg_stars","open_source_contribution_count","hr_interview_score","linkedin_profile_score","portfolio_score"]
        df["_miss_pattern"]=df[_mc].isna().astype(int).astype(str).agg("".join,axis=1)
        df["_x_pq_team"]=df["project_quality_score"]*df["teamwork_score"]/100
        df["_r_cgpa_port"]=df["cgpa"]/(df["portfolio_score"].abs()+1)
        if os.environ.get("INTERACT2","0")=="1":   # 2. tier etkileşimler
            df["_r_cgpa_devops"]=df["cgpa"]/(df["devops_score"].abs()+1)
            df["_x_cgpa_port"]=df["cgpa"]*df["portfolio_score"]/100
            df["_x_ti_front"]=df["technical_interview_score"]*df["frontend_score"]/100
            df["_x_ps_port"]=df["problem_solving_score"]*df["portfolio_score"]/100
    _yi = (df["application_year"]-2018).astype(float)
    df["_pq_yr"]=df["project_quality_score"]*_yi
    df["_ti_yr"]=df["technical_interview_score"]*_yi
    df["_cgpa_yr"]=df["cgpa"]*_yi
    df["_rc_yr"]=df["real_client_project_count"]*_yi
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
    df["_pos_cnt"]=sum(low.str.count(re.escape(w)) for w in POS)
    df["_neg_cnt"]=sum(low.str.count(re.escape(w)) for w in NEG)
    df["_polarity"]=df["_pos_cnt"]-df["_neg_cnt"]
    df["_text_mentions_role"]=[int(bool(re.search(ROLE_KW.get(r,"$^"),t))) for r,t in zip(df["target_role"],low)]
    return df

def build_text_oof(text, yv, seed=2024, k=5):
    """Sızıntısız OOF text_score: her satır, kendisi hariç eğitilmiş Ridge(TF-IDF) tahmini."""
    oof=np.zeros(len(text)); kf=KFold(k,shuffle=True,random_state=seed)
    for tr,va in kf.split(text):
        wv=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=30000,sublinear_tf=True)
        cv=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,max_features=30000,sublinear_tf=True)
        Xtr=hstack([wv.fit_transform(text[tr]),cv.fit_transform(text[tr])]).tocsr()
        Xva=hstack([wv.transform(text[va]),cv.transform(text[va])]).tocsr()
        r=Ridge(alpha=2.0).fit(Xtr,yv[tr])
        oof[va]=np.clip(r.predict(Xva),0,100)
    return oof

def run_cv(df_use, cat_idx, label, seed=42, k=5):
    oof=np.zeros(len(df_use)); kf=KFold(k,shuffle=True,random_state=seed); t0=time.time()
    print(f"[{label}] başlıyor ({df_use.shape[1]} feat)",flush=True)
    for fold,(tr,va) in enumerate(kf.split(df_use)):
        m=CatBoostRegressor(loss_function="RMSE",eval_metric="RMSE",depth=8,learning_rate=0.05,
            iterations=2000,l2_leaf_reg=3.0,random_seed=seed,od_type="Iter",od_wait=120,verbose=0)
        m.fit(Pool(df_use.iloc[tr],y[tr],cat_features=cat_idx),
              eval_set=Pool(df_use.iloc[va],y[va],cat_features=cat_idx),use_best_model=True)
        pred=np.clip(m.predict(df_use.iloc[va]),0,100); oof[va]=pred
        print(f"  fold {fold+1}/{k} MSE={np.mean((pred-y[va])**2):.3f} it={m.get_best_iteration()} ({time.time()-t0:.0f}s)",flush=True)
    mse=np.mean((oof-y)**2)
    print(f"[{label}] OOF MSE={mse:.3f} | RMSE={mse**0.5:.3f} ({time.time()-t0:.0f}s)",flush=True)
    return oof,mse

if __name__=="__main__":
    print("="*70)
    text=train[TEXTCOL].fillna("").to_numpy()
    t0=time.time()
    text_oof=build_text_oof(text,y)
    r=np.corrcoef(text_oof,y)[0,1]; mse_t=np.mean((text_oof-y)**2)
    print(f"[Metin-only Ridge] OOF MSE={mse_t:.3f} RMSE={mse_t**0.5:.3f} r={r:.3f} ({time.time()-t0:.0f}s)",flush=True)

    df=add_features(train); df["_text_score"]=text_oof
    feats=[c for c in df.columns if c not in DROP+[TEXTCOL]]
    du=df[feats].copy()
    cat_idx=[c for c in CAT if c in du.columns]
    for c in cat_idx: du[c]=du[c].astype(str).fillna("nan")
    print("-"*70)
    _,mseC=run_cv(du,cat_idx,"C: mühendislik + TF-IDF text_score")
    print("-"*70)
    print(f"OOF MSE (C): {mseC:.3f}")
