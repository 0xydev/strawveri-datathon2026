# -*- coding: utf-8 -*-
"""AutoGluon (extreme preset): ham metin, tabular ve etkileşim feature'ları.
Kaggle GPU T4, internet açık, ~4 saat. Çıktı: /kaggle/working/ag_extreme.npz."""
import os, time, subprocess, sys
T0 = time.time()
def log(s): print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "autogluon.tabular[all]"], check=True)
log("autogluon kuruldu")

import numpy as np, pandas as pd
DATA = None
for cand in ["/kaggle/input/datathon-2026", "/kaggle/input/competitions/datathon-2026"]:
    if os.path.exists(f"{cand}/train.csv"): DATA = cand; break
if DATA is None:
    for r, _, fs in os.walk("/kaggle/input"):
        if "train.csv" in fs: DATA = r; break
print("DATA =", DATA)
train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test_x.csv")
def add_inter(df):
    df=df.copy()
    df["_pq_ti"]=df["project_quality_score"]*df["technical_interview_score"]/100
    df["_x_ti_comm"]=df["technical_interview_score"]*df["communication_score"]/100
    df["_x_pq_team"]=df["project_quality_score"]*df["teamwork_score"]/100
    df["_r_cgpa_port"]=df["cgpa"]/(df["portfolio_score"].abs()+1)
    _mcc=["internship_duration_months","english_exam_score","github_avg_stars","open_source_contribution_count","hr_interview_score","linkedin_profile_score","portfolio_score"]
    df["_miss_pattern"]=df[_mcc].isna().astype(int).astype(str).agg("".join,axis=1)
    return df
train=add_inter(train); test=add_inter(test)
TGT = "career_success_score"

from autogluon.tabular import TabularPredictor
TIME_LIMIT = 14400  # 4 saat
pred = TabularPredictor(label=TGT, problem_type="regression", eval_metric="mean_squared_error",
                        path="/kaggle/working/agx")
pred.fit(train.drop(columns=["student_id"]),
         time_limit=TIME_LIMIT,
         presets=(os.environ.get("AGPRESET","extreme")),
         num_bag_folds=5, num_bag_sets=1,
         excluded_model_types=[],   # metin kolonu otomatik algılanır (raw text feature)
         verbosity=2)
log("fit bitti")
print(pred.leaderboard(silent=True).to_string())

# OOF (bagged modellerin out-of-fold tahmini) + test
oof = pred.predict_oof().to_numpy() if hasattr(pred, "predict_oof") else pred.get_oof_pred().to_numpy()
te = pred.predict(test.drop(columns=["student_id"])).to_numpy()
y = train[TGT].to_numpy(float)
print("AG OOF MSE:", round(float(np.mean((np.clip(oof,0,100)-y)**2)), 3))
np.savez("/kaggle/working/ag_extreme.npz", oof=oof, te=te)
log("BİTTİ -> Output'tan ag_extreme.npz indir")
