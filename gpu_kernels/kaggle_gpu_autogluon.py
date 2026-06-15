# -*- coding: utf-8 -*-
"""AutoGluon (best_quality): ham metin + tabular uçtan uca. Kaggle GPU T4, internet açık, ~2.5-3 saat.
AutoGluon kendi 5-fold bagging'iyle OOF üretir. Çıktı: /kaggle/working/ag_results.npz."""
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
TGT = "career_success_score"

from autogluon.tabular import TabularPredictor
TIME_LIMIT = 9000  # 2.5 saat
pred = TabularPredictor(label=TGT, problem_type="regression", eval_metric="mean_squared_error",
                        path="/kaggle/working/ag")
pred.fit(train.drop(columns=["student_id"]),
         time_limit=TIME_LIMIT,
         presets="best_quality",
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
np.savez("/kaggle/working/ag_results.npz", oof=oof, te=te)
log("BİTTİ -> Output'tan ag_results.npz indir")
