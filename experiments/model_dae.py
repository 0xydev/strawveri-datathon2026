# -*- coding: utf-8 -*-
"""Denoising Autoencoder (512->128->512, swap-noise) ile denetimsiz bottleneck feature'ları
çıkarıp CatBoost base'e ekler (train+test üzerinde fit, sızıntı yok).
Sonuç: ağaç ailesine ek çeşitlilik katmadı, ensemble'a alınmadı."""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, Pool
import model_cv2 as M

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
train = M.train.copy(); y = M.y
test = pd.read_csv(os.path.join(D, "test_x.csv"))
T0 = time.time()
def log(s): print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
torch.manual_seed(42); np.random.seed(42)

a5 = np.load(os.path.join(D, "v5_artifacts.npz"), allow_pickle=True)
dtr = M.add_features(train); dtr["_text_score"]=a5["text_oof"]; dtr["_emb_score"]=a5["emb_oof"]
dte = M.add_features(test);  dte["_text_score"]=a5["text_te"];  dte["_emb_score"]=a5["emb_te"]
feats = [c for c in dtr.columns if c not in M.DROP + [M.TEXTCOL]]
Xa = dtr[feats].copy(); Xe = dte[feats].copy()
for c in [c for c in M.CAT if c in feats]:
    cu = pd.api.types.union_categoricals([pd.Categorical(Xa[c].astype(str)), pd.Categorical(Xe[c].astype(str))]).categories
    Xa[c] = pd.Categorical(Xa[c].astype(str), categories=cu).codes
    Xe[c] = pd.Categorical(Xe[c].astype(str), categories=cu).codes
med = Xa.median(); Xa = Xa.fillna(med); Xe = Xe.fillna(med)
sc = StandardScaler().fit(pd.concat([Xa, Xe]))
Z = np.vstack([sc.transform(Xa), sc.transform(Xe)]).astype(np.float32)

class DAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 128))
        self.dec = nn.Sequential(nn.ReLU(), nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, d))
    def forward(self, x): return self.dec(self.enc(x))

log("DAE eğitimi (swap-noise 0.15, 25 epoch, train+test)")
Zt = torch.tensor(Z)
m = DAE(Z.shape[1])
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
n = len(Z)
for ep in range(25):
    m.train(); perm = torch.randperm(n); tot = 0.0
    for i in range(0, n, 256):
        idx = perm[i:i+256]
        xb = Zt[idx].clone()
        # swap noise: her hücre %15 ihtimalle başka satırın aynı kolonuyla değişir
        mask_ = torch.rand_like(xb) < 0.15
        donor = Zt[torch.randint(0, n, (len(idx),))]
        xb[mask_] = donor[mask_]
        opt.zero_grad()
        loss = ((m(xb) - Zt[idx]) ** 2).mean()
        loss.backward(); opt.step()
        tot += float(loss) * len(idx)
    if (ep+1) % 5 == 0: log(f"  epoch {ep+1}/25 loss={tot/n:.4f}")
m.eval()
with torch.no_grad():
    E = m.enc(Zt).numpy()
E_tr, E_te = E[:len(Xa)], E[len(Xa):]
np.savez(os.path.join(D, "dae_emb.npz"), tr=E_tr, te=E_te)
log(f"bottleneck embeddingler hazır {E_tr.shape}")

# DAE feature'larını CatBoost base'e ekleyip OOF'u baseline ile karşılaştır
dub = dtr[feats].copy()
cat_cols = [c for c in M.CAT if c in dub.columns]
for c in cat_cols: dub[c] = dub[c].astype(str).fillna("nan")
for i in range(0, 128, 4):  # 32 bileşen yeter (hepsi 128 şişirir)
    dub[f"_dae_{i}"] = E_tr[:, i]
folds = list(KFold(5, shuffle=True, random_state=42).split(dub))
oof = np.zeros(len(y))
for tr, va in folds:
    mm = CatBoostRegressor(loss_function="RMSE", depth=5, learning_rate=0.02, iterations=7000,
                           l2_leaf_reg=3.0, random_seed=42, od_type="Iter", od_wait=150, verbose=0)
    mm.fit(Pool(dub.iloc[tr], y[tr], cat_features=cat_cols),
           eval_set=Pool(dub.iloc[va], y[va], cat_features=cat_cols), use_best_model=True)
    oof[va] = mm.predict(dub.iloc[va])
log(f"[CB + DAE32] OOF={np.mean((np.clip(oof,0,100)-y)**2):.3f} (referans cb42=75.690)")
np.save(os.path.join(D, "dae_cb_oof.npy"), oof)
