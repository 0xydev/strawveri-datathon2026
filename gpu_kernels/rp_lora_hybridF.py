"""gte-Qwen2-7B + LoRA ile career_success_score regresyonu (RunPod GPU, bf16).
FOLDS env ile fold bölünür (örn FOLDS=0,1,2)."""
import os, gc, math, random, time
os.environ["HF_HOME"] = "/root/hf"
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
T0 = time.time()
def log(s): print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
FOLDS = [int(x) for x in os.environ.get("FOLDS", "0,1,2,3,4").split(",")]
TAG = os.environ.get("TAG", "A")
D = "/root/sv/data"
train = pd.read_csv(f"{D}/train.csv"); test = pd.read_csv(f"{D}/test_x.csv")
y = train["career_success_score"].to_numpy(float)
def _f(v, d=0):
    try:
        return f"{float(v):.{d}f}" if v == v else "?"
    except Exception: return "?"
def hybrid(df):
    rows = []
    for r, t in zip(df.itertuples(), df["mentor_feedback_text"].fillna("")):
        rows.append(
            f"Proje:{_f(r.project_quality_score)} Mulakat:{_f(r.technical_interview_score)} Not:{_f(r.cgpa,1)} "
            f"Yil:{r.application_year} Problem:{_f(r.problem_solving_score)} Iletisim:{_f(r.communication_score)} "
            f"Kod:{_f(r.coding_score)} ML:{_f(r.machine_learning_score)} VeriYapi:{_f(r.data_structures_score)} "
            f"IK:{_f(r.hr_interview_score)} Staj:{_f(r.internship_count)} Musteri:{_f(r.real_client_project_count)} "
            f"Rol:{r.target_role} Uni:{r.university_tier} | {t}")
    return rows
ttr = hybrid(train); tte = hybrid(test)
NAME = "Alibaba-NLP/gte-Qwen2-7B-instruct"
PREF = "Instruct: Mentör değerlendirmesinin olumluluk düzeyini belirle\nQuery: "
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
tok = AutoTokenizer.from_pretrained(NAME)
def enc(texts, idx, ml=224):
    b = tok([PREF + texts[i] for i in idx], padding=True, truncation=True, max_length=ml, return_tensors="pt")
    return b["input_ids"].to("cuda"), b["attention_mask"].to("cuda")
class LoraReg(nn.Module):
    def __init__(self):
        super().__init__()
        bb = AutoModel.from_pretrained(NAME, dtype=torch.bfloat16, attn_implementation="sdpa")
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules=["q_proj","k_proj","v_proj","o_proj"])
        self.bb = get_peft_model(bb, cfg)
        self.head = nn.Linear(self.bb.config.hidden_size, 1, dtype=torch.bfloat16)
    def forward(self, ids, am):
        h = self.bb(input_ids=ids, attention_mask=am).last_hidden_state
        last = am.sum(1) - 1
        e = h[torch.arange(h.size(0)), last]
        return self.head(e).squeeze(-1).float()
folds = list(KFold(5, shuffle=True, random_state=2024).split(ttr))
out = {}
for fi in FOLDS:
    tr, va = folds[fi]
    set_seed(8242 + fi)
    m = LoraReg().cuda(); m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    bs = int(os.environ.get("BS", "3"))
    steps = math.ceil(len(tr)/bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=steps, pct_start=0.1)
    perm = np.random.permutation(tr)
    for s in range(0, len(perm), bs):
        idx = perm[s:s+bs]
        ids, am = enc(ttr, idx)
        loss = nn.functional.mse_loss(m(ids, am), torch.tensor(y[idx], device="cuda", dtype=torch.float32))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if (s//bs) % 400 == 0: log(f"fold{fi} step {s//bs}/{steps} loss={loss.item():.1f}")
    m.eval()
    def pred(texts, idxs):
        ps = []
        with torch.no_grad():
            for s2 in range(0, len(idxs), 16):
                ids, am = enc(texts, idxs[s2:s2+16])
                ps.append(m(ids, am).cpu().numpy())
        return np.clip(np.concatenate(ps), 0, 100)
    ov = pred(ttr, va)
    out[f"oof_{fi}"] = ov; out[f"va_{fi}"] = va.astype(np.int64)
    out[f"te_{fi}"] = pred(tte, np.arange(len(tte)))
    log(f"fold {fi} val-MSE={np.mean((ov-y[va])**2):.2f}")
    np.savez(f"/root/sv/lorahybF_part{TAG}.npz", **out)
    del m, opt; gc.collect(); torch.cuda.empty_cache()
log(f"BİTTİ part{TAG}")
