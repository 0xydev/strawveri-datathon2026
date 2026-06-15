"""MLM-TAPT (Task-Adaptive Pretraining): BERTurk'ü yarışma metinlerinde MLM ile devam-eğitir,
sonra regresyona fine-tune eder (RunPod GPU). Çıktı: tapt_bert.npz (oof, te)."""
import os, gc, math, random, time
os.environ["HF_HOME"] = "/root/hf"
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import KFold
from transformers import (AutoTokenizer, AutoModel, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)
import datasets as hfds
T0 = time.time()
def log(s): print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
D = "/root/sv/data"
train = pd.read_csv(f"{D}/train.csv"); test = pd.read_csv(f"{D}/test_x.csv")
y = train["career_success_score"].to_numpy(float)
ttr = train["mentor_feedback_text"].fillna("").tolist()
tte = test["mentor_feedback_text"].fillna("").tolist()
NAME = "dbmdz/bert-base-turkish-cased"
tok = AutoTokenizer.from_pretrained(NAME)
# --- 1) MLM TAPT: train+test 20k metin ---
all_txt = ttr + tte
ds = hfds.Dataset.from_dict({"text": all_txt}).map(
    lambda b: tok(b["text"], truncation=True, max_length=192), batched=True, remove_columns=["text"])
mlm = AutoModelForMaskedLM.from_pretrained(NAME).cuda()
args = TrainingArguments(output_dir="/root/tmp/tapt", num_train_epochs=30, per_device_train_batch_size=64,
    learning_rate=5e-5, warmup_ratio=0.06, fp16=True, logging_steps=500, save_strategy="no", report_to=[])
tr = Trainer(model=mlm, args=args, train_dataset=ds,
             data_collator=DataCollatorForLanguageModeling(tok, mlm_probability=0.15))
tr.train()
mlm.base_model.save_pretrained("/root/tmp/tapt_bb"); tok.save_pretrained("/root/tmp/tapt_bb")
log("TAPT bitti")
del mlm, tr; gc.collect(); torch.cuda.empty_cache()
# --- 2) regresyon fine-tune (5 fold) ---
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def enc(texts, idx, ml=192):
    b = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=ml, return_tensors="pt")
    return {k: v.to("cuda") for k, v in b.items()}
class ER(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = AutoModel.from_pretrained("/root/tmp/tapt_bb")
        self.head = nn.Linear(self.bb.config.hidden_size, 1)
    def forward(self, **b):
        am = b["attention_mask"]
        h = self.bb(**b).last_hidden_state
        e = (h*am.unsqueeze(-1)).sum(1)/am.sum(1,keepdim=True)
        return torch.sigmoid(self.head(e)).squeeze(-1)*100.0
folds = list(KFold(5, shuffle=True, random_state=2024).split(ttr))
oof = np.zeros(len(y)); te_acc = np.zeros(len(tte))
for fi, (trn, va) in enumerate(folds):
    set_seed(42+fi)
    m = ER().cuda(); m.train()
    opt = torch.optim.AdamW([{"params": m.bb.parameters(), "lr": 2e-5},
                             {"params": m.head.parameters(), "lr": 1e-4}], weight_decay=0.01)
    bs = 32; epochs = 3
    steps = math.ceil(len(trn)/bs)*epochs
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for ep in range(epochs):
        perm = np.random.permutation(trn)
        for s in range(0, len(perm), bs):
            idx = perm[s:s+bs]
            loss = nn.functional.mse_loss(m(**enc(ttr, idx)), torch.tensor(y[idx], device="cuda", dtype=torch.float32))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step()
    m.eval()
    def pred(texts, idxs):
        ps = []
        with torch.no_grad():
            for s2 in range(0, len(idxs), 64):
                ps.append(m(**enc(texts, idxs[s2:s2+64])).cpu().numpy())
        return np.clip(np.concatenate(ps), 0, 100)
    oof[va] = pred(ttr, va); te_acc += pred(tte, np.arange(len(tte)))/5
    log(f"tapt fold {fi} val-MSE={np.mean((oof[va]-y[va])**2):.2f}")
    np.savez("/root/sv/tapt_bert.npz", oof=oof, te=te_acc)
    del m, opt; gc.collect(); torch.cuda.empty_cache()
log(f"TAPT-BERT OOF={np.mean((oof-y)**2):.2f}")
