# -*- coding: utf-8 -*-
"""
strawveri - final submission'ın DETERMİNİSTİK yeniden-üretimi.

Seçtiğimiz resmi submission `bv34_v44s`, dört dekorele
pipeline çıktısının sabit-ağırlıklı harmanıdır. Harman tamamen deterministiktir; aşağıdaki
kod submit ettiğimiz CSV'yi BİT-BİT yeniden üretir (resmi dosyayla max|fark| ~ 4e-14).

Kullanım:
    python blend_final.py
        -> submission_bv34_v44s_repro.csv üretir ve resmi submission ile bit-bit doğrular.
"""
import os
import numpy as np
import pandas as pd

SUB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submissions")
SCORE = "career_success_score"


def _load(name):
    """Bir bileşen submission'ını student_id'ye göre sıralı oku (hizalama için)."""
    df = pd.read_csv(os.path.join(SUB, name))
    return df.sort_values("student_id").reset_index(drop=True)


def main():
    v49cons = _load("submission_v49cons.csv")
    v47     = _load("submission_v47.csv")
    v34     = _load("submission_v34.csv")
    v44s    = _load("submission_v44s.csv")

    # dekorele blend: 3-blend (v49cons/v47/v34) + v44s anchor
    blendv34  = np.clip(0.425 * v49cons[SCORE] + 0.425 * v47[SCORE] + 0.15 * v34[SCORE], 0, 100)
    bv34_v44s = 0.90 * blendv34 + 0.10 * v44s[SCORE]

    out = pd.DataFrame({"student_id": v49cons["student_id"], SCORE: bv34_v44s})
    out.to_csv("submission_bv34_v44s_repro.csv", index=False)
    print(f"submission_bv34_v44s_repro.csv yazildi ({len(out)} satir)")

    # resmi submission elimizdeyse bit-bit doğrula
    official = os.path.join(SUB, "submission_bv34_v44s.csv")
    if os.path.exists(official):
        ref = _load("submission_bv34_v44s.csv")[SCORE].to_numpy()
        d = np.abs(bv34_v44s.to_numpy() - ref)
        verdict = "bit-esdeger (~0)" if d.max() < 1e-9 else "FARK VAR"
        print(f"BIREBIR dogrulama: max|fark|={d.max():.2e}  ->  {verdict}")


if __name__ == "__main__":
    main()
