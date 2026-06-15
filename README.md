# strawveri (BTK Datathon 2026)

Bu repo, BTK Datathon 2026 yarışmasında kurduğumuz çözümün kodunu içeriyor. Görev şuydu: öğrencilerin kariyer başarı puanını sıfırla yüz arasında tahmin etmek. Bir sayı tahmin ettiğimiz için regresyon, başarıyı da MSE ile ölçtüler. Veri karışıktı; bir yanda tablo halinde özellikler, diğer yanda kuralların zorunlu tuttuğu bir Türkçe metin alanı vardı, yani mentörün öğrenci hakkında yazdığı değerlendirme (mentor_feedback_text).

Çözümün uçtan uca anlatımını, grafikleri ve çıktıları gömülü olarak `notebook/strawveri_datathon2026.ipynb` içinde bulabilirsiniz. Repoda ayrıca submission'ları üretirken gerçekten kullandığımız pipeline kodu ve ara model tahminleri var.

## Repoda neler var

| Klasör | İçerik |
|---|---|
| src/ | model_cv2.py: özellik üretimi ve sızıntısız metin skoru çekirdeği. Bütün pipeline bunu çağırıyor. |
| pipeline/ | Submission pipeline'ları: model_v34, v44, v44s, v47, v49 ve tekrarlı koşum için model_repeat. |
| gpu_kernels/ | Ağır parçaların kodu: hibrit LoRA (gte-Qwen2-7B), TAPT, SetFit, TabPFN, AutoGluon ve embedding. |
| experiments/ | Deneyip bıraktığımız fikirler: pseudo-label, hedef dönüşümü, target encoding ile GMM, DAE. |
| notebook/ | Jüri için hazırladığımız, okunabilir uçtan uca notebook. |
| submissions/ | Dört ara tahmin ve final submission (bv34_v44s). |
| blend_final.py | Final submission'ı bu dört bileşenden birebir yeniden üreten script. |

## İzlediğimiz yöntem

Önce özellik üretimiyle başladık: değişkenler arası etkileşimler, yıla göre değişen etkileşimler, eksik alanları işaretleyen bayraklar ve metinden çıkardığımız sayısal sinyaller.

Zorunlu Türkçe metni baştan savmadık, tam tersine dört katmanda işledik. En basitten en güçlüye doğru: TF-IDF, transformer embedding, SetFit ile TAPT, ve en tepede tabloyu metinle birleştirip eğittiğimiz hibrit LoRA modeli.

Tahmin tarafında sığ ağaç ailelerini (CatBoost, LightGBM, XGBoost) klasik modeller ve modern tablo modelleriyle (TabPFN, TabM) birlikte kullandık. Hepsini, yıla göre ağırlıklandırılmış bir Ridge meta-model birleştirdi. Üstüne iki aşamalı bir kuyruk kalibrasyonu koyduk ve son adımda birbirinden farklı çıktıları harmanladık.

Yöntemin uzun anlatımı notebook'ta.

## Final submission

```
bv34_v44s = 0.9 * clip(0.425*v49cons + 0.425*v47 + 0.15*v34, 0, 100) + 0.10*v44s
          = 0.3825*v49cons + 0.3825*v47 + 0.135*v34 + 0.10*v44s
```

Dört bileşenin sabit ağırlıklı harmanı. Tamamen deterministik, yani aynı bileşenlerden her zaman aynı dosyayı veriyor:

```bash
python blend_final.py
# submission_bv34_v44s_repro.csv üretir
# ve submit ettiğimiz dosyayla karşılaştırır: max fark ~4e-14 (pratikte birebir)
```

## Sonucu nasıl tekrar üretirsiniz

Notebook'un çekirdek yöntemi (özellik üretimi, TF-IDF, ağaç toplulukları, meta ve kuyruk) CPU'da uçtan uca koşuyor ve grafikleriyle birlikte gömülü geliyor, saklı bir adım yok. Final submission'ı `blend_final.py` ile birebir üretebilirsiniz, submit ettiğimiz dosyayla farkı ölçülemeyecek kadar küçük.

Tek istisna, yedi milyarlık dil modelini sıfırdan eğitmek; o kısım ayrı GPU saatleri istiyor ve kodu gpu_kernels/ altında. Dil modelleri tam olarak deterministik olmadığı için, modeli taze bir RTX 3090'da yeniden eğittiğimizde orijinalle yaklaşık yüzde 98 korelasyon aldık. Final'i birebir üretmek için yeniden eğitmeye gerek yok; bileşen çıktıları submissions/ altında hazır duruyor.

## Üzerinde durduğumuz birkaç nokta

Zorunlu Türkçe metni, modelin en güçlü sinyallerinden biri yaptık. Sadece metne dayalı hatamız kademeli düştü: TF-IDF ile 144, embedding ile 139, SetFit ile 134, hibrit LoRA ile 89. En tepede gte-Qwen2-7B modelini LoRA ile eğittik ve öğrencinin tablo bilgileriyle mentör metnini tek bir girdide birleştirdik (gpu_kernels/rp_lora_hybridF.py).

Disiplin tarafında, her aday iki ayrı tohumla (321 ve 777) çapraz doğrulamadan geçti; ikisinde de iyileştirmeyeni almadık. Kendi ölçümümüzle gerçek skor neredeyse el ele gitti, o yüzden görünen küçük public dilimini değil kendi doğrulamamızı kovaladık.

Bir de denediğimiz ama bıraktığımız çok şey oldu: pseudo-label, derin ağaçlar, nonlineer meta, PFN-boost, çok kollu sinir ağı. Ölçüp de gerçek skora çevirmeyenleri eledik; örnekleri experiments/ altında.

## Çalıştırma

```bash
pip install -r requirements.txt

# Final submission'ı bileşenlerden birebir üret (hızlı, GPU gerekmez):
python blend_final.py
```

Çözümün tamamı `notebook/strawveri_datathon2026.ipynb` içinde; grafikler ve çıktılar gömülü olduğu için açar açmaz görünür. Notebook'un çekirdeğini kendiniz çalıştırmak isterseniz yarışma verisini (`train.csv`, `test_x.csv`) `data/` klasörüne koymanız yeterli. Ağır embedding ve dil modeli adımları ayrı bir GPU'da üretildiği için notebook'ta yorum satırına alındı; yerlerine TF-IDF skoru kullanıldı, böylece geri kalan her şey CPU'da koşuyor.

Takım strawveri: Furkan ADIKTI, İrem SARGIN. BTK Datathon 2026.
