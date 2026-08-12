# HANDOFF UPDATE — Doğrulama, Final Test, SHAP (2026-08-10)

Bu belge, önceki handoff özetindeki sayıları mevcut kod tabanı ve CSV'lerle karşılaştırıp
doğrulamak, eksik olan final test değerlendirmesini üretmek ve SHAP analizini eklemek için
yapılan çalışmanın sonucudur. **Hiçbir model yeniden eğitilmedi, train/validation/test split'i
değiştirilmedi, test seti sadece bir kez (final test aşamasında) kullanıldı.**

Tüm sayılar aşağıda hangi dosyanın hangi satırından geldiği belirtilerek raporlanmıştır.

---

## 0. Repo keşfi — isim/yapı notları

- Modüller handoff özetiyle uyumlu: `src/data_loader.py`, `src/preprocessing.py`,
  `src/model_registry.py`, `src/train.py`, `src/evaluation.py`, `src/run_30_models.py`,
  `src/attention_models.py`, `src/selfies_features.py`, `src/imputation.py` — hepsi mevcut.
- `results/tables/` **kanonik** sonuç klasörüdür. `results/corrupt_backup/` ve
  `results/smoke_backup/` klasörleri **rapor için kullanılmamalı**:
  - `corrupt_backup/kfold_results.csv`: 211 satır, `max_iter=50` gibi smoke-test parametreleriyle
    üretilmiş, model registry ile tutarsız bir erken/bozuk koşum (adından da anlaşılacağı gibi).
  - `smoke_backup/`: kasıtlı olarak küçük bir "smoke test" koşumu (24 satır/split, 2-fold CV).
  - Her iki klasördeki sayılar `results/tables/kfold_results.csv` ile **eşleşmiyor** ve
    raporda kullanılırsa yanlış sonuç verir.
- `src/train.py` içinde zaten tam bir `run_final_test()` fonksiyonu vardı (kazanan modeli
  train+validation üzerinde eğitip test setinde tek seferlik değerlendiren, sonucu
  `results/tables/final_test_results.csv`'ye yazan resmi pipeline aşaması). Bu adım
  daha önce hiç çalıştırılmamıştı — `results/tables/final_test_results.csv` bu oturumdan
  önce **repoda yoktu** (yalnızca `corrupt_backup/` ve `smoke_backup/` içinde 2 baytlık,
  yalnızca başlık satırından oluşan boş dosyalar vardı).
- `results/tables/leaderboard_{density,pld,lcd}.csv`, `learning_curve_summary.csv`,
  `failed_runs.csv` da bu oturumdan önce `results/tables/` içinde yoktu; bu koşum sırasında
  pipeline'ın kendi yan etkisi olarak (winner seçimi / eksik tablo doldurma) otomatik üretildi.
- **Notebook `notebooks/08_results_analysis.ipynb` tamamen boş (0 byte).** Handoff özetinde
  bahsedilen "mevcut permütasyon-tabanlı grup önem testi" repoda (ne `src/` ne `notebooks/`
  içinde) **bulunamadı** — muhtemelen önceki oturumda yalnızca konuşulmuş ama koda/notebook'a
  yazılmamış, ya da bu repoya commit edilmemiş. Bu yüzden bu görevde SHAP'e ek olarak
  sıfırdan bir permütasyon-tabanlı grup önem testi de üretildi (bkz. Bölüm 3), böylece
  karşılaştırma istendiği gibi yapılabildi.
- `requirements.txt` UTF-16 kodlanmış bir dosya (muhtemelen `pip freeze > requirements.txt`
  PowerShell'de yapılırken oluşmuş); içerik doğru ama düz metin okuyucularla açarken dikkat.

---

## 1. Düzeltilen / Doğrulanan Sayılar

### 1.1 SGDRegressor tutarsızlığı (ÇÖZÜLDÜ)

Elindeki iki farklı rakam da **doğru** — farklı değerlendirme aşamalarından geliyorlar, dosya
hatası ya da yanlış hesaplama değil:

| Kaynak | Dosya / satır | density validation RMSE | density validation R² |
|---|---|---|---|
| **Holdout (tek 70/15/15 split)** | `results/tables/holdout_results.csv`, `stage=holdout, target=density, model=sgd_regressor` | **4.6797 × 10¹¹** (467 971 349 461.02) | −5.5665 × 10²³ |
| **5-fold CV ortalaması** | `results/tables/kfold_results.csv`, `stage=kfold, target=density, model=sgd_regressor` | **9.5053 × 10¹¹** (950 530 290 634.45, std=4.380×10¹⁰) | −2.2218 × 10²⁴ |

**Raporda kullanılacak sayı:** Rapor "5 katlı CV" protokolünü referans alıyorsa
(30 modelin karşılaştırıldığı ana protokol budur), doğru rakam **9.51 × 10¹¹**'dir
(kaynak: `kfold_results.csv`). 4.68×10¹¹ rakamı yanlış değil, sadece farklı bir aşamaya
(tek seferlik holdout validation) ait — eğer bu ikisinden biri rapora giriyorsa mutlaka
"holdout" ya da "5-fold CV" olarak etiketlenmeli, aksi halde okuyucu ikisini karıştırır.

Her iki aşamada da R² değeri feci şekilde negatif (~−10²³ ile −10²⁴ mertebesinde) —
yani SGDRegressor bu problemde (ham/ölçeklenmemiş hedef büyüklüğü + ~1794 boyutlu seyrek
TF-IDF öznitelik uzayı) sayısal olarak ıraksıyor. Bu, tek bir kötü koşumun tesadüfü değil,
iki bağımsız değerlendirme aşamasında da tutarlı şekilde gözlemleniyor — modelin bu veri
setinde kullanılamaz olduğu sonucu güvenle raporlanabilir.

### 1.2 Kazanan model metrikleri (DOĞRULANDI — birebir eşleşiyor)

Kaynak: `results/tables/kfold_results.csv`, `stage=kfold` satırları (`cv_validation_rmse_mean`,
`cv_validation_r2_mean` kolonları).

| Hedef | Model | Elindeki RMSE | CSV'deki RMSE | Elindeki R² | CSV'deki R² | Durum |
|---|---|---|---|---|---|---|
| density | LightGBM (`lgbm_regressor`) | ≈0.2684 | 0.268351 | ≈0.8233 | 0.823323 | ✅ eşleşiyor |
| pld | Extra Trees (`extra_trees`) | ≈1.3925 | 1.392520 | ≈0.8384 | 0.838375 | ✅ eşleşiyor |
| lcd | Extra Trees (`extra_trees`) | ≈1.5223 | 1.522330 | ≈0.8648 | 0.864764 | ✅ eşleşiyor |

Kazananlar `write_leaderboards()` sıralama mantığıyla (önce `cv_validation_rmse_mean` artan,
sonra `cv_validation_r2_mean` azalan) bağımsız olarak da yeniden üretildi ve aynı üç model
çıktı; `results/tables/leaderboard_density.csv` içinde `lgbm_regressor` rank=1 olarak
doğrulandı (dosya bu oturumda üretildi, bkz. Bölüm 0).

---

## 2. Final Test Seti Değerlendirmesi

### 2.1 Yöntem ve gerekçe

`src/train.py::run_final_test()` fonksiyonu (halihazırda kodda mevcut, hiç çalıştırılmamıştı)
kullanıldı: `python -m src.run_30_models --mode full --stages final_test` komutuyla, **sadece**
final_test aşaması çalıştırıldı (holdout/kfold/groupkfold/learning_curve aşamaları atlandı —
30 model yeniden eğitilmedi). Çalıştırmadan önce ve sonra `dataset_hash` / `split_hash` değerleri
programatik olarak `kfold_results.csv`'deki kayıtlarla karşılaştırıldı ve **birebir eşleştiği**
doğrulandı (`45b1899e...bb558` / `d6d1e615...de48`) — yani final test, 30 model karşılaştırmasında
kullanılan tam olarak aynı split üzerinde, test setine daha önce hiç dokunulmamış haliyle çalıştı.

**Train+validation birleştirme kararı:** Kazanan modeller train+validation birleşimi
(12 143 + 2 602 = 14 745 satır) üzerinde eğitildi, test setinde (2 603 satır) tek seferlik
değerlendirildi. Gerekçe: validation seti zaten model/algoritma seçimi (leaderboard sıralaması)
için kullanıldı; seçim artık kesinleştiği için validation'ı ayrı tutmanın ek bir "sızıntı
önleme" faydası kalmıyor, buna karşılık final modeli mümkün olan en fazla veriyle eğitmek
genelleme performansını iyileştirir. Bu zaten pipeline'ın tasarlanmış standart pratiğidir,
ayrıca elle bir karar/kod değişikliği gerekmedi.

Her iki ExtraTrees fit'i ~135-145 saniye, LightGBM fit'i ~3 saniye sürdü; toplam final_test
koşumu ~5 dakika. Kod: `results/models/{target}_{model}.joblib` içine kaydedilen 3 model,
üçü de tam olarak bir kez eğitildi ve test setinde tam olarak bir kez tahmin üretti.

### 2.2 Sonuç tablosu

Kaynak: `results/tables/final_test_results.csv` (ham, tüm kolonlarla) ve
`results/tables/final_test_summary.csv` (rapor için temiz alt küme).

| Hedef | Model | Test N | Test MAE | **Test RMSE** | **Test R²** | Test Median AE | Test Normalized RMSE |
|---|---|---|---|---|---|---|---|
| density | lgbm_regressor | 2603 | 0.187577 | **0.257124** | **0.819803** | 0.140410 | 0.403878 |
| pld | extra_trees | 2603 | 0.790868 | **1.321797** | **0.831653** | 0.432601 | 0.377604 |
| lcd | extra_trees | 2603 | 0.876586 | **1.433301** | **0.867168** | 0.526929 | 0.344797 |

(Train+validation üzerindeki fit metrikleri de `final_test_results.csv`'de
`train_validation_*` önekiyle mevcuttur; density R²=0.888, pld R²=0.985, lcd R²=0.989 — beklenen
şekilde test'ten yüksek, çünkü bu ağaç modelleri eğitim verisine güçlü şekilde uyum sağlıyor.)

### 2.3 5-fold CV ile karşılaştırma

Kaynak: `results/tables/cv_vs_test_comparison.csv`.

| Hedef | CV RMSE (ort±std) | Test RMSE | Fark (test−CV) | CV R² | Test R² | Fark |
|---|---|---|---|---|---|---|
| density | 0.2684 ± 0.0062 | 0.2571 | **−0.0112 (−4.2%)** | 0.8233 | 0.8198 | −0.0035 |
| pld | 1.3925 ± 0.1163 | 1.3218 | **−0.0707 (−5.1%)** | 0.8384 | 0.8317 | −0.0067 |
| lcd | 1.5223 ± 0.0922 | 1.4333 | **−0.0890 (−5.8%)** | 0.8648 | 0.8672 | +0.0024 |

**Yorum:** Üç hedefte de test RMSE'si, 5-fold CV ortalamasından hafifçe **daha iyi** (düşük)
çıktı — fark, CV standart sapmasının (~1 std) ötesine geçmiyor ve R² farkları hepsinde 0.007'nin
altında. Bu, model seçiminin (30 model arasından en iyisini seçmenin) validation setine aşırı
uyum sağlamadığını, CV tahmininin görülmemiş test verisine güvenilir şekilde genellediğini
gösteriyor. Test RMSE'sinin sistematik olarak biraz daha iyi çıkması muhtemelen final modelin
CV'deki her bir fold'dan (~9 714 satır) daha fazla veriyle (train+validation birleşimi,
14 745 satır) eğitilmiş olmasından kaynaklanıyor.

### 2.4 Üretilen görseller

- `results/plots/density_lgbm_regressor_test_parity.png`
- `results/plots/pld_extra_trees_test_parity.png`
- `results/plots/lcd_extra_trees_test_parity.png`
  (aynı grafiklerin orijinalleri + residual plot karşılıkları `results/figures/parity_plots/`
  ve `results/figures/residual_plots/` altında `*_test.png` olarak da mevcut, pipeline'ın
  standart çıktı konumu)

---

## 3. SHAP Analizi

### 3.1 Yöntem

- Kazanan 3 modelin `results/models/*.joblib` içindeki (train+validation üzerinde eğitilmiş,
  final) pipeline'ları yüklendi; SHAP, sklearn/lightgbm modelinin kendisine (pipeline'ın
  `model` adımına), ön işleme sonrası yoğun (dense) öznitelik matrisine uygulandı
  (1794 öznitelik: 838 linker TF-IDF + 727 node TF-IDF + 31 point_group one-hot +
  197 topology one-hot + 1 topology_missing).
- Örneklem **validation setinden** alındı (test seti değil — test seti sadece Bölüm 2'deki
  tek seferlik metrik için kullanıldı, SHAP için tekrar dokunulmadı), `random_state=42` ile
  deterministik: density için 600 satır, pld/lcd için 500 satır (+100 satır arka plan verisi).
- **Önemli sayısal not:** LightGBM için varsayılan hızlı ("tree_path_dependent") TreeSHAP
  algoritması makine hassasiyetinde doğru sonuç verdi (additivity hatası ~1.2×10⁻¹³). Ancak
  kazanan **ExtraTrees** modellerinde bu hızlı algoritma **sayısal olarak patladı**
  (additivity hatası ~8.8×10¹⁹ — tamamen anlamsız) çünkü `min_samples_leaf=2` ve
  `max_depth=None` ile eğitilen ağaçlar aşırı derin (örneklenen 20 ağaçta gözlenen derinlik:
  86-139!) — bu, LightGBM'in sığ ağaçlarıyla (`num_leaves=31`) tezat oluşturuyor. Çözüm:
  ExtraTrees için `feature_perturbation="interventional"` + 100 satırlık arka plan örneği
  kullanıldı (additivity hatası ~10⁻⁷'ye düştü, kabul edilebilir). Bu yüzden pld/lcd SHAP
  grafikleri density'ye göre biraz daha yaklaşık (100 arka plan satırına göre enterpolasyon)
  ama sayısal olarak sağlam.

### 3.2 Üretilen görseller (her hedef için)

`results/figures/shap/{target}_{model}_{tür}.png`:
- `bar` — en önemli 20 özniteliğin ortalama |SHAP| değeri
- `beeswarm` — dağılım + öznitelik değeri renklendirmesi
- `waterfall_low` / `waterfall_median` / `waterfall_high` — düşük/orta/yüksek tahminli
  3 örnek MOF için bireysel tahmin ayrıştırması

### 3.3 Grup bazlı özet ve permütasyon karşılaştırması

Kaynak: `results/tables/shap_group_summary.csv` (SHAP, toplam |SHAP| değerinin grup payı) ve
`results/tables/permutation_group_importance.csv` (bu oturumda sıfırdan üretilen permütasyon
bazlı grup önemi — aynı örneklem üzerinde, grubun tüm kolonları satırlar arası karıştırılıp
RMSE'nin ne kadar bozulduğu ölçülerek, 10 tekrarın ortalaması).

| Hedef | Grup | SHAP payı (%) | Permütasyon payı (%) |
|---|---|---|---|
| density | linker_selfies | 47.3 | 43.1 |
| density | node_selfies | 33.3 | 40.5 |
| density | topology | 9.4 | 9.4 |
| density | point_group | 9.9 | 7.0 |
| pld | linker_selfies | 42.1 | 28.4 |
| pld | node_selfies | 25.9 | 23.9 |
| pld | topology | 13.7 | 22.8 |
| pld | point_group | 16.2 | 22.6 |
| lcd | linker_selfies | 42.7 | 28.8 |
| lcd | node_selfies | 29.6 | 32.7 |
| lcd | topology | 12.5 | 17.2 |
| lcd | point_group | 11.8 | 17.3 |

**Özet (3-5 cümle):** İki yöntem de her üç hedefte de `linker_selfies` ve `node_selfies`
özellik gruplarının en baskın iki grup olduğu konusunda **hemfikir** — linker (bağlayıcı
molekül) SELFIES token'ları tutarlı şekilde en yüksek katkıyı veriyor, node (metal düğüm)
token'ları ikinci sırada. `topology_missing` bayrağı her iki yöntemde de ihmal edilebilir
düzeyde (density'de SHAP'e göre sıfıra yakın, permütasyona göre de düşük). En büyük fark
pld/lcd'de `topology` ve `point_group` gruplarının göreli ağırlığında: permütasyon testi
bu iki gruba SHAP'ten belirgin şekilde daha fazla pay veriyor (~%22-23 vs ~%14-16) — bu
beklenen bir durum, çünkü SHAP ortalama|değer| bireysel tahmine yapılan marjinal katkıyı
ölçerken, permütasyon testi bir grubun **tamamen** karıştırılmasının toplam RMSE'ye etkisini
ölçüyor (öznitelikler arası etkileşimleri de içerir); bu yüzden iki metrik birbirinin
yerine geçmez ama üst sıradaki iki grubun sıralaması konusunda birbirini doğruluyor. Bireysel
örnek düzeyinde (waterfall grafikleri), density için Zn içeren node token'ları ve
`topology_missing` bayrağı; pld/lcd için Cu node'u, `srs` topolojisi, `point_group_1` ve
belirli N/C içeren linker n-gramları en güçlü bireysel katkı sağlayan öznitelikler olarak öne
çıkıyor — kimyasal olarak makul (Zn, QMOF veri setindeki en yaygın metal düğüm).

---

## 4. Üretilen Tüm Yeni Dosyalar

**Tablolar (`results/tables/`):**
- `final_test_results.csv` — pipeline'ın kanonik final test çıktısı (tüm kolonlarla)
- `final_test_summary.csv` — rapor için temizlenmiş özet
- `cv_vs_test_comparison.csv` — CV vs test karşılaştırma tablosu
- `shap_group_summary.csv` — SHAP grup bazlı özet
- `permutation_group_importance.csv` — permütasyon bazlı grup önemi (yeni, sıfırdan üretildi)
- `leaderboard_density.csv`, `leaderboard_pld.csv`, `leaderboard_lcd.csv` — final_test
  koşumunun yan etkisi olarak dolduruldu (önceden yoktu)
- `learning_curve_summary.csv`, `failed_runs.csv` — pipeline tarafından boş placeholder
  olarak oluşturuldu (bu koşumda ilgili aşamalar çalıştırılmadı)

**Görseller:**
- `results/plots/density_lgbm_regressor_test_parity.png`
- `results/plots/pld_extra_trees_test_parity.png`
- `results/plots/lcd_extra_trees_test_parity.png`
- `results/figures/parity_plots/{density,pld,lcd}_{lgbm_regressor,extra_trees}_test.png` (kaynak)
- `results/figures/residual_plots/{density,pld,lcd}_{lgbm_regressor,extra_trees}_test.png`
- `results/figures/shap/density_lgbm_regressor_{bar,beeswarm,waterfall_low,waterfall_median,waterfall_high}.png`
- `results/figures/shap/pld_extra_trees_{bar,beeswarm,waterfall_low,waterfall_median,waterfall_high}.png`
- `results/figures/shap/lcd_extra_trees_{bar,beeswarm,waterfall_low,waterfall_median,waterfall_high}.png`

**Modeller:**
- `results/models/density_lgbm_regressor.joblib`
- `results/models/pld_extra_trees.joblib`
- `results/models/lcd_extra_trees.joblib`

**Kod (yeniden çalıştırılabilir, `scripts/`):**
- `scripts/build_final_test_summary.py`
- `scripts/shap_analysis.py`

**Diğer:**
- `results/run_manifest.json` — final_test koşumuyla güncellendi (dataset/split hash'leri
  aynı kaldı, sadece final_test aşamasının çalıştığı bilgisi eklendi)
- `results/run_manifest_kfold_run_backup.json` — final_test'ten önceki (30 modelin
  eğitildiği) orijinal manifest'in yedeği, kayıt amaçlı saklandı

---

## Öncelik durumu

Talep edilen 4 adımın hepsi tamamlandı: (1) doğrulama, (2) final test seti, (3) SHAP
(beeswarm + waterfall dahil, basitleştirilmiş versiyon değil), (4) bu belge.
