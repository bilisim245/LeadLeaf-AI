# Kod Notlarım — Kendi Cümlelerimle Anlatmak İçin

Bu dosya, projedeki kodun her parçasını **basit anlatımla** açıklıyor. Amaç: bootcamp/jüri/mülakatta
"bunu nasıl yaptın, ne işe yarıyor" sorusuna kendi cümlelerimle cevap verebilmek.

Her bölüm: **kod parçası → sade anlatım**. Yeni parça açıklandıkça buraya eklenecek.

---

## Dosya: `notebooks/01_train_model_kaggle.py` (Kaggle'da model eğitimi)

### 1) Hazırlık — kütüphaneler + ayarlar

```python
import os, json, glob, random, shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
```

> Önce gerekli araçları çağırıyorum. `tensorflow` (Google'ın yapay zekâ kütüphanesi) modeli
> kuracağım yer; `matplotlib` grafik/görsel çizdirir; `os`, `json`, `shutil` dosyalarla uğraşmak için.

```python
IMG = 224
BATCH = 32
SEED = 42
MAX_PER_CLASS = None
EPOCHS_HEAD = 8
EPOCHS_FINE = 5
```

> - `IMG = 224`: her fotoğrafı 224×224 piksele küçültüyorum (model hepsini aynı boyutta ister)
> - `BATCH = 32`: modele fotoğrafları 32'lik gruplar halinde gösteriyorum (hepsini birden değil — bellek yetmez)
> - `SEED = 42`: rastgelelik sabitleniyor, her çalıştırmada aynı sonucu alıyorum (tekrarlanabilirlik)
> - `EPOCHS_HEAD/FINE`: modele veriyi kaç tur göstereceğim (aşağıda 2 aşamalı eğitimde açıklanıyor)

```python
SELECTED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Septoria_leaf_spot",
]
```

> Veri setinde 38 bitki/hastalık sınıfı var. İşi kolaylaştırmak için **sadece domatesi ve 4 yaygın
> hastalığını** seçtim: sağlıklı, erken yanıklık, geç yanıklık, bakteriyel leke, septoria yaprak
> lekesi. İleride bu listeyi kaldırıp (`None` yaparak) tüm bitkilere genişletebilirim.

### 2) Veriyi bulma

```python
CANDIDATES = [
    "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)",
    "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)",
]
DATA_ROOT = next((p for p in CANDIDATES if os.path.isdir(os.path.join(p, "train"))), None)
```

> Kaggle'a eklediğim veri seti bazen bir klasör içinde bir daha aynı isimli klasörde duruyor —
> bunu garantiye almak için iki olası yolu da deniyorum, hangisi varsa onu kullanıyorum.

```python
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VALID_DIR_FULL = os.path.join(DATA_ROOT, "valid")
```

> Veri iki parçaya ayrılmış geliyor: `train` (modelin **öğrendiği** fotoğraflar) ve `valid`
> (modelin **hiç görmediği**, kendini test ettiğimiz fotoğraflar). Bu ayrım şart — yoksa model
> ezberler, gerçek başarımını bilemeyiz.

### 3) Model kurulumu — MobileNetV2 transfer learning + FINE-TUNING

```python
base = keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")
base.trainable = False
...
h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, ...)   # 1. aşama

base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False
h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, ...)  # 2. aşama
```

> **"Fine-tuning yapıyor muyuz?" sorusuna cevap: EVET, zaten yapıyoruz.** İki aşamalı bir
> transfer learning stratejisi kullanıyoruz:
> 1. **1. aşama (feature extraction):** MobileNetV2, ImageNet'te 1.4 milyon görselle
>    önceden eğitilmiş. Bu ağırlıkları DONDURUP (`trainable = False`) sadece üstüne
>    eklediğimiz yeni sınıflandırma katmanını (`Dense(5, softmax)`) eğitiyoruz. Bu, az
>    veriyle hızlı ve stabil bir başlangıç sağlar.
> 2. **2. aşama (fine-tuning):** Base modelin SON 40 katmanının kilidini açıp
>    (`base.trainable = True` + ilk katmanları tekrar dondurma) çok küçük bir öğrenme
>    oranıyla (`1e-5`, 1. aşamadaki `1e-3`'ten 100x küçük) devam ediyoruz. Bu, ImageNet'in
>    genel görsel özelliklerini (kenar, doku) korurken, ağın SON katmanlarını "domates
>    yaprağı hastalığı" gibi bize özel detaylara ince ayarlıyor.
>
> **Neden TÜM ağı baştan fine-tune etmiyoruz?** Veri setimiz küçük (domates + 5 sınıf).
> Tüm 155 katmanı büyük öğrenme oranıyla eğitmek → EZBERLEME (overfitting) riski çok
> yüksek olurdu. Son 40 katmanla sınırlamak, "az veriyle güvenli fine-tuning" için
> standart bir pratiktir.
>
> **Ne zaman DAHA FAZLA fine-tuning yapılır?** Veri arttıkça (ör. 38 sınıfa/tüm bitkilere
> genişleyince — `SELECTED_CLASSES = None`), dondurulan katman sayısını azaltıp
> (`base.layers[:-40]` yerine `[:-80]` gibi) daha agresif fine-tuning denenebilir. Bu,
> TÜBİTAK/TEKNOFEST aşamasında planlanan bir iyileştirme.

### 4) Değerlendirme — confusion matrix + classification report

```python
y_pred = model.predict(val_ds).argmax(axis=1)
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
cm = confusion_matrix(y_true, y_pred)
```

> Modelin sadece "genel doğruluğunu" değil, HANGİ sınıfı HANGİ sınıfla karıştırdığını da
> görmek istiyoruz (confusion matrix). Örn. Erken yanıklık ile Septoria birbirine
> benzer göründüğünden (bkz. `agent/knowledge/`), model bunları karıştırabilir — bu
> beklenen ve rapora yazılacak bir sınırlılıktır, gizlenmeyecek.

---

## Dosya: `notebooks/00_veri_kesfi.py` (Veri seti keşfi — "veri setini ayrıntılı incele" geri bildirimi üzerine)

> Modele geçmeden önce veriyi VARSAYIMLA değil SAYIYLA incelemek için yazıldı. Üç şey
> ölçüyor: (1) sınıf dengesizliği, (2) görsel boyutu tutarlılığı, (3) **train/valid
> sızıntısı (data leakage)** — bu veri seti "Augmented" (döndürme/aynalama ile
> çoğaltılmış), yani aynı orijinal fotoğrafın varyasyonları teorik olarak hem train'e
> hem valid'e düşebilir. Bunu `imagehash.phash` (perceptual hash) ile örnekleyerek
> kontrol ediyoruz: iki görselin hash'i çok yakınsa (Hamming mesafesi ≤4), muhtemelen
> aynı fotoğrafın türevidirler. Sonuç yüksekse (ör. >%10), doğrulama doğruluğumuzun
> GERÇEKTE OLDUĞUNDAN yüksek görünebileceğini rapora sınırlılık olarak yazacağız —
> bu, PlantVillage tabanlı veri setlerine literatürde yöneltilen bilinen bir eleştiridir
> ve bunu bilip söylemek, görmezden gelmekten çok daha güçlü bir sunum noktasıdır.

---

## Dosya: `inference/app.py` (FastAPI — CNN'i HTTP ile sunar)

> Tek görevi: bir görsel al, model ile sınıflandır, JSON döndür. **DEMO MODU**: model
> dosyaları (`model/model.keras`) henüz yoksa (Colab eğitimi bitmediyse) servis
> ÇÖKMEZ — rastgele ama biçimce doğru bir sonuç üretir. Bu sayede n8n/Gradio akışının
> TAMAMI, gerçek model gelmeden test edilip gösterilebilir; model gelince (aynı
> dosya yoluna kopyalanınca) kod değişmeden gerçek tahmine geçer.
>
> **Görsel RAG entegrasyonu:** Gerçek model yüklendiğinde, modelin son sınıflandırma
> katmanından (Dense+softmax) BİR ÖNCEKİ katmanı (`GlobalAveragePooling2D` çıktısı,
> 1280 sayılık bir vektör) ikinci bir "embedding modeli" olarak da kullanıyoruz.
> Neden ayrı bir görsel-embedding modeli (CLIP vb.) İNDİRMİYORUZ: modelimiz zaten
> "bu yaprak neye benziyor" bilgisini öğrenmiş durumda — bu temsili yeniden kullanmak
> hem ekstra indirme/karmaşıklık gerektirmiyor hem de doğrudan AÇIKLANABİLİR (modelin
> kendi kararına dayanan bir benzerlik). `/predict` yanıtına `benzer_gorseller` alanı
> olarak, veri setinden en yakın referans görseller + benzerlik yüzdesi ekleniyor.

---

## Dosya: `rag/build_index.py` + `agent/rag.py` (Metin RAG — hastalık bilgi tabanı)

> **Neden RAG:** LLM'e "bu hastalık nedir, ne yapılmalı" diye sorduğumuzda, cevabı kendi
> eğitim verisinden (ezberinden) üretir — bu hem HALÜSİNASYON riski taşır hem de "bu
> bilgi nereden geliyor" sorusuna cevap veremeyiz. Bunun yerine, `agent/knowledge/`
> altına HER hastalık için elle yazılmış, doğrulanmış bir doküman (etken, belirtiler,
> uygun koşullar, karıştırılabilecek hastalıklar, kültürel/biyolojik önlemler) koyduk.
> `rag/build_index.py` bunları parçalara (chunk) ayırıp çok dilli bir embedding modeliyle
> (`paraphrase-multilingual-MiniLM-L12-v2` — Türkçe metin için seçildi) vektörleştirip
> Chroma vektör veritabanına kaydediyor. `agent/rag.py`'deki `retrieve_context()`,
> CNN'in bulduğu hastalık adına göre en alakalı parçaları geri getirip LLM'in
> promptuna ekliyor — LLM artık "kaynağa dayalı" konuşuyor, ezberden değil.
>
> **Neden sadece 5 dokümanla "gerçek" vektör arama yapıyoruz, direkt sözlük (dict)
> yeterli olmaz mıydı?** Fonksiyonel olarak evet, 5 sınıf için basit bir sözlük de
> işi görür. Ama vektör tabanlı tasarım (a) veri tabanı büyüdükçe (38 sınıfa
> çıkınca) DOĞRUDAN ölçeklenir, (b) serbest metin sorgularla da (ör. çiftçinin kendi
> tarif ettiği belirtiyle) arama yapılmasına izin verir — bu, "İleri seviye" bootcamp
> hedefine (RAG ile zenginleştirilmiş öneri) uygun, ileriye dönük bir mimari kararı.

---

## Dosya: `bot/db.py` + `agent/weather.py` (artık MVP'de — Tarla 360 dashboard'u besliyor)

> `db.py`: SQLite tabanlı basit bir "tarla defteri" — her analiz kaydını (hastalık,
> güven, tarih) saklar. `ui/app.py`, aynı çiftçi/tarla için geçmiş kayıtları çekip
> bir TREND GRAFİĞİ çiziyor (referans dashboard'daki "alışveriş aralığı trendi"nin
> karşılığı). `recent_cluster()` fonksiyonu, "son 7 günde aynı ilçede kaç farklı
> çiftçi aynı hastalığı bildirdi" sorusuna cevap veriyor — bölgesel salgın erken
> uyarısının temeli.
>
> `weather.py`: Open-Meteo'dan (ücretsiz, API anahtarı gerekmez) o bölgenin 3 günlük
> hava tahminini çekip basit bir sezgisel kuralla ("nem ≥%80 veya yağış ≥10mm → yüksek
> risk") mantar hastalığı riski hesaplıyor. Bu, "Riski düşürmek için ne yapmalı?"
> senaryo tablosundaki risk skoruna girdi oluyor.

---

## "Zaman serisi için ayrı bir veri seti gerekir mi?" — HAYIR

Bootcamp'in "İleri seviye" hedefi "geçmiş raporlarla trend analizi" istiyor. Bunun için
DIŞARIDAN bir zaman serisi veri setine ihtiyaç YOK — zaman serisi, sistemin KENDİ
kullanımından doğal olarak birikiyor: her analiz `bot/db.py`'ye bir gözlem (tarih +
hastalık + güven) olarak kaydediliyor; aynı çiftçi/tarla tekrar fotoğraf attıkça bu
kayıtlar birikip `ui/app.py`'deki "📈 Geçmiş trend" grafiğini oluşturuyor. Yani zaman
serisi verisi = ürünün kendi telemetrisi, ayrı bir Kaggle veri seti değil.

## "Veri Analizi" sekmesi — bootcamp'in istediği EDA nerede?

`notebooks/00_veri_kesfi.py` Colab'da çalışıp şunları üretiyor: sınıf dağılımı/dengesizliği,
görsel boyutu istatistiği, **bozuk/açılamayan görsel** kontrolü ("boş veri" karşılığı —
tablo verisinde NaN'a denk gelen şey, görsel veride 0 byte'lık veya PIL'in açamadığı
dosyalardır), ve **tekrar eden (duplicate) görsel** kontrolü (train/valid'e ayrı ayrı
düşüp yapay sızıntı yaratabilecek neredeyse-aynı fotoğraflar). Bu, indirilen
`eda_ciktilari.zip` içeriği `report/eda_ciktilari/` klasörüne çıkarılınca `ui/app.py`'nin
**"📊 Veri Analizi"** sekmesinde otomatik olarak grafiklere/tablolara dökülüyor — Colab
çıktısını statik bir PNG olarak rapora yapıştırmak yerine, canlı ve gezilebilir.

## "Senaryo analizi" tablosu neden ML modeli DEĞİL (dürüstlük notu)

`ui/app.py`'deki tablo ("Mevcut durum" / "Kültürel önlem" / "Tekrar kontrol" / "Uzmana
danış" ve her biri için bir risk yüzdesi) referans alınan dashboard'daki "Riski
düşürmek için ne yapmalı?" panelinden esinlendi. AMA o dashboard muhtemelen gerçek
geçmiş verilerle EĞİTİLMİŞ bir öneri/uplift modeli kullanıyor; bizim burada milyonlarca
etiketli "müdahale → sonuç" verisi yok. Bu yüzden tablomuz **kural tabanlı, açıkça
etiketlenmiş bir simülasyondur** (katsayılar: kültürel önlem ~%30 risk azaltımı, uzmana
danışma ~%50 — sezgisel, kalibre edilmemiş). Sunumda bu ayrımı net yapmak
("yapısı aynı, ama biz kalibre edilmiş bir model değil şeffaf bir sezgisel kural
kullandık") hem dürüst hem de veri biliminde olgunluk göstergesidir — sahte kesinlik
iddia etmemek, MVP'nin zaten benimsediği "kesin teşhis değil, ön değerlendirme"
ilkesiyle birebir tutarlı.
