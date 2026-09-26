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

---

# 26 Eylül 2026 — "Mısır sanıyor, domates deyince konuyu dağıtıyor" hatası ve düzeltmesi

Bu bölüm, gerçek bir Telegram testinde çıkan bir hatanın **nasıl bulunduğunu, neden olduğunu ve
nasıl düzeltildiğini** adım adım anlatıyor. Jüri "sistemde hata çıktı mı, nasıl çözdünüz?" diye
sorarsa anlatılacak hikâye bu.

## Önce bir kavram: "betik" (script) nedir?

**Betik**, bir işi baştan sona otomatik yapan, **çalıştırılınca işini bitirip kapanan** küçük bir
programdır. Elle tek tek yapılacak adımları (tıkla, kopyala, yapıştır, düzelt…) bir dosyaya yazarsın,
bilgisayar o adımları senin yerine aynı sırayla yapar.

Projede iki tür Python dosyası var, farkı önemli:

| | **Betik** | **Servis (sunucu)** |
|---|---|---|
| Örnek | `n8n/bitki_duzeltme_ekle.py`, `n8n/tam_akis_olustur.py`, `sunum/karar_defteri_pdf.py` | `inference/app.py` (FastAPI) |
| Ne yapar | Bir kere çalışır, bir çıktı üretir (dosya, PDF), kapanır | Açık kalır, gelen istekleri bekler ve cevaplar |
| Nasıl çalışır | `.venv\Scripts\python n8n\bitki_duzeltme_ekle.py` | `.venv\Scripts\python -m uvicorn inference.app:app --port 8000` |
| Benzetme | Yemek tarifi: uygula, yemek çıksın, bitti | Lokanta: kapı açık, müşteri geldikçe servis |

**Neden n8n değişikliğini elle değil de betikle yaptık?**
1. **Tekrarlanabilir:** Aynı betik tekrar çalıştırılınca aynı sonucu verir. Elle yapınca bir adımı
   atlamak/yanlış yazmak kolay (22 Eylül'de Sheets alanlarının "Fixed" kalması tam böyle bir hataydı).
2. **Okunabilir/denetlenebilir:** Ne değiştiği satır satır dosyada yazılı; git'e girer, geçmişi görülür.
3. **Güvenli:** Betik n8n veritabanını **salt okunur** açar (`mode=ro`), canlı akışa hiçbir şey
   YAZMAZ. Sadece yeni bir `.json` dosyası üretir; onu n8n'e içe aktarıp yayınlama kararı bizde kalır.

`n8n/bitki_duzeltme_ekle.py` şunu yapıyor: canlı akışın ("tam") bugünkü hâlini n8n'in veritabanından
okur → aşağıda anlatılan değişiklikleri uygular → `n8n/leadleaf_tam_akis.json` dosyasını yazar
("tam v2"). **Neden canlı akışı okuyor da eski dosyayı değil?** Çünkü canlı akış sabah n8n içinde
açılıp kaydedilmişti; eski dosyadan üretseydik n8n'de yapılmış değişiklikler kaybolabilirdi. (Kontrol
edildi: farklar sadece n8n'in varsayılan alanları silmesiydi, gerçek düzenleme yoktu — ama bunu
varsaymak yerine en güncel hâlin üzerine kurmak doğru yol.)

## 1) Belirti — kullanıcı ne gördü?

Gerçek bir domates yaprağı fotoğrafı **açıklama yazmadan** gönderildi:
- Bot: "Mısır — Sağlıklı" dedi.
- Ardından "Bu mısır değil" ve "Domates" yazıldı → bot fotoğrafla hiç ilgisi olmayan genel sohbet
  cevapları verdi ("Domates yetiştiriciliğiyle mi ilgileniyorsunuz? … fotoğraf gönderirseniz…").

## 2) Teşhis — hatayı nasıl bulduk?

Tahmin yürütmek yerine **n8n'in kendi kayıtlarına** baktık. n8n her çalıştırmayı (execution)
veritabanına kaydeder: hangi düğümler çalıştı, her düğüme ne girdi, ne çıktı. Bu kayıtlar
(`~/.n8n/database.sqlite`, tablolar `execution_entity` + `execution_data`) yine salt okunur açılıp okundu:

| Çalıştırma | Gelen mesaj | Hangi dal | Ne oldu |
|---|---|---|---|
| #59 | Fotoğraf (açıklama yok) | Fotoğraf dalı | CNN: `Corn_(maize)___healthy` **%29**, 2. sırada domates sarı yaprak kıvırcıklığı %22,1 → güven < %70 olduğu için "Uzmana Bildir" de çalıştı |
| #60 | "Bu mısır değil" | Sohbet dalı | Genel sohbet cevabı |
| #61 | "Domates" | Sohbet dalı | Genel sohbet cevabı |

**Bütün çalıştırmalar "success" (başarılı).** Yani program çökmedi — hata **mantıkta/tasarımda**.
Ders: "başarılı" yeşil tik, "doğru çalıştı" demek değildir; girdi-çıktıya bakmak gerekir.

## 3) Kök nedenler — neden oldu?

**Kök neden 1 — Model neden mısır dedi?**
- Açıklama yazılmadığı için model **38 sınıfın hepsi** arasında seçim yaptı (bitki filtresi devreye girmedi).
- Model PlantVillage ile eğitildi: laboratuvarda, düz arka planda, tek yaprak fotoğrafları. Tarlada,
  karışık arka planlı gerçek bir fotoğraf onun gördüğü dünyadan farklı (**alan kayması / domain shift**).
  Grad-CAM sayfasında gösterdiğimiz "model bazen arka plana bakıyor" bulgusu da bununla ilgili.
- **Önemli:** Güven sadece %29'du ve sistem bunu doğru şekilde "emin değilim → uzmana yönlendir"
  olarak işaretledi. Yani **%70 eşiği işini yaptı.** Eksik olan: çiftçiye ne yapabileceğini
  söylememesiydi.

**Kök neden 2 — "Domates" yazınca neden konu dağıldı?**
- n8n akışında yazı mesajı gelince "Fotoğraf var mı?" → hayır → "Selamlaşma mı?" → hayır →
  **sohbet dalı** (Claude). Sohbet dalı her mesajı **tek başına** görüyor; önceki fotoğraftan
  haberi yok (**durumsuz / stateless**). Ona "Domates" bağlamsız tek kelimelik bir sohbet
  mesajı gibi geliyor, o da elinden gelen en makul genel cevabı veriyor.
- Yani sohbet LLM'i "yanlış" davranmadı; ona gereken bilgi (önceki fotoğraf) hiç verilmemişti.

## 4) Çözüm fikri — ne yapmalıydı?

İnsan gibi düşün: bir ziraat mühendisine fotoğraf gösterdin, "mısır" dedi, sen "hayır, domates"
dedin. Mühendis **aynı fotoğrafa tekrar, bu sefer domates gözüyle bakar.** Bot da bunu yapmalı:

```
Fotoğraf gelir ──► son fotoğrafın kimliği (file_id) saklanır ──► normal rapor
       ...
"Domates" yazılır ──► "Bu bir bitki adı mı? Son 30 dk'da fotoğraf var mı?"
        ├─ EVET ──► son fotoğraf, "Domates" bilgisiyle fotoğraf dalına GERİ gönderilir
        │           → model sadece domates sınıfları arasında seçer → yeni rapor
        └─ HAYIR ──► normal sohbet (Claude)
```

**Neden sohbet LLM'ine "hafıza" vermedik (n8n'in Memory düğümü gibi)?** Çünkü hafıza olsa bile
Claude'un sohbet dalı **fotoğrafı göremiyor** ve teşhisi CNN yapıyor, LLM değil. "Domates"
bilgisiyle doğru sonucu verebilecek tek şey, CNN'i aynı fotoğrafla bitki filtresi açık şekilde
**yeniden çalıştırmak**. Bu deterministik (her seferinde aynı sonuç) ve açıklanabilir bir yol;
LLM'in "herhalde şudur" demesinden çok daha güvenilir.

## 5) Kod — FastAPI tarafı (`inference/app.py`)

### 5a) "Değil" kelimesini anlamak — `_bitki_bul`

Önce: metinde bitki adı geçiyor mu diye bakıyordu. "Bu mısır **değil**" → "mısır" buluyordu (tam tersi!).

```python
for ad in sorted(BITKI_ONEKLERI, key=len, reverse=True):
    for m in re.finditer(re.escape(ad), sade):
        if not re.match(r"\w*\s*degil", sade[m.end():]):
            return ad, BITKI_ONEKLERI[ad]
```

> - `sade`: metnin küçük harfe çevrilmiş, Türkçe harfleri sadeleştirilmiş hâli ("Değil" → "degil",
>   "Mısır" → "misir"). Böylece "MISIR", "mısır", "misir" hepsi aynı şekilde yakalanır.
> - `re.finditer`: metinde bitki adının geçtiği **her** yeri bulur.
> - `sade[m.end():]`: bitki adından **sonra** gelen kısım.
> - `re.match(r"\w*\s*degil", ...)`: bu kısım "(ek) + boşluk + değil" ile mi başlıyor?
>   `\w*` → "mısır**ın** değil" gibi ekleri, `\s*` → boşlukları kapsar.
> - Başlıyorsa bu bitki **reddedilmiş** demektir, atlanır; başlamıyorsa kabul edilir.
>
> Sonuçlar (test edildi): "Domates" → domates · "bu mısır değil" → **yok** · "Bu mısır değil, domates"
> → domates · "mısırın değil biberin fotoğrafı" → biber.

`re` = Python'un **düzenli ifade (regular expression)** kütüphanesi: metinde kalıp aramak için
kullanılan küçük bir dil.

### 5b) Son fotoğrafı hatırlamak — `_son_foto_kaydet` ve `/predict`

```python
SON_FOTO_YOLU = ".../data/son_fotolar.json"
SON_FOTO_SURESI = 30 * 60        # 30 dakika (saniye cinsinden)
SON_FOTO_EN_COK_KELIME = 6
```

`/predict` artık iki isteğe bağlı alan daha alıyor: `chat_id` (hangi sohbet) ve `file_id`
(fotoğrafın Telegram'daki kimliği). İkisi de gelirse kaydedilir:

```python
if chat_id and file_id:
    _son_foto_kaydet(chat_id, file_id)
```

> **Neden fotoğrafın kendisini değil `file_id`'yi saklıyoruz?** Telegram gönderilen fotoğrafı kendi
> sunucusunda tutuyor ve her fotoğrafa bir kimlik veriyor. Bu kimlikle fotoğraf istediğimiz zaman
> yeniden indirilebilir. Böylece bizim diskimizde çiftçinin fotoğrafı birikmiyor (gizlilik + yer),
> sadece ~90 karakterlik bir kimlik tutuluyor.
>
> **Neden JSON dosyası, neden bellek (değişken) değil?** Bellekte tutsaydık FastAPI yeniden
> başlatılınca (bugün birkaç kez oldu) kayıtlar silinirdi. Küçük bir dosya bu sorunu çözüyor.
> Veritabanı (Redis, PostgreSQL) bu ölçek için gereksiz karmaşıklık olurdu.
>
> **Neden 30 dakika?** Düzeltme fotoğraftan hemen sonra yapılır. Yarım gün sonra "domates" yazan
> biri büyük ihtimalle yeni bir konu açıyordur. Kaydederken 30 dakikadan eski kayıtlar da silinir,
> dosya büyümez.
>
> `data/` klasörü `.gitignore`'da → bu dosya GitHub'a gitmez (kullanıcı verisi).

### 5c) Yeni uç nokta — `GET /son-foto/{chat_id}?metin=...`

n8n, sohbet dalına girmeden önce buraya sorar. Üç kontrol, sırayla:

| Kontrol | Başarısızsa cevap |
|---|---|
| Metinde (reddedilmemiş) bir bitki adı var mı? | `yeniden: false` — "metinde bitki adı yok" |
| Mesaj kısa mı (≤ 6 kelime)? | `yeniden: false` — "uzun mesaj, sohbet olarak ele alınır" |
| Bu sohbette son 30 dk'da fotoğraf var mı? | `yeniden: false` — "son 30 dakikada fotoğraf yok" |
| Hepsi tamam | `yeniden: true, file_id: ..., bitki: "Domates"` |

> **Neden 6 kelime sınırı?** "Domates nasıl sulanır, her gün mü sulamalıyım acaba" da domates içeriyor
> ama bu bir düzeltme değil, bir soru. Kısa mesaj = düzeltme, uzun mesaj = sohbet. Basit ama işe
> yarayan bir kural (**sezgisel kural / heuristic**). `neden` alanı kararın gerekçesini açıkça
> döndürüyor (hata ayıklarken ve anlatırken çok işe yarıyor).

## 6) n8n tarafı (betik: `n8n/bitki_duzeltme_ekle.py` → `n8n/leadleaf_tam_akis.json`)

Yeni sohbet dalı:

```
Selamlaşma mı? ──hayır──► HTTP Request - Son Fotoğraf ──► Bitki düzeltmesi mi?
                                                           ├─ evet ─► Yeniden Değerlendirme Hazırla
                                                           │           ├─► Telegram - İnceleniyor
                                                           │           └─► Fotoğrafı İndir ─► (fotoğraf dalı aynen)
                                                           └─ hayır ─► Basic LLM Chain - Sohbet (eskisi gibi)
```

**Yeni düğümler:**

1. **HTTP Request - Son Fotoğraf** — `/son-foto/{chat_id}`'e sorar. Ayarı `onError:
   continueRegularOutput` → FastAPI kapalıysa bile akış durmaz, sohbet dalına devam eder.
   (Bir özellik bozulunca bütün botun bozulmaması: **zarif bozulma / graceful degradation**.)
2. **Bitki düzeltmesi mi?** (IF) — koşul `{{ $json.yeniden === true }}`. `=== true` yazıldı çünkü
   HTTP hata verirse `yeniden` alanı hiç olmaz; `undefined === true` → false → sohbete gider.
3. **Yeniden Değerlendirme Hazırla** (Code) — bu değişikliğin en "akıllıca" parçası:

```js
const mesaj = $('Telegram Trigger').first().json.message;
const son = $input.first().json;
return [{ json: { message: { ...mesaj, photo: [{ file_id: son.file_id }] }, yeniden_bitki: son.bitki } }];
```

> Gelen **yazı** mesajını, **fotoğraf mesajı gibi görünen** bir veriye çeviriyor: mesajın kendisini
> kopyalıyor (`...mesaj` = "bütün alanlarını al") ve içine `photo` alanı olarak son fotoğrafın
> kimliğini koyuyor. Böylece fotoğraf dalındaki **hiçbir düğümü değiştirmeden** (Fotoğrafı İndir,
> Predict CNN, RAG, Claude rapor, Sheets, PDF, uzmana bildir…) aynı yol yeniden kullanılıyor.
> "Fotoğrafı İndir" zaten `message.photo[son].file_id`'yi okuyordu — ona hiç dokunmadık.
> Alternatifi bütün fotoğraf dalını kopyalamaktı (iki kopya = bakım yükü, biri güncellenip öbürü
> unutulur).

**Değişen düğümler:**

| Düğüm | Değişiklik | Neden |
|---|---|---|
| HTTP Request - Predict CNN | `bitki` = `caption \|\| text \|\| ''` | Fotoğrafta bitki adı açıklamada (caption) olur; yeniden değerlendirmede ise yazı (text) olarak gelir |
| HTTP Request - Predict CNN | yeni alanlar `chat_id`, `file_id` (`$json.result.file_id`) | FastAPI son fotoğrafı saklayabilsin. `file_id`, "Fotoğrafı İndir" düğümünün çıktısından geliyor (gerçek bir çalıştırmada `result.file_id` içinde olduğu doğrulandı) |
| Telegram - İnceleniyor | yeniden değerlendirmede "🔁 Son gönderdiğiniz fotoğraf Domates olarak yeniden inceleniyor…" | Çiftçi ne olduğunu anlasın |
| Basic LLM Chain - Sohbet | metin artık `$('Telegram Trigger').item.json.message.text` | Eskiden `$json.message.text` idi. Araya HTTP düğümü girdiği için `$json` artık HTTP cevabı, Telegram mesajı değil → metni doğrudan Trigger'dan almak gerekti. (**Araya düğüm ekleyince sonraki düğümün `$json`'u değişir** — n8n'de sık yapılan hata) |
| Basic LLM Chain - Sohbet | sistem promptuna: "itiraz ediyorsa (ör. 'bu mısır değil') SADECE bitki adını yazmasını iste" | "Bu mısır değil" bitki adı içermiyor → sohbete gider → Claude artık doğru yönlendiriyor |
| Rapor JSON'unu Ayrıştır (`n8n/rapor_ayristir_kod.js`) | Güven düşük VE bitki adı verilmemişse: "💬 Bitki yanlış mı? Bitkinin adını yazın (ör. domates), aynı fotoğrafı yeniden değerlendireyim." | Çiftçiye çıkış yolu gösterilsin |

## 7) Test — nasıl doğruladık?

FastAPI tarafı (sahte `chat_id=TEST1`, `file_id=FILE_ABC` ile; test sonrası kayıt silindi):

| Girdi | Sonuç | Beklenen mi? |
|---|---|---|
| Fotoğraf yokken "Domates" | `yeniden: false` (son 30 dk'da fotoğraf yok) | ✅ |
| Fotoğraftan sonra "Domates" | `yeniden: true, bitki: Domates` | ✅ |
| "bu mısır değil" | `yeniden: false` (bitki adı yok) → sohbet | ✅ |
| "Bu mısır değil, domates" | `yeniden: true, bitki: Domates` | ✅ |
| "domates nasıl sulanır, her gün mü sulamalıyım acaba" | `yeniden: false` (uzun mesaj) | ✅ |
| Başka bir sohbetten "Domates" | `yeniden: false` (o sohbette fotoğraf yok) | ✅ |
| Geç yanıklık fotoğrafı + bitki "Domates" | Geç Yanıklık | ✅ |

n8n dosyası: 51 ifadenin (`{{ ... }}`) ve 2 kod düğümünün sözdizimi otomatik kontrol edildi, hata yok.
**Kalan:** içe aktarıp Telegram'dan gerçek uçtan uca test (açıklamasız fotoğraf → "Domates").

**Canlıya alma adımları (n8n'de, elle):** Workflows → Import from File → `n8n/leadleaf_tam_akis.json`
("tam v2" olarak gelir) → eski "(tam)" akışını **yayından kaldır** (aynı bota bağlı iki aktif akış
çakışır) → v2'yi **Publish** → Telegram'dan test.

## 8) Aynı gün çıkan iki altyapı konusu

**a) "certificate signed by unknown authority" hatası (ağ sorunu, kod değil).** Servisler açıldığında
ngrok, n8n ve hatta google.com'a HTTPS bağlantısı kurulamadı. Sebep: bağlı olunan ağ (FATİH/okul ağı)
şifreli trafiği araya girip denetliyor (**SSL denetimi**): araya kendi sertifikasını koyuyor, programlar
bu sertifikayı tanımadığı için bağlantıyı güvenlik gereği reddediyor. Sertifika doğrulamasını kapatmak
bir "çözüm" değil (güvenlik açığı olur). Doğru çözüm: başka ağa geçmek (ör. telefon hotspot'u). Ağ
değişince ngrok kendiliğinden bağlandı; **n8n yeniden başlatıldı**, çünkü Telegram'a "mesajları şu
adrese gönder" kaydını (webhook) **açılışta** yapıyor — açılışta internet yoksa bu kayıt olmuyor.
**Sunum günü için:** okul ağında Telegram demosu çalışmaz → hotspot hazır olsun.

**b) "n8n Cloud ücretli, biz nasıl kullanıyoruz?"** n8n Cloud'u kullanmıyoruz (14 günlük deneme,
sonra ücretli — 20 Eylül'de bu yüzden vazgeçildi). n8n **açık kaynak** bir program; kendi
bilgisayarımızda ücretsiz çalıştırıyoruz (`npx n8n start` → `http://localhost:5678`; buna
**self-hosted / kendi sunucunda barındırma** denir). Sorun: Telegram'ın evdeki bilgisayarımıza
internetten ulaşması gerekiyor. Bunu **ngrok** çözüyor: ücretsiz planda verilen kalıcı adrese
(`enclose-afterglow-sappiness.ngrok-free.dev`) gelen istekleri bilgisayarımızdaki 5678 portuna
iletiyor (**tünel**). Bedeli: **bilgisayar kapanınca / internet gidince bot da durur.** Gerçek bir
ürün için sonraki adım: aylık düşük ücretli bir sunucu (VPS) ya da n8n Cloud.

Servisleri açma sırası (PowerShell):
```
.venv\Scripts\python -m uvicorn inference.app:app --host 127.0.0.1 --port 8000
ngrok http --url=enclose-afterglow-sappiness.ngrok-free.dev 5678
$env:NODE_OPTIONS="--dns-result-order=ipv4first"; $env:WEBHOOK_URL="https://enclose-afterglow-sappiness.ngrok-free.dev/"; npx n8n start
.venv\Scripts\python -m streamlit run ui/sunum.py
```
(Her biri ayrı bir terminal penceresinde; hepsi açık kalmalı.)

## 9) Jüri sorarsa — kısa cevaplar

- **"Model neden yanlış bildi?"** — Laboratuvar fotoğraflarıyla eğitildi, tarla fotoğrafı farklı bir
  dağılım. Ama güveni %29'du ve sistem bunu "emin değilim, uzmana danışın" olarak işaretledi —
  güvenlik mekanizması çalıştı. Eksik olan kullanıcıya düzeltme yolu sunmaktı, onu ekledik.
- **"Neden LLM'e hafıza vermediniz?"** — Teşhisi CNN yapıyor; LLM fotoğrafı görmüyor. Doğru sonuç için
  CNN'i aynı fotoğrafla, bitki filtresiyle yeniden çalıştırmak gerekiyordu. Hafıza bunu sağlamazdı.
- **"Kullanıcı fotoğraflarını saklıyor musunuz?"** — Hayır. Sadece Telegram'daki kimliğini, 30 dakika,
  git'e girmeyen bir dosyada tutuyoruz.
- **"'Bu mısır değil' yazınca ne oluyor?"** — Olumsuzluk algılanıyor, mısır seçilmiyor; bot bitkinin
  adını yazmasını istiyor.
- **"Bitki adı verilince olasılıklar neden yeniden normalize edilmiyor?"** — O bitkiye düşen ham
  olasılık düşükse bu "fotoğraf bu bitkinin bilinen hastalıklarına benzemiyor" demek; normalize
  edersek bu uyarıyı kaybeder, yapay bir kesinlik üretirdik (bkz. `BITKI_UYUM_ESIGI`).
- **"Hatayı nasıl buldunuz?"** — Tahmin etmedik; n8n'in çalıştırma kayıtlarında her düğümün girdisini
  ve çıktısını okuduk. Tüm çalıştırmalar "başarılı" görünüyordu; hata mantıktaydı.
