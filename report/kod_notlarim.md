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

---

*(Devamı geldikçe buraya eklenecek: alt küme oluşturma, veri seti okuma, model kurulumu — MobileNetV2 ve
transfer learning, 2 aşamalı eğitim, değerlendirme/confusion matrix, kaydetme.)*
