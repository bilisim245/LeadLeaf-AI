# =====================================================================
#  BİTKİ HASTALIĞI TESPİTİ — CNN Transfer Learning (Keras)
#  Google Colab (ücretsiz GPU, sıfır kurulum)
# =====================================================================
#
#  NASIL ÇALIŞTIRILIR:
#   1. https://colab.research.google.com/ -> Yeni not defteri
#   2. Üst menü > Çalışma zamanı (Runtime) > Çalışma zamanı türünü değiştir
#      > Donanım hızlandırıcı = GPU (T4) seç > Kaydet
#   3. Kaggle hesabından API token indir: kaggle.com > sağ üstte profil >
#      Settings > API > "Create New Token" -> kaggle.json iner.
#      (Kaggle hesabı SADECE veri setini indirmek için gerekiyor, eğitim
#      burada Colab'ın kendi GPU'sunda çalışacak.)
#   4. Bu dosyanın tamamını Colab'a yapıştır (birden fazla hücreye
#      "# %%" işaretlerinden bölerek de yapıştırabilirsin) ve Run All.
#      İlk çalıştırmada kaggle.json yüklemen istenecek (dosya seçme kutusu).
#
#  ÇIKTI (sol panel > Dosyalar > /content/outputs):
#   - model.keras            -> eğitilmiş model (indir, proje "model/" içine koy)
#   - class_names.json       -> sınıf isimleri (indir, "model/" içine koy)
#   - metrics.txt            -> doğruluk + sınıf bazlı rapor (rapora koy)
#   - confusion_matrix.png   -> rapora koy
#   - demo_images/           -> demo için örnek yaprak görselleri
#
#  Not: Colab oturumu kapanınca /content silinir — dosyaları İNDİRMEYİ
#  unutma (son hücre otomatik zip indirtir).
# =====================================================================

# %% 0) Kaggle veri setini indir (hesap SADECE veri için, eğitim Colab GPU'sunda)
from google.colab import files
import os

if not os.path.exists("/root/.kaggle/kaggle.json"):
    print("kaggle.json yükle (Kaggle > Settings > API > Create New Token):")
    uploaded = files.upload()  # kaggle.json seç
    os.makedirs("/root/.kaggle", exist_ok=True)
    for fname in uploaded:
        os.rename(fname, "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 0o600)

os.system("pip -q install kaggle")
os.makedirs("/content/data", exist_ok=True)
os.system(
    "kaggle datasets download -d vipoooool/new-plant-diseases-dataset "
    "-p /content/data --unzip"
)
print("Veri indirildi: /content/data")

# %% 1) Importlar + ayarlar
import json, glob, random, shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
assert tf.config.list_physical_devices("GPU"), (
    "GPU görünmüyor! Runtime > Change runtime type > GPU (T4) seçip tekrar dene."
)

IMG = 224
BATCH = 32
SEED = 42
MAX_PER_CLASS = None
EPOCHS_HEAD = 8
EPOCHS_FINE = 5
OUT_DIR = "/content/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# LeadLeaf AI — 10 günlük ZORUNLU kapsam: 38 sınıf değil, sadece domates + 4 yaygın hastalık.
SELECTED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Septoria_leaf_spot",
]  # None = veri setindeki tüm 38 sınıf

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# %% 2) Veri yolları (Kaggle'daki /kaggle/input yerine Colab'da /content/data)
CANDIDATES = [
    "/content/data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)",
    "/content/data/New Plant Diseases Dataset(Augmented)",
]
DATA_ROOT = next((p for p in CANDIDATES if os.path.isdir(os.path.join(p, "train"))), None)
assert DATA_ROOT, "Veri seti bulunamadi. Yukaridaki indirme hucresini kontrol et."

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VALID_DIR_FULL = os.path.join(DATA_ROOT, "valid")
print("train (tam):", TRAIN_DIR)
print("valid (tam):", VALID_DIR_FULL)
print("veri setindeki toplam sinif sayisi:", len(os.listdir(TRAIN_DIR)))


def _alt_kume_olustur(kaynak_dir: str, hedef_dir: str, siniflar, max_per_class):
    """SELECTED_CLASSES ve/veya MAX_PER_CLASS'a gore klasoru kucult."""
    if not siniflar and not max_per_class:
        return kaynak_dir
    if os.path.isdir(hedef_dir):
        return hedef_dir
    kaynak_siniflar = siniflar if siniflar else os.listdir(kaynak_dir)
    for cls in kaynak_siniflar:
        src = os.path.join(kaynak_dir, cls)
        if not os.path.isdir(src):
            print(f"UYARI: sinif klasoru yok, atlaniyor: {cls}")
            continue
        dst = os.path.join(hedef_dir, cls)
        os.makedirs(dst, exist_ok=True)
        files_ = os.listdir(src)
        random.shuffle(files_)
        if max_per_class:
            files_ = files_[:max_per_class]
        for f in files_:
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    return hedef_dir


# %% 3) Domates alt kümesi (+ istenirse sınıf başına görsel sınırı)
TRAIN_DIR = _alt_kume_olustur(TRAIN_DIR, "/content/_subset/train",
                              SELECTED_CLASSES, MAX_PER_CLASS)
VALID_DIR = _alt_kume_olustur(VALID_DIR_FULL, "/content/_subset/valid",
                              SELECTED_CLASSES, None)
print("kullanilan train:", TRAIN_DIR)
print("kullanilan valid:", VALID_DIR)
print("kullanilan sinif sayisi:", len(os.listdir(TRAIN_DIR)))

# %% 4) tf.data setleri
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=(IMG, IMG), batch_size=BATCH, seed=SEED, shuffle=True)
val_ds = keras.utils.image_dataset_from_directory(
    VALID_DIR, image_size=(IMG, IMG), batch_size=BATCH, seed=SEED, shuffle=False)

class_names = train_ds.class_names
print(len(class_names), "sinif")
with open(f"{OUT_DIR}/class_names.json", "w") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# örnek görseller
plt.figure(figsize=(10, 6))
for imgs, lbls in train_ds.take(1):
    for i in range(min(9, imgs.shape[0])):
        plt.subplot(3, 3, i + 1)
        plt.imshow(imgs[i].numpy().astype("uint8"))
        plt.title(class_names[lbls[i]][:22], fontsize=8)
        plt.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/ornek_gorseller.png", dpi=120); plt.show()

# %% 5) Model — MobileNetV2 transfer learning
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="augment")

base = keras.applications.MobileNetV2(
    input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
base.trainable = False

inputs = keras.Input((IMG, IMG, 3))
x = augment(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# %% 6) 1. aşama — üst katmanı eğit
cb = [keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                    patience=3, restore_best_weights=True)]
h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=cb)

# %% 7) 2. aşama — ince ayar (base'in son katmanları)
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, callbacks=cb)

# %% 8) Değerlendirme
from sklearn.metrics import classification_report, confusion_matrix

y_true = np.concatenate([y.numpy() for _, y in val_ds])
y_prob = model.predict(val_ds)
y_pred = y_prob.argmax(axis=1)

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
cm = confusion_matrix(y_true, y_pred)
val_acc = (y_true == y_pred).mean()
print("Dogrulama dogrulugu:", round(float(val_acc), 4))
print(report)

with open(f"{OUT_DIR}/metrics.txt", "w") as f:
    f.write(f"model: MobileNetV2 transfer learning\n")
    f.write(f"sinif sayisi: {len(class_names)}\n")
    f.write(f"dogrulama dogrulugu: {val_acc:.4f}\n\n")
    f.write(report)

plt.figure(figsize=(12, 10))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix — Dogrulama")
plt.xlabel("Tahmin"); plt.ylabel("Gercek")
plt.colorbar()
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=120); plt.show()

# eğitim eğrileri
acc = h1.history["accuracy"] + h2.history["accuracy"]
vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
plt.figure(figsize=(7, 4))
plt.plot(acc, label="egitim"); plt.plot(vacc, label="dogrulama")
plt.xlabel("epoch"); plt.ylabel("dogruluk"); plt.legend(); plt.title("Ogrenme egrisi")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/ogrenme_egrisi.png", dpi=120); plt.show()

# %% 9) Kaydet
model.save(f"{OUT_DIR}/model.keras")
print("kaydedildi: model.keras")

# demo için birkaç örnek görsel
os.makedirs(f"{OUT_DIR}/demo_images", exist_ok=True)
for cls in random.sample(class_names, min(8, len(class_names))):
    files_ = glob.glob(os.path.join(VALID_DIR, cls, "*"))
    if files_:
        src = random.choice(files_)
        shutil.copy(src, f"{OUT_DIR}/demo_images/{cls}.jpg")
print("demo_images/ hazir.")

# %% 10) İndir (Colab oturumu kapanınca /content silinir!)
shutil.make_archive("/content/leadleaf_model_ciktisi", "zip", OUT_DIR)
files.download("/content/leadleaf_model_ciktisi.zip")
print("\nBITTI. leadleaf_model_ciktisi.zip indi -> ac, icindekileri projede "
      "model/ klasorune koy (model.keras, class_names.json, metrics.txt, "
      "confusion_matrix.png, demo_images/).")
