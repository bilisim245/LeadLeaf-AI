# =====================================================================
#  BİTKİ HASTALIĞI TESPİTİ — CNN Transfer Learning (Keras)
#  Kaggle GPU (P100)
# =====================================================================
#
#  NASIL ÇALIŞTIRILIR:
#   1. kaggle.com > Create > New Notebook
#   2. Sağ panel > Session options > Accelerator = GPU
#   3. Sağ panel > Add Input > ara: "New Plant Diseases Dataset" (vipoooool) > Add
#   4. Bu dosyanın tamamını yapıştır ve Run All.
#      (İstersen "# %%" işaretlerinden bölerek de çalıştırabilirsin.)
#
#  ÇIKTI (sağ panel > Output > /kaggle/working):
#   - model.keras            -> eğitilmiş model (indir, proje "model/" içine koy)
#   - class_names.json       -> sınıf isimleri (indir, "model/" içine koy)
#   - metrics.txt            -> doğruluk + sınıf bazlı rapor (rapora koy)
#   - confusion_matrix.png   -> rapora koy
#   - demo_images/           -> demo için örnek yaprak görselleri
# =====================================================================

# %% 1) Importlar + ayarlar
import os, json, glob, random, shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

IMG = 224
BATCH = 32
SEED = 42
MAX_PER_CLASS = None   # hız için sınıf başına görsel sınırı (ör. 500). None = hepsi.
EPOCHS_HEAD = 8        # 1. aşama: sadece üst katman
EPOCHS_FINE = 5        # 2. aşama: ince ayar

# LeadLeaf AI — 10 günlük ZORUNLU kapsam: 38 sınıf değil, sadece domates + 4 yaygın hastalık.
# Sonraki aşamada (TÜBİTAK/TEKNOFEST) None yapıp tüm bitkilere genişletilir.
SELECTED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Septoria_leaf_spot",
]  # None = veri setindeki tüm 38 sınıf

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# %% 2) Veri yolları
CANDIDATES = [
    "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)",
    "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)",
]
DATA_ROOT = next((p for p in CANDIDATES if os.path.isdir(os.path.join(p, "train"))), None)
assert DATA_ROOT, "Veri seti bulunamadi. Sag panelden 'New Plant Diseases Dataset' (vipoooool) ekle."

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
        files = os.listdir(src)
        random.shuffle(files)
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    return hedef_dir


# %% 3) Domates alt kümesi (+ istenirse sınıf başına görsel sınırı)
TRAIN_DIR = _alt_kume_olustur(TRAIN_DIR, "/kaggle/working/_subset/train",
                              SELECTED_CLASSES, MAX_PER_CLASS)
VALID_DIR = _alt_kume_olustur(VALID_DIR_FULL, "/kaggle/working/_subset/valid",
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
with open("/kaggle/working/class_names.json", "w") as f:
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
plt.tight_layout(); plt.savefig("/kaggle/working/ornek_gorseller.png", dpi=120); plt.show()

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

with open("/kaggle/working/metrics.txt", "w") as f:
    f.write(f"model: MobileNetV2 transfer learning\n")
    f.write(f"sinif sayisi: {len(class_names)}\n")
    f.write(f"dogrulama dogrulugu: {val_acc:.4f}\n\n")
    f.write(report)

plt.figure(figsize=(12, 10))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix — Dogrulama")
plt.xlabel("Tahmin"); plt.ylabel("Gercek")
plt.colorbar()
plt.tight_layout(); plt.savefig("/kaggle/working/confusion_matrix.png", dpi=120); plt.show()

# eğitim eğrileri
acc = h1.history["accuracy"] + h2.history["accuracy"]
vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
plt.figure(figsize=(7, 4))
plt.plot(acc, label="egitim"); plt.plot(vacc, label="dogrulama")
plt.xlabel("epoch"); plt.ylabel("dogruluk"); plt.legend(); plt.title("Ogrenme egrisi")
plt.tight_layout(); plt.savefig("/kaggle/working/ogrenme_egrisi.png", dpi=120); plt.show()

# %% 9) Kaydet
model.save("/kaggle/working/model.keras")
print("kaydedildi: model.keras")

# demo için birkaç örnek görsel
os.makedirs("/kaggle/working/demo_images", exist_ok=True)
for cls in random.sample(class_names, min(8, len(class_names))):
    files = glob.glob(os.path.join(VALID_DIR, cls, "*"))
    if files:
        src = random.choice(files)
        shutil.copy(src, f"/kaggle/working/demo_images/{cls}.jpg")
print("demo_images/ hazir.")

print("\nBITTI. Output panelinden indir: model.keras, class_names.json, "
      "metrics.txt, confusion_matrix.png, demo_images/")
