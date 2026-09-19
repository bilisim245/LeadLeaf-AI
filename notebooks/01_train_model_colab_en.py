# =====================================================================
#  PLANT DISEASE DETECTION — 3-Model Comparative Transfer Learning
#  TÜBİTAK research track — Google Colab (free GPU, zero setup)
#
#  This is an English-identifier / English-comment MIRROR of
#  `01_train_model_colab.py` (the Turkish original, kept as the canonical
#  version for the local team). Same logic, same constants, same output
#  filenames (so ui/app.py and inference/app.py keep working no matter
#  which of the two scripts produced the model files) — only variable/
#  function names and comments are in English, for an international
#  audience (GitHub README, TÜBİTAK 2204 write-up, code review by
#  non-Turkish speakers).
# =====================================================================
#
#  WHAT THIS VERSION DOES:
#   1. Splits the data ONCE, per class, in a STRATIFIED way into
#      train/valid/test (70/15/15) and records which file went where in
#      split_manifest.json (reproducibility + leakage audit).
#   2. Trains MobileNetV2, MobileNetV3Small, EfficientNetB0 SEPARATELY on
#      the SAME split, under the SAME augmentation/epoch/early-stopping
#      conditions.
#   3. Each model uses its OWN correct preprocess_input (see "DOUBLE
#      NORMALIZATION" note below — the three behave differently).
#   4. For each model, computes accuracy, macro precision/recall/F1/AUC,
#      model size (MB) and average per-image inference time (ms) ONLY on
#      the held-out test set; produces a confusion matrix + learning
#      curve for each.
#   5. Saves all three models as .keras, produces class_names.json +
#      split_manifest.json + model_comparison.csv, zips everything into
#      leadleaf_tubitak_models.zip and downloads it.
#
#  DOUBLE NORMALIZATION NOTE (important, an easy mistake to miss):
#   Keras's MobileNetV2 preprocess_input REALLY rescales pixels to
#   [-1, 1] — you must call it manually. But MobileNetV3 and EfficientNet
#   already have a Rescaling/Normalization layer BUILT INTO the model
#   graph itself; that's why their own preprocess_input functions are, in
#   the official Keras source, pass-throughs (they return the input
#   unchanged). The code below calls each model's OWN preprocess_input —
#   for MobileNetV3/EfficientNet this is already a no-op, so there is no
#   risk of "double normalization" (e.g. rescaling to -1..1 and then
#   feeding that into the model's own internal Rescaling layer again).
#
#  HOW TO RUN:
#   1. https://colab.research.google.com/ -> New notebook
#   2. Runtime > Change runtime type > Hardware accelerator = GPU (T4) > Save
#   3. Download a Kaggle API token: kaggle.com > profile > Settings >
#      API > "Create Legacy Token" -> kaggle.json downloads (ONLY needed
#      to fetch the dataset; training itself runs on Colab's own GPU).
#   4. Paste this whole file into Colab (split it into multiple cells at
#      the "# %%" markers if you like) and Run All.
#   5. Three models train back to back, so total time is ~3x the old
#      single-model version (roughly 30-60 min on a T4, depends on data size).
#
#  OUTPUT (left panel > Files > /content/outputs, all inside the zip too):
#   - model_MobileNetV2.keras, model_MobileNetV3Small.keras, model_EfficientNetB0.keras
#   - model.keras + model_meta.json  -> the AUTOMATICALLY SELECTED best model
#                                        of the three (the production /predict
#                                        endpoint expects a single model;
#                                        model_meta.json records which
#                                        architecture it is so inference/app.py
#                                        can pick the CORRECT preprocess_input)
#   - class_names.json            -> class names (all 3 models share these)
#   - split_manifest.json         -> which file went into train/valid/test + seed/ratios
#   - model_comparison.csv        -> comparison table across the 3 models
#   - model_karsilastirma_dogruluk.png -> accuracy/macro-F1 bar chart across the 3 models
#   - confusion_matrix_<model>.png, ogrenme_egrisi_<model>.png  (per model)
#   - demo_images/                -> sample leaf images for the Streamlit demo
#
#  NOTE ON OUTPUT FILE NAMES: these stay in Turkish/as in the original
#  script ON PURPOSE — ui/app.py and inference/app.py already look for
#  these exact filenames, and either script (Turkish or English) can
#  produce them interchangeably.
#
#  Once the zip is extracted, files are split into TWO destinations:
#  model.keras + model_meta.json + class_names.json + demo_images/ -> model/
#  (production); the remaining 3 models + split_manifest.json +
#  model_comparison.csv -> model/tubitak/ (Streamlit comparison tab).
#  The last cell prints this reminder again.
#
#  Note: the Colab session wipes /content when it closes —
#  leadleaf_tubitak_models.zip downloads automatically, grab it right away.
# =====================================================================

# %% 0) Download the Kaggle dataset — via kagglehub (account is ONLY for the data)
from google.colab import files
import os

if not os.path.exists("/root/.kaggle/kaggle.json"):
    print("Upload kaggle.json (Kaggle > Settings > API > Create Legacy Token):")
    uploaded = files.upload()  # pick kaggle.json
    os.makedirs("/root/.kaggle", exist_ok=True)
    for fname in uploaded:
        os.rename(fname, "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
    import json as _json
    with open("/root/.kaggle/kaggle.json") as _f:
        _cred = _json.load(_f)
    os.environ["KAGGLE_USERNAME"] = _cred["username"]
    os.environ["KAGGLE_KEY"] = _cred["key"]

os.system("pip -q install kagglehub")
import kagglehub

DATASET_PATH = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
print("Dataset downloaded to:", DATASET_PATH)
print("Contents:", os.listdir(DATASET_PATH))

# %% 1) Imports + shared settings
import csv
import glob
import json
import random
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet, mobilenet_v2, mobilenet_v3
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
assert tf.config.list_physical_devices("GPU"), (
    "No GPU found! Go to Runtime > Change runtime type > GPU (T4) and try again."
)

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
MAX_PER_CLASS = None
EPOCHS_HEAD = 8
EPOCHS_FINE = 5
UNFREEZE_RATIO = 0.25  # last 25% of the backbone is opened for fine-tuning — SAME ratio for all 3 architectures
OUT_DIR = "/content/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

SPLIT_RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}

# LeadLeaf AI — 10-day REQUIRED scope: not all 38 classes, only tomato + 4 common diseases.
SELECTED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Septoria_leaf_spot",
]  # None = all 38 classes in the dataset

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# The 3 architectures being compared — each carries its own correct preprocess_input.
# (mobilenet_v3.preprocess_input and efficientnet.preprocess_input are KNOWINGLY
#  pass-throughs: see the "DOUBLE NORMALIZATION NOTE" above.)
MODEL_DEFINITIONS = [
    {
        "name": "MobileNetV2",
        "factory": keras.applications.MobileNetV2,
        "preprocess": mobilenet_v2.preprocess_input,
    },
    {
        "name": "MobileNetV3Small",
        "factory": keras.applications.MobileNetV3Small,
        "preprocess": mobilenet_v3.preprocess_input,
    },
    {
        "name": "EfficientNetB0",
        "factory": keras.applications.EfficientNetB0,
        "preprocess": efficientnet.preprocess_input,
    },
]

# %% 2) Data path — RAW PlantVillage (color images). The inner folder name can
# vary by mirror ("color", "PlantVillage/color", etc.) — so we search dynamically.
CANDIDATES = [
    os.path.join(DATASET_PATH, "color"),
    os.path.join(DATASET_PATH, "PlantVillage", "color"),
    os.path.join(DATASET_PATH, "plantvillage dataset", "color"),
]
RAW_DIR = next((p for p in CANDIDATES if os.path.isdir(p)), None)
if RAW_DIR is None:
    found = glob.glob(os.path.join(DATASET_PATH, "**", "color"), recursive=True)
    RAW_DIR = found[0] if found else None
assert RAW_DIR, (
    f"Could not auto-locate a 'color' folder. DATASET_PATH contents: {os.listdir(DATASET_PATH)}\n"
    f"Check the list above and add the correct path to CANDIDATES manually."
)
print("Raw data (color):", RAW_DIR)
available_classes = sorted(os.listdir(RAW_DIR))
print(f"Total number of classes in the dataset: {len(available_classes)}")

missing_classes = [c for c in (SELECTED_CLASSES or []) if c not in available_classes]
if missing_classes:
    print(f"\nWARNING: these classes were not found: {missing_classes}")
    print("Actual tomato-related folder names:")
    for c in available_classes:
        if "tomato" in c.lower():
            print(" -", c)
    raise AssertionError("Fix SELECTED_CLASSES to match the real folder names listed above.")

# Data analysis — BEFORE the split, raw per-class image counts (imbalance check)
print("\nImage counts for the 5 selected classes (before split, raw data):")
total_images = 0
for cls in (SELECTED_CLASSES or available_classes):
    n = len(os.listdir(os.path.join(RAW_DIR, cls)))
    total_images += n
    print(f"  {cls:32s} {n:5d}")
print(f"  {'TOTAL':32s} {total_images:5d}")

# %% 3) ONE-TIME stratified train/valid/test split (70/15/15) + manifest
SPLIT_DIR = "/content/_split"


def _stratified_three_way_split(source_dir, target_root, classes, ratios, max_per_class, seed):
    """Shuffles each class's own file list ONCE and splits it into
    train/valid/test (stratified: each class is split at its own ratio, so
    small classes don't vanish from the test set). The same image can never
    end up in two splits at once — leakage is structurally impossible. The
    result is stored in split_manifest.json in a readable form (for
    reproducibility + auditing)."""
    targets = {k: os.path.join(target_root, k) for k in ("train", "valid", "test")}
    if all(os.path.isdir(d) for d in targets.values()):
        print("Split folders already exist, not regenerating:", target_root)
        return targets, None

    rng = random.Random(seed)
    source_classes = classes if classes else os.listdir(source_dir)
    manifest_files = {"train": {}, "valid": {}, "test": {}}

    for cls in source_classes:
        src = os.path.join(source_dir, cls)
        if not os.path.isdir(src):
            print(f"WARNING: class folder missing, skipping: {cls}")
            continue
        files_list = os.listdir(src)
        rng.shuffle(files_list)
        if max_per_class:
            files_list = files_list[:max_per_class]

        n = len(files_list)
        n_test = max(1, int(round(n * ratios["test"])))
        n_valid = max(1, int(round(n * ratios["valid"])))
        n_train = n - n_test - n_valid
        assert n_train > 0, f"{cls}: not enough images for train (n={n})"

        splits = {
            "test": files_list[:n_test],
            "valid": files_list[n_test:n_test + n_valid],
            "train": files_list[n_test + n_valid:],
        }
        for split_name, split_files in splits.items():
            manifest_files[split_name][cls] = split_files
            dst = os.path.join(targets[split_name], cls)
            os.makedirs(dst, exist_ok=True)
            for f in split_files:
                shutil.copy(os.path.join(src, f), os.path.join(dst, f))
        print(f"  {cls}: {len(splits['train'])} train, {len(splits['valid'])} valid, "
              f"{len(splits['test'])} test")

    manifest = {
        "data_source": "abdallahalidev/plantvillage-dataset (kagglehub)",
        "seed": seed,
        "ratios": ratios,
        "classes": source_classes,
        "counts": {
            split_name: {cls: len(files_list) for cls, files_list in split_dict.items()}
            for split_name, split_dict in manifest_files.items()
        },
        "files": manifest_files,
    }
    return targets, manifest


SPLIT_DIRS, MANIFEST = _stratified_three_way_split(
    RAW_DIR, SPLIT_DIR, SELECTED_CLASSES, SPLIT_RATIOS, MAX_PER_CLASS, SEED,
)
if MANIFEST is not None:
    with open(f"{OUT_DIR}/split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(MANIFEST, f, ensure_ascii=False, indent=2)
    print("split_manifest.json saved.")
else:
    print("WARNING: split folders already existed, split_manifest.json was NOT regenerated this run.")

TRAIN_DIR, VALID_DIR, TEST_DIR = SPLIT_DIRS["train"], SPLIT_DIRS["valid"], SPLIT_DIRS["test"]

# %% 4) tf.data pipelines — RAW pixels (0-255, float32); normalization happens
# INSIDE THE MODEL (each model applies its own preprocess_input), so there is
# no risk of accidentally adding a second normalization at the dataset level
# ("double normalization" bug).
def load_dataset(directory, batch_size, shuffle):
    return keras.utils.image_dataset_from_directory(
        directory, image_size=(IMG_SIZE, IMG_SIZE), batch_size=batch_size, seed=SEED, shuffle=shuffle,
    )


train_ds_raw = load_dataset(TRAIN_DIR, BATCH_SIZE, shuffle=True)
val_ds_raw = load_dataset(VALID_DIR, BATCH_SIZE, shuffle=False)
# Test set uses batch_size=1: lets us do per-image sklearn predictions AND
# measure "average per-image inference time" in the same loop.
test_ds_raw = load_dataset(TEST_DIR, 1, shuffle=False)

class_names = train_ds_raw.class_names
print(len(class_names), "classes:", class_names)
with open(f"{OUT_DIR}/class_names.json", "w") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds_raw = train_ds_raw.prefetch(AUTOTUNE)
val_ds_raw = val_ds_raw.prefetch(AUTOTUNE)
# test_ds_raw has NO prefetch — deliberately, so prefetch's hidden parallelism
# doesn't distort the inference-time measurement.

y_true_test = np.concatenate([y.numpy() for _, y in test_ds_raw])

# %% 5) Shared data augmentation layer — used with the SAME parameters for all 3 models
def build_augmentation_layer():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augment")


def build_early_stopping():
    return keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True,
    )


def build_model(factory, preprocess_fn, num_classes):
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = build_augmentation_layer()(inputs)
    x = preprocess_fn(x)  # each model applies its own correct preprocessing here
    base = factory(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)  # kept low — anything higher can underfit on a small dataset
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs), base


# %% 6) Evaluation — ONLY on the held-out test set, same protocol for every model
def evaluate_model(model, test_ds, y_true):
    # batched prediction for the sklearn metrics
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = y_prob.argmax(axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    # macro AUC (one-vs-rest) — accuracy alone can be misleading on imbalanced
    # classes, so we also measure how well the model SEPARATES the classes
    try:
        macro_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = float("nan")  # if the test set is missing a class entirely
    cm = confusion_matrix(y_true, y_pred)

    # Average per-image inference time — one at a time, close to the real
    # serving scenario (FastAPI also takes one image at a time). The first
    # call (warmup / graph tracing) is excluded so it doesn't skew the measurement.
    inference_times_ms = []
    for i, (x, _) in enumerate(test_ds):
        t0 = time.perf_counter()
        model.predict(x, verbose=0)
        t1 = time.perf_counter()
        if i > 0:  # first call is warmup, excluded from the measurement
            inference_times_ms.append((t1 - t0) * 1000)
    avg_ms = float(np.mean(inference_times_ms)) if inference_times_ms else float("nan")

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "macro_auc": macro_auc,
        "confusion_matrix": cm,
        "avg_inference_ms": avg_ms,
    }


def save_confusion_matrix_plot(cm, name, class_names):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix — {name} (held-out test set)")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right", fontsize=7)
    plt.yticks(range(len(class_names)), class_names, fontsize=7)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix_{name}.png", dpi=120)
    plt.close()


def parameter_summary(model, label):
    """Explainability helper: how many parameters are trainable right now vs.
    frozen. Counterpart of the same-named function in the reference lecture
    notebook (cnn2_ders.ipynb, 11.3) — shows concretely how much is open at
    each stage."""
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    frozen = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    print(f"  {label:38s} trainable: {trainable:>10,}   frozen: {frozen:>10,}")
    return trainable, frozen


def save_learning_curve_plot(h1, h2, name):
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]
    boundary = len(h1.history["accuracy"]) - 0.5

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.4))
    left.plot(acc, label="train"); left.plot(val_acc, label="validation")
    left.axvline(boundary, color="gray", linestyle="--", label="fine-tuning starts")
    left.set_xlabel("epoch"); left.set_ylabel("accuracy"); left.legend()
    left.set_title(f"{name} — Accuracy")

    right.plot(loss, label="train"); right.plot(val_loss, label="validation")
    right.axvline(boundary, color="gray", linestyle="--", label="fine-tuning starts")
    right.set_xlabel("epoch"); right.set_ylabel("loss"); right.legend()
    right.set_title(f"{name} — Loss")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ogrenme_egrisi_{name}.png", dpi=120)
    plt.close()


# %% 7) Train + evaluate all three models in sequence, under identical conditions
comparison_rows = []

for definition in MODEL_DEFINITIONS:
    name, factory, preprocess_fn = definition["name"], definition["factory"], definition["preprocess"]
    print(f"\n{'=' * 60}\nTraining {name}\n{'=' * 60}")

    model, base = build_model(factory, preprocess_fn, len(class_names))
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    print(f"  Backbone total layers: {len(base.layers)}   total parameters: {base.count_params():,}")
    trainable_1, frozen_1 = parameter_summary(model, "Stage 1 (backbone fully frozen)")

    # Stage 1 — train the head only (backbone frozen)
    h1 = model.fit(train_ds_raw, validation_data=val_ds_raw, epochs=EPOCHS_HEAD,
                    callbacks=[build_early_stopping()])

    # Stage 2 — fine-tuning: open the last UNFREEZE_RATIO of the backbone (SAME ratio for all 3 architectures)
    base.trainable = True
    n_freeze = int(len(base.layers) * (1 - UNFREEZE_RATIO))
    for layer in base.layers[:n_freeze]:
        layer.trainable = False
    print(f"{name}: opened the last {len(base.layers) - n_freeze} of {len(base.layers)} layers for fine-tuning.")

    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    trainable_2, frozen_2 = parameter_summary(model, f"Stage 2 (last {int(UNFREEZE_RATIO * 100)}% open)")

    h2 = model.fit(train_ds_raw, validation_data=val_ds_raw, epochs=EPOCHS_FINE,
                    callbacks=[build_early_stopping()])

    result = evaluate_model(model, test_ds_raw, y_true_test)
    print(f"{name} — test accuracy: {result['accuracy']:.4f}, macro F1: {result['macro_f1']:.4f}, "
          f"macro AUC: {result['macro_auc']:.4f}, avg inference: {result['avg_inference_ms']:.1f} ms/image")

    model_file = f"{OUT_DIR}/model_{name}.keras"
    model.save(model_file)
    model_size_mb = round(os.path.getsize(model_file) / (1024 * 1024), 2)

    save_confusion_matrix_plot(result["confusion_matrix"], name, class_names)
    save_learning_curve_plot(h1, h2, name)

    comparison_rows.append({
        "model": name,
        "dogruluk": round(result["accuracy"], 4),
        "macro_precision": round(result["macro_precision"], 4),
        "macro_recall": round(result["macro_recall"], 4),
        "macro_f1": round(result["macro_f1"], 4),
        "macro_auc": round(result["macro_auc"], 4),
        "model_boyutu_mb": model_size_mb,
        "ort_inference_ms": round(result["avg_inference_ms"], 2),
        # explainability: how many parameters were actually updated at each stage
        "egitilebilir_parametre_1_asama": trainable_1,
        "donuk_parametre_1_asama": frozen_1,
        "egitilebilir_parametre_2_asama": trainable_2,
        "donuk_parametre_2_asama": frozen_2,
    })

    # memory cleanup — so GPU RAM doesn't balloon while training 3 models back to back
    del model, base
    keras.backend.clear_session()

# %% 8) model_comparison.csv
# NOTE: column names are kept identical to the Turkish script on purpose —
# ui/app.py reads this CSV by these exact column names regardless of which
# training script produced it.
with open(f"{OUT_DIR}/model_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
    writer.writeheader()
    writer.writerows(comparison_rows)

print("\nmodel_comparison.csv:")
for row in comparison_rows:
    print(row)

# %% 8a) Accuracy comparison chart (3 models side by side) — for Streamlit and the report
names = [s["model"] for s in comparison_rows]
accuracies = [s["dogruluk"] for s in comparison_rows]
f1_scores = [s["macro_f1"] for s in comparison_rows]

x = np.arange(len(names))
width = 0.35
plt.figure(figsize=(7, 5))
plt.bar(x - width / 2, accuracies, width, label="accuracy", color="#346cb0")
plt.bar(x + width / 2, f1_scores, width, label="macro F1", color="#4e8a6b")
for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
    plt.text(i - width / 2, acc + 0.01, f"{acc:.3f}", ha="center", fontsize=9)
    plt.text(i + width / 2, f1 + 0.01, f"{f1:.3f}", ha="center", fontsize=9)
plt.xticks(x, names)
plt.ylabel("score (held-out test set)")
plt.ylim(0, 1.08)
plt.title("Model comparison — accuracy and macro F1")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_karsilastirma_dogruluk.png", dpi=120)
plt.close()
print("model_karsilastirma_dogruluk.png saved.")

# %% 8b) Pick the best model for the production (single-model) pipeline and copy it to model.keras
# inference/app.py's /predict endpoint (the one n8n/Telegram talks to) expects a
# SINGLE model. Whichever architecture wins doesn't have to be MobileNetV2 — so
# we also write which one it is to model_meta.json; inference/app.py reads that
# to pick the CORRECT preprocess_input. Skipping this would apply the wrong
# preprocessing to the wrong model in production (exactly the "double/wrong
# normalization" bug we're trying to avoid — just in production instead of training).
best_model = max(comparison_rows, key=lambda s: s["dogruluk"])
print(f"\nModel selected for the production (single-model) pipeline: {best_model['model']} "
      f"(accuracy={best_model['dogruluk']}, macro_f1={best_model['macro_f1']})")
shutil.copy(f"{OUT_DIR}/model_{best_model['model']}.keras", f"{OUT_DIR}/model.keras")
with open(f"{OUT_DIR}/model_meta.json", "w", encoding="utf-8") as f:
    json.dump({"mimari": best_model["model"]}, f, ensure_ascii=False, indent=2)

# %% 9) Demo images (for the Streamlit comparison tab) — samples from the test set
os.makedirs(f"{OUT_DIR}/demo_images", exist_ok=True)
for cls in random.sample(class_names, min(8, len(class_names))):
    files_list = glob.glob(os.path.join(TEST_DIR, cls, "*"))
    if files_list:
        src = random.choice(files_list)
        shutil.copy(src, f"{OUT_DIR}/demo_images/{cls}.jpg")
print("demo_images/ ready.")

# %% 10) Zip everything and download (the Colab session wipes /content when it closes!)
shutil.make_archive("/content/leadleaf_tubitak_models", "zip", OUT_DIR)
files.download("/content/leadleaf_tubitak_models.zip")
print(f"""
DONE. leadleaf_tubitak_models.zip downloaded -> open it and distribute its contents to TWO places:

  1) model/  (PRODUCTION — the n8n/Telegram flow and the Streamlit "Analysis" tab use this)
       - model.keras        (auto-selected best model: {best_model['model']})
       - model_meta.json    (inference/app.py reads this to pick the correct preprocessing)
       - class_names.json
       - demo_images/

  2) model/tubitak/  (Streamlit "Model Comparison" tab)
       - model_MobileNetV2.keras
       - model_MobileNetV3Small.keras
       - model_EfficientNetB0.keras
       - class_names.json
       - split_manifest.json
       - model_comparison.csv
       - model_karsilastirma_dogruluk.png   (shown automatically in Streamlit)

class_names.json can be copied to both, unchanged (all 3 models share the same classes).
""")
