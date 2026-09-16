"""
Inference servisi — FastAPI (Gün 4)

Görev: CNN modelini HTTP ile sunmak. Proje mimarisinde n8n (local veya n8n Cloud)
bu servise HTTP Request node'uyla görsel gönderir, sınıflandırma + güven döner.

DEMO MODU: model/model.keras henüz yoksa (Kaggle eğitimi tamamlanmadıysa) servis
çökmez — rastgele ama makul bir tahmin üretip demo_mode=true ile işaretler. Böylece
n8n / Gradio UI, gerçek model gelmeden ÖNCE de uçtan uca test edilebilir. Kaggle'dan
model.keras + class_names.json inip model/ klasörüne konunca otomatik gerçek moda geçer
(servisi yeniden başlatmak yeterli).

Çalıştırma:
    .venv\\Scripts\\python.exe -m uvicorn inference.app:app --reload --port 8000

Test:
    http://localhost:8000/docs  (Swagger UI'dan görsel yükleyip dene)
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
from collections import Counter
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.image_rag import find_similar, index_var_mi

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "./model/model.keras")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "./model/class_names.json")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
IMG_SIZE = 224

# TÜBİTAK model karşılaştırması (notebooks/01_train_model_colab.py çıktısı) —
# üretim /predict'ten AYRI: n8n/Telegram akışını etkilemez, sadece Streamlit'in
# "Model Karşılaştırma" sekmesi bunu kullanır.
TUBITAK_MODEL_DIR = os.getenv("TUBITAK_MODEL_DIR", "./model/tubitak")
TUBITAK_MODEL_DEFS = [
    {"ad": "MobileNetV2", "dosya": "model_MobileNetV2.keras"},
    {"ad": "MobileNetV3Small", "dosya": "model_MobileNetV3Small.keras"},
    {"ad": "EfficientNetB0", "dosya": "model_EfficientNetB0.keras"},
]

# Sınıf adı -> sade Türkçe adı (rapor ve Gradio arayüzü için)
TR_ADLAR = {
    "Tomato___healthy": "Sağlıklı",
    "Tomato___Early_blight": "Erken Yanıklık (Early Blight)",
    "Tomato___Late_blight": "Geç Yanıklık (Late Blight)",
    "Tomato___Bacterial_spot": "Bakteriyel Leke (Bacterial Spot)",
    "Tomato___Septoria_leaf_spot": "Septoria Yaprak Lekesi",
}
DEMO_CLASS_NAMES = list(TR_ADLAR.keys())

app = FastAPI(title="LeadLeaf AI — Inference Servisi")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP: n8n / Gradio herhangi bir origin'den çağırabilsin
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_embedding_model = None  # gorsel RAG icin - modelin son katmandan onceki (GAP) ciktisi
_class_names: list[str] = []
DEMO_MODE = True

_tubitak_models: dict = {}  # ad -> yüklü keras.Model (sadece dosyası bulunanlar)
_tubitak_class_names: list[str] = []


def _load_model_if_available() -> None:
    """Model dosyaları varsa yükler; yoksa DEMO_MODE'da kalır (servis çökmez)."""
    global _model, _embedding_model, _class_names, DEMO_MODE

    if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH)):
        print(f"[UYARI] Model bulunamadı ({MODEL_PATH}). DEMO MODU aktif — "
              f"Colab eğitimi bitince model/ klasörüne kopyalayıp servisi yeniden başlat.")
        _class_names = DEMO_CLASS_NAMES
        DEMO_MODE = True
        return

    # TensorFlow'u sadece gerçekten gerekince import ediyoruz (demo modda hızlı açılış için)
    import tensorflow as tf

    _model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        _class_names = json.load(f)
    DEMO_MODE = False
    print(f"[OK] Model yüklendi: {MODEL_PATH} ({len(_class_names)} sınıf)")

    # Görsel RAG için: son Dense (softmax) katmanından ÖNCEKİ (GAP) çıktısını veren
    # ikinci bir model — ayrı bir CLIP modeli indirmeden, sınıflandırıcının kendi
    # öğrenilmiş temsilini yeniden kullanıyoruz (bkz. rag/build_image_index.py).
    if index_var_mi():
        try:
            _embedding_model = tf.keras.Model(
                inputs=_model.input, outputs=_model.layers[-2].output
            )
            print("[OK] Görsel RAG embedding modeli hazır.")
        except Exception as e:
            print(f"[UYARI] Görsel RAG embedding modeli kurulamadı: {e}")


def _load_tubitak_models_if_available() -> None:
    """3 karşılaştırma modelini (varsa) yükler. Kısmi eksik olabilir — eksik
    olan her model /predict_compare içinde tek başına DEMO MODU'na düşer,
    servis çökmez (bkz. inference/app.py'nin genel DEMO MODU felsefesi)."""
    global _tubitak_class_names

    class_names_path = os.path.join(TUBITAK_MODEL_DIR, "class_names.json")
    if not os.path.exists(class_names_path):
        _tubitak_class_names = DEMO_CLASS_NAMES
        print(f"[UYARI] TÜBİTAK karşılaştırma modelleri bulunamadı ({TUBITAK_MODEL_DIR}). "
              f"/predict_compare tamamen DEMO MODU'nda çalışacak.")
        return

    import tensorflow as tf

    with open(class_names_path, "r", encoding="utf-8") as f:
        _tubitak_class_names = json.load(f)

    for tanim in TUBITAK_MODEL_DEFS:
        yol = os.path.join(TUBITAK_MODEL_DIR, tanim["dosya"])
        if os.path.exists(yol):
            try:
                _tubitak_models[tanim["ad"]] = tf.keras.models.load_model(yol)
                print(f"[OK] TÜBİTAK modeli yüklendi: {tanim['ad']}")
            except Exception as e:
                print(f"[UYARI] TÜBİTAK modeli yüklenemedi ({tanim['ad']}): {e}")
        else:
            print(f"[UYARI] TÜBİTAK modeli eksik, bu model demo modda kalacak: {tanim['ad']} ({yol})")


@app.on_event("startup")
def _startup() -> None:
    _load_model_if_available()
    _load_tubitak_models_if_available()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "demo_mode": DEMO_MODE,
        "num_classes": len(_class_names),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "tubitak_models_yuklu": sorted(_tubitak_models.keys()),
        "tubitak_models_toplam": len(TUBITAK_MODEL_DEFS),
    }


def _demo_predict() -> tuple[str, np.ndarray]:
    """Rastgele ama tutarlı bir softmax benzeri dağılım üretir (sadece demo modu)."""
    n = len(DEMO_CLASS_NAMES)
    secilen = random.randrange(n)
    probs = np.random.dirichlet(np.ones(n) * 0.6)  # tek sınıfa çekik dağılım
    # seçilen sınıfı en yükseğe zorla (demo daha gerçekçi görünsün diye)
    probs[secilen], probs[probs.argmax()] = max(probs[secilen], probs.max()), probs[secilen]
    probs = probs / probs.sum()
    return DEMO_CLASS_NAMES[int(probs.argmax())], probs


def _gercek_predict(img: Image.Image) -> np.ndarray:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    probs = _model.predict(arr, verbose=0)[0]
    return probs


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Sadece görsel dosyası kabul edilir (image/*).")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Görsel okunamadı: {e}")

    if DEMO_MODE:
        top_class, probs = _demo_predict()
        siniflar = DEMO_CLASS_NAMES
    else:
        probs = _gercek_predict(img)
        siniflar = _class_names
        top_class = siniflar[int(probs.argmax())]

    guven = float(probs.max()) * 100
    sirali = sorted(zip(siniflar, probs.tolist()), key=lambda x: x[1], reverse=True)
    ilk3 = [
        {"sinif": s, "sinif_tr": TR_ADLAR.get(s, s), "olasilik": round(p * 100, 1)}
        for s, p in sirali[:3]
    ]

    # Görsel RAG: modelin GAP-katmanı embedding'iyle en benzer referans görselleri bul
    benzer_gorseller: list[dict] = []
    if not DEMO_MODE and _embedding_model is not None:
        try:
            arr = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype("float32")
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            arr = np.expand_dims(preprocess_input(arr), axis=0)
            emb = _embedding_model.predict(arr, verbose=0)[0]
            benzer_gorseller = find_similar(emb, k=3)
        except Exception as e:
            print(f"[UYARI] Görsel RAG sorgusu başarısız: {e}")

    return {
        "hastalik": top_class,
        "hastalik_tr": TR_ADLAR.get(top_class, top_class),
        "guven": round(guven, 1),
        "ilk3": ilk3,
        "uzmana_yonlendir": guven < CONFIDENCE_THRESHOLD * 100,
        "demo_mode": DEMO_MODE,
        "benzer_gorseller": benzer_gorseller,
    }


def _tubitak_on_isle_fonksiyonu(ad: str):
    """Her mimarinin KENDİ doğru preprocess_input'u — MobileNetV3Small ve
    EfficientNetB0 için bu fonksiyonlar resmi Keras kaynağında pass-through'tur
    (rescaling zaten modelin içinde), MobileNetV2 için piksefli [-1,1]'e ölçekler.
    Notebook'taki "ÇİFT NORMALİZASYON NOTU" ile birebir aynı mantık."""
    from tensorflow.keras.applications import efficientnet, mobilenet_v2, mobilenet_v3

    return {
        "MobileNetV2": mobilenet_v2.preprocess_input,
        "MobileNetV3Small": mobilenet_v3.preprocess_input,
        "EfficientNetB0": efficientnet.preprocess_input,
    }[ad]


def _tubitak_tek_model_tahmin(ad: str, model, img: Image.Image) -> np.ndarray:
    arr = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype("float32")
    arr = _tubitak_on_isle_fonksiyonu(ad)(arr)
    arr = np.expand_dims(arr, axis=0)
    return model.predict(arr, verbose=0)[0]


@app.post("/predict_compare")
async def predict_compare(file: UploadFile = File(...)) -> dict:
    """Aynı fotoğrafı 3 modele (MobileNetV2, MobileNetV3Small, EfficientNetB0)
    birden verir — TÜBİTAK araştırma karşılaştırması için (bkz.
    notebooks/01_train_model_colab.py). Üretim /predict akışını etkilemez.
    Eksik model dosyası varsa o model tek başına DEMO MODU'na düşer."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Sadece görsel dosyası kabul edilir (image/*).")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Görsel okunamadı: {e}")

    sonuclar = []
    for tanim in TUBITAK_MODEL_DEFS:
        ad = tanim["ad"]
        model = _tubitak_models.get(ad)
        if model is not None:
            probs = _tubitak_tek_model_tahmin(ad, model, img)
            siniflar_bu = _tubitak_class_names
            demo = False
        else:
            _, probs = _demo_predict()
            siniflar_bu = DEMO_CLASS_NAMES
            demo = True

        top_idx = int(np.argmax(probs))
        hastalik = siniflar_bu[top_idx]
        guven = float(probs[top_idx]) * 100
        sonuclar.append({
            "model": ad,
            "hastalik": hastalik,
            "hastalik_tr": TR_ADLAR.get(hastalik, hastalik),
            "guven": round(guven, 1),
            "uzmana_yonlendir": guven < CONFIDENCE_THRESHOLD * 100,
            "demo_mode": demo,
        })

    # Model uzlaşması: üçü aynı sınıfı mı buldu (tam), sadece ikisi mi (kısmi),
    # yoksa üçü de farklı mı (yok) — kural tabanlı, ML tahmini DEĞİL.
    tahminler = [s["hastalik"] for s in sonuclar]
    essiz_sayisi = len(set(tahminler))
    if essiz_sayisi == 1:
        uzlasma_durumu = "tam"
    elif essiz_sayisi == len(tahminler):
        uzlasma_durumu = "yok"
    else:
        uzlasma_durumu = "kismi"

    sayac = Counter(tahminler)
    cogunluk_sinif, cogunluk_adet = sayac.most_common(1)[0]
    en_dusuk_guven = min(s["guven"] for s in sonuclar)
    esik_yuzde = CONFIDENCE_THRESHOLD * 100

    if uzlasma_durumu == "tam" and en_dusuk_guven >= esik_yuzde:
        tavsiye = "Üç model de aynı sonuçta hemfikir ve hepsinin güveni eşiğin üzerinde — sonuç güvenilir."
    elif uzlasma_durumu == "yok":
        tavsiye = "Üç model üç farklı sonuç verdi — modeller anlaşamadı, ziraat mühendisine danışın."
    elif en_dusuk_guven < esik_yuzde:
        tavsiye = "En az bir modelin güveni eşiğin altında — teyit için ikinci bir fotoğraf çekin ya da uzmana danışın."
    else:
        tavsiye = "Modeller kısmen hemfikir — çoğunluk sonucu dikkate alınabilir ama teyit önerilir."

    return {
        "sonuclar": sonuclar,
        "uzlasma": {
            "durum": uzlasma_durumu,
            "cogunluk_sinif": cogunluk_sinif,
            "cogunluk_sinif_tr": TR_ADLAR.get(cogunluk_sinif, cogunluk_sinif),
            "cogunluk_adet": cogunluk_adet,
            "toplam_model": len(sonuclar),
            "esik_yuzde": esik_yuzde,
            "en_dusuk_guven": round(en_dusuk_guven, 1),
            "tavsiye": tavsiye,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("inference.app:app", host="0.0.0.0", port=8000, reload=True)
