"""Sunum panosu (ui/sunum.py) sayfalarının ortak yardımcıları: yollar, veri yükleyiciler,
renkler ve sayfalar arası "Önceki / Sonraki" gezinmesi."""
from __future__ import annotations

import ast
import json
import os
import random

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERI_DIR = os.path.join(KOK, "data", "plantvillage", "raw", "color")
M38_DIR = os.path.join(KOK, "model", "model_38sinif")
TUBITAK_DIR = os.path.join(KOK, "model", "tubitak")
KNOWLEDGE_DIR = os.path.join(KOK, "agent", "knowledge")

LACIVERT, ORTA, ACIK, YESIL, TURUNCU = "#0F2347", "#2E5A9A", "#8FB4E8", "#2F7D4A", "#D9822B"
DURUM_RENK = alt.Scale(domain=["Hastalıklı", "Sağlıklı"], range=["#1B3A6B", ACIK])

BITKI_TR = {"Tomato": "Domates", "Potato": "Patates", "Pepper,_bell": "Biber", "Apple": "Elma",
            "Peach": "Şeftali", "Cherry_(including_sour)": "Kiraz", "Grape": "Üzüm",
            "Corn_(maize)": "Mısır", "Strawberry": "Çilek", "Orange": "Portakal",
            "Raspberry": "Ahududu", "Soybean": "Soya", "Squash": "Kabak", "Blueberry": "Yaban mersini"}

# Sunum sırası — sidebar'daki sıra ve Önceki/Sonraki butonları bu listeyi izler
SAYFALAR = [
    ("sayfalar/01_ozet.py", "Proje Özeti", "🏠"),
    ("sayfalar/02_veri_seti.py", "Veri Seti", "🗂️"),
    ("sayfalar/03_veri_analizi.py", "Keşifsel Veri Analizi", "📊"),
    ("sayfalar/04_cnn.py", "CNN ve Transfer Learning", "🧠"),
    ("sayfalar/05_model_secimi.py", "Model Karşılaştırma", "⚖️"),
    ("sayfalar/06_fine_tuning.py", "Fine-Tuning", "🔧"),
    ("sayfalar/07_test_analizi.py", "Test Sonuçları (38 Sınıf)", "🎯"),
    ("sayfalar/08_aciklanabilirlik.py", "Model Nereye Bakıyor?", "🔎"),
    ("sayfalar/09_rag.py", "RAG ve Rapor", "📚"),
    ("app.py", "Canlı Demo — Tarla 360", "🌿"),
    ("sayfalar/10_sinirliliklar.py", "Sınırlılıklar ve Sonraki Adım", "🧭"),
]


def gezinme(dosya: str) -> None:
    """Sayfanın altına Önceki / Sonraki butonları koyar (sunumda sırayla gitmek için)."""
    yollar = [s[0] for s in SAYFALAR]
    anahtar = os.path.relpath(dosya, os.path.join(KOK, "ui")).replace("\\", "/")
    if anahtar not in yollar:
        return
    i = yollar.index(anahtar)
    st.divider()
    sol, orta, sag = st.columns([1, 2, 1])
    if i > 0 and sol.button(f"← {SAYFALAR[i - 1][1]}", use_container_width=True):
        st.switch_page(SAYFALAR[i - 1][0])
    orta.markdown(f"<div style='text-align:center;color:#6A7890'>{i + 1} / {len(SAYFALAR)}</div>",
                  unsafe_allow_html=True)
    if i < len(SAYFALAR) - 1 and sag.button(f"{SAYFALAR[i + 1][1]} →", type="primary",
                                            use_container_width=True):
        st.switch_page(SAYFALAR[i + 1][0])


def bitki(sinif: str) -> str:
    return BITKI_TR.get(sinif.split("___")[0], sinif.split("___")[0])


def saglikli(sinif: str) -> bool:
    return sinif.endswith("___healthy")


@st.cache_data
def tr_adlar() -> dict:
    """inference/app.py'deki TR_ADLAR sözlüğü (TensorFlow import etmeden, ast ile)."""
    with open(os.path.join(KOK, "inference", "app.py"), encoding="utf-8") as f:
        agac = ast.parse(f.read())
    for dugum in agac.body:
        if isinstance(dugum, ast.Assign) and getattr(dugum.targets[0], "id", "") == "TR_ADLAR":
            return ast.literal_eval(dugum.value)
    return {}


def etiket(sinif: str) -> str:
    """'Domates — Geç Yanıklık' gibi kısa Türkçe etiket."""
    if saglikli(sinif):
        return f"{bitki(sinif)} — Sağlıklı"
    return f"{bitki(sinif)} — {tr_adlar().get(sinif, sinif).split(' (')[0]}"


@st.cache_data
def manifest() -> dict:
    with open(os.path.join(M38_DIR, "split_manifest.json"), encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def sinif_tablosu() -> pd.DataFrame:
    """Her sınıf için eğitim/doğrulama/test sayısı (manifest'ten)."""
    s = manifest()["sayilar"]
    df = pd.DataFrame([
        {"Sınıf": c, "Etiket": etiket(c), "Bitki": bitki(c),
         "Durum": "Sağlıklı" if saglikli(c) else "Hastalıklı",
         "Eğitim": s["train"].get(c, 0), "Doğrulama": s["valid"].get(c, 0), "Test": s["test"].get(c, 0)}
        for c in s["train"]
    ])
    df["Toplam"] = df[["Eğitim", "Doğrulama", "Test"]].sum(axis=1)
    return df


def veri_var() -> bool:
    return os.path.isdir(VERI_DIR)


def veri_yok_uyarisi() -> None:
    st.warning("Veri seti bu bilgisayarda yok (`data/plantvillage/raw/color`). İndirmek için: "
               "`git clone --depth 1 --filter=blob:none --sparse "
               "https://github.com/spMohanty/PlantVillage-Dataset.git data/plantvillage` ve "
               "`git -C data/plantvillage sparse-checkout set raw/color`")


def ornek_dosyalar(sinif: str, n: int, tohum: int) -> list[str]:
    klasor = os.path.join(VERI_DIR, sinif)
    dosyalar = sorted(os.listdir(klasor))
    random.Random(tohum).shuffle(dosyalar)
    return [os.path.join(klasor, d) for d in dosyalar[:n]]


@st.cache_data
def test_sonuclari() -> dict | None:
    yol = os.path.join(M38_DIR, "test_sonuclari.npz")
    if not os.path.exists(yol):
        return None
    z = np.load(yol, allow_pickle=False)
    return {"y_true": z["y_true"].astype(int), "y_prob": z["y_prob"].astype(np.float32),
            "siniflar": [str(s) for s in z["siniflar"]], "dosyalar": [str(d) for d in z["dosyalar"]]}


@st.cache_resource
def uretim_modeli():
    """Üretim modeli (model/model.keras) — ilk çağrıda ~10 sn yüklenir, sonra bellekte kalır."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import keras
    return keras.models.load_model(os.path.join(KOK, "model", "model.keras"), compile=False)


def kod_parcasi(dosya: str, baslangic: str, bitis: str | None = None) -> str:
    """Bir dosyadan `baslangic` satırıyla başlayan (ve `bitis` satırından önce biten) kod
    bloğunu okur — sunumdaki kodlar her zaman projedeki GERÇEK kod olsun diye."""
    with open(os.path.join(KOK, dosya), encoding="utf-8") as f:
        satirlar = f.read().splitlines()
    i = next(k for k, s in enumerate(satirlar) if s.strip().startswith(baslangic))
    j = len(satirlar)
    if bitis:
        j = next((k for k in range(i + 1, len(satirlar)) if satirlar[k].strip().startswith(bitis)), j)
    return "\n".join(satirlar[i:j]).rstrip()
