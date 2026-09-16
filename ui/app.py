"""
Streamlit Demo Arayüzü — "Tarla 360" (jüriye/hızlı teste yerel görselleştirme)

DEĞİŞİKLİK (2026-09-16): Gradio'dan Streamlit'e geçildi — geri bildirim: Gradio'da
kart görünümü için kullanılan özel HTML/CSS zorlama durdu ve akış istenen gibi
olmadı. Streamlit'in kendi native bileşenleri (st.success/warning/error, st.metric,
st.line_chart, st.dataframe) aynı bilgiyi CSS'e gerek kalmadan, üstten-alta doğal
bir akışla (sırayla: bilgi gir → fotoğraf yükle → sonucu gör) veriyor.

İki sekme var:
  1. "Analiz" — asıl akış: çiftçi/tarla bilgisi + görsel → /predict (CNN) →
     agent/report.py (LLM + RAG) → db.py'ye kaydet → trend + hava riski + senaryo.
  2. "Veri Analizi" — notebooks/00_veri_kesfi.py'nin Colab'da ürettiği EDA
     çıktılarını (sınıf dağılımı, boyut istatistiği, bozuk/tekrar eden görsel
     kontrolü) grafiklerle gösterir. "Bootcamp'ten istenen veri analizini nerede
     görüyoruz" sorusunun cevabı burası.

Çalıştırma (önce inference servisini ayrı bir terminalde başlat):
    .venv\\Scripts\\python.exe -m uvicorn inference.app:app --port 8000
    .venv\\Scripts\\python.exe -m streamlit run ui/app.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from agent.report import generate_report
from agent.weather import weather_summary
from bot.db import DB

load_dotenv()

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8000")
PROJE_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_IMAGES_DIR = os.path.join(PROJE_KOKU, "model", "demo_images")
EDA_DIR = os.path.join(PROJE_KOKU, "report", "eda_ciktilari")

RISK_CARPANI = {"dusuk": 0.9, "orta": 1.0, "yuksek": 1.15, "bilinmiyor": 1.0}


@st.cache_resource
def get_db() -> DB:
    return DB()  # bot/tarla_defteri.sqlite — yerel dosya, ek kurulum yok


def _demo_user_id(isim: str, il: str, ilce: str) -> int:
    """Telegram user_id yok (bu yerel demo) — isim+konumdan sabit bir id üretir,
    böylece aynı çiftçi tekrar girince geçmişi/trendi görülür."""
    h = hashlib.md5(f"{isim}|{il}|{ilce}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _risk_skoru(hastalik: str, guven: float, hava_riski: str, kume_sayisi: int) -> float:
    """Basit, ŞEFFAF bir sezgisel formül — ML tahmini DEĞİL.
    hastalik saglikliysa risk dusuk baz alinir; degilse guven kendisi risk baz alinir.
    Hava durumu (mantar riski) ve bolgesel kumelenme carpani/ek puan ekler."""
    baz = guven if hastalik != "Tomato___healthy" else max(5.0, 100 - guven)
    carpan = RISK_CARPANI.get(hava_riski, 1.0)
    kume_bonus = min(20.0, kume_sayisi * 5.0)
    return round(min(100.0, baz * carpan + kume_bonus), 1)


def _senaryo_tablosu(risk: float) -> pd.DataFrame:
    """Kural tabanli, aciklamali senaryo simulasyonu (referans: 'Riski dusurmek
    icin ne yapmali?' paneli). Katsayilar sezgiseldir, kalibre edilmemistir."""
    senaryolar = [
        ("Mevcut durum (hiçbir şey yapma)", 1.00, "—"),
        ("Kültürel/biyolojik önlem uygula", 0.70, "Budama, sulama düzeni, havalandırma"),
        ("3 gün içinde tekrar fotoğraf çek", 0.85, "Erken takip, ilerleme kontrolü"),
        ("Ziraat mühendisine danış", 0.50, "Ruhsatlı ürün/doz uzman kararıyla"),
    ]
    satirlar = []
    for ad, katsayi, not_ in senaryolar:
        yeni_risk = round(risk * katsayi, 1)
        fark = round(yeni_risk - risk, 1)
        satirlar.append({"Senaryo": ad, "Modellenen risk (%)": yeni_risk,
                          "Fark": fark, "Not": not_})
    return pd.DataFrame(satirlar)


def _risk_durum_kutusu(risk: float, mesaj: str) -> None:
    """Streamlit'in native renkli kutularini kullaniyoruz — ozel CSS yok."""
    if risk < 30:
        st.success(mesaj)
    elif risk < 60:
        st.warning(mesaj)
    else:
        st.error(mesaj)


st.set_page_config(page_title="LeadLeaf AI — Tarla 360", page_icon="🍅", layout="wide")

st.title("🍅 LeadLeaf AI — Tarla 360")

tab_analiz, tab_veri, tab_karsilastirma = st.tabs(
    ["🔎 Analiz", "📊 Veri Analizi", "🔬 Model Karşılaştırma"]
)

# =====================================================================
# SEKME 1 — ANALİZ (asıl akış)
# =====================================================================
with tab_analiz:
    st.caption(
        "Çiftçi/tarla bilgisi + yaprak fotoğrafı → CNN sınıflandırma + RAG destekli LLM "
        "raporu + geçmiş trend + hava durumu riski + müdahale senaryo analizi. "
        "*(Üretimde bu akış n8n/Telegram bot üzerinden çalışır — bu ekran yerel "
        "görselleştirme/test amaçlıdır.)*"
    )

    with st.sidebar:
        st.header("👤 Çiftçi ve tarla bilgisi")
        isim = st.text_input("Çiftçi adı", value="Ahmet")
        il = st.text_input("İl", value="Antalya")
        ilce = st.text_input("İlçe", value="Serik")
        urun = st.text_input("Ürün", value="domates")

    st.header("📸 Fotoğraf")
    yuklenen = st.file_uploader("Yaprak fotoğrafı yükle", type=["jpg", "jpeg", "png"])

    col_img, col_btn = st.columns([1, 3])
    with col_img:
        if yuklenen:
            st.image(yuklenen, caption="Yüklenen fotoğraf", width=220)

    analiz_tiklandi = st.button("🔍 Analiz Et", type="primary", disabled=yuklenen is None)

    if analiz_tiklandi and yuklenen is not None:
        image = Image.open(yuklenen)

        with st.spinner("Görsel analiz ediliyor..."):
            try:
                buf = io.BytesIO()
                image.convert("RGB").save(buf, format="JPEG")
                buf.seek(0)
                r = requests.post(
                    f"{INFERENCE_URL}/predict",
                    files={"file": ("yaprak.jpg", buf, "image/jpeg")},
                    timeout=30,
                )
                r.raise_for_status()
                cnn = r.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Inference servisine ulaşılamadı ({INFERENCE_URL}).\n\n"
                    "Önce şunu ayrı bir terminalde çalıştır:\n"
                    "`.venv\\Scripts\\python.exe -m uvicorn inference.app:app --port 8000`"
                )
                st.stop()
            except Exception as e:
                st.error(f"❌ Hata: {e}")
                st.stop()

        if cnn.get("demo_mode"):
            st.warning(
                "⚠️ **DEMO MODU:** Gerçek model henüz `model/` klasörüne konmadı "
                "(Colab eğitimi tamamlanmadı). Sınıflandırma sonucu RASTGELE üretildi."
            )

        with st.spinner("Rapor hazırlanıyor (RAG + LLM)..."):
            rapor = generate_report(cnn["hastalik"], cnn["hastalik_tr"], cnn["guven"])

        db = get_db()
        uid = _demo_user_id(isim, il, ilce)
        db.upsert_user(uid, isim, il=il, ilce=ilce)
        fid = db.get_or_create_default_field(uid, crop=urun or "domates")
        db.add_observation(uid, fid, hastalik=cnn["hastalik"], guven=cnn["guven"] / 100,
                            baglam={"il": il, "ilce": ilce}, ozet=rapor.get("aciklama", ""))
        gecmis = db.history(uid, fid, limit=10)
        kume_sayisi = (db.recent_cluster(cnn["hastalik"], ilce, gun=7)
                       if cnn["hastalik"] != "Tomato___healthy" else 0)

        with st.spinner("Hava durumu kontrol ediliyor..."):
            hava = weather_summary(f"{ilce}, {il}")

        risk = _risk_skoru(cnn["hastalik"], cnn["guven"],
                            hava.get("mantar_riski", "bilinmiyor"), kume_sayisi)

        st.divider()
        st.header("🔎 Tespit Sonucu")

        c1, c2, c3 = st.columns(3)
        c1.metric("Hastalık", cnn["hastalik_tr"])
        c2.metric("Model güveni", f"%{cnn['guven']}")
        c3.metric("Risk skoru (hava + bölge dahil)", f"%{risk}")

        _risk_durum_kutusu(
            risk,
            f"**{cnn['hastalik_tr']}** — risk seviyesi "
            f"{'düşük' if risk < 30 else 'orta' if risk < 60 else 'yüksek'}."
            + (" ⚠️ Bir ziraat mühendisine danışmanız önerilir."
               if cnn.get("uzmana_yonlendir") else ""),
        )

        st.subheader("Açıklama")
        st.write(rapor.get("aciklama", "-"))

        st.subheader("Önerilen kültürel/biyolojik önlemler")
        st.markdown(rapor.get("onlem", "-"))

        if kume_sayisi > 0:
            st.info(f"📍 Son 7 günde **{ilce}** ilçesinde aynı hastalığı bildiren "
                    f"**{kume_sayisi} farklı çiftçi** daha var.")

        st.caption(rapor.get("uyari", "Bu bir ön değerlendirmedir, kesin teşhis değildir."))
        if rapor.get("_rag_kullanildi"):
            st.caption("🔗 Bu açıklama doğrulanmış kaynak dokümandan (RAG) getirilen bağlama dayanıyor.")
        else:
            st.caption("⚠️ RAG bağlamı bulunamadı — `rag/build_index.py` çalıştırılmamış olabilir.")
        if rapor.get("_kaynak") == "sablon":
            st.caption("*(Rapor: yerel şablon — `ANTHROPIC_API_KEY` .env'de tanımlı değil.)*")

        st.subheader("İlk 3 tahmin")
        for it in cnn["ilk3"]:
            st.progress(it["olasilik"] / 100, text=f"{it['sinif_tr']} — %{it['olasilik']}")

        col_trend, col_hava = st.columns(2)

        with col_trend:
            st.subheader("📈 Geçmiş trend (bu tarla)")
            if gecmis:
                df = pd.DataFrame(reversed(gecmis))
                df["tarih"] = df["ts"].str[:10]
                df["guven_yuzde"] = df["guven"].apply(lambda g: g * 100 if g <= 1 else g)
                st.line_chart(df.set_index("tarih")["guven_yuzde"])
            else:
                st.caption("Henüz geçmiş kayıt yok — bu ilk analiz.")

        with col_hava:
            st.subheader("🌦️ Hava durumu — mantar riski")
            risk_etiket = {"dusuk": "Düşük", "orta": "Orta", "yuksek": "Yüksek",
                            "bilinmiyor": "Bilinmiyor"}.get(hava.get("mantar_riski"), "Bilinmiyor")
            st.metric("Mantar hastalığı riski", risk_etiket)
            st.text(hava.get("ozet_metni", "-"))

        st.subheader("🎯 Riski düşürmek için ne yapmalı?")
        st.caption("(kural tabanlı senaryo — ML tahmini DEĞİL)")
        st.dataframe(_senaryo_tablosu(risk), use_container_width=True, hide_index=True)

        benzerler = cnn.get("benzer_gorseller", [])
        st.subheader("🖼️ Benzer referans görseller")
        st.caption("(görsel RAG — modelin öğrendiği özniteliklerle en yakın örnekler)")
        if benzerler:
            cols = st.columns(len(benzerler))
            for col, it in zip(cols, benzerler):
                yol = os.path.join(DEMO_IMAGES_DIR, it["dosya"])
                if os.path.exists(yol):
                    col.image(yol, caption=f"{it['sinif']} — %{it['benzerlik']}")
        else:
            st.caption("Henüz yok — gerçek model + referans görseller eklenince burada görünecek.")

        with st.expander("🔧 Ham CNN çıktısı (debug)"):
            st.json(cnn)

    elif not yuklenen:
        st.info("👆 Başlamak için bir yaprak fotoğrafı yükle.")

# =====================================================================
# SEKME 2 — VERİ ANALİZİ (notebooks/00_veri_kesfi.py çıktıları)
# =====================================================================
with tab_veri:
    st.caption(
        "`notebooks/00_veri_kesfi.py` Colab'da çalıştırılıp indirilen "
        "`eda_ciktilari.zip` içeriği `report/eda_ciktilari/` klasörüne çıkarılınca "
        "burası otomatik dolar. Amaç: modele geçmeden veriyi VARSAYIMLA değil "
        "SAYIYLA incelemek — sınıf dengesizliği, görsel kalitesi, bozuk/tekrar "
        "eden görsel kontrolü."
    )

    ozet_json_yolu = os.path.join(EDA_DIR, "veri_ozeti.json")
    if not os.path.exists(ozet_json_yolu):
        st.warning(
            "⚠️ Henüz veri analizi çıktısı yok.\n\n"
            "**Nasıl doldurulur:** Google Colab'da `notebooks/00_veri_kesfi.py`'yi "
            "çalıştır (önce `01_train_model_colab.py`'deki veri indirme hücresini "
            "çalıştırmış olman gerekir) → inen `eda_ciktilari.zip`'i aç → "
            "içindekileri projede `report/eda_ciktilari/` klasörüne kopyala → "
            "bu sayfayı yenile."
        )
    else:
        with open(ozet_json_yolu, "r", encoding="utf-8") as f:
            ozet = json.load(f)

        st.subheader("📌 Genel özet")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam sınıf (tüm veri seti)", ozet["toplam_sinif"])
        c2.metric("Toplam görsel", f"{ozet['toplam_gorsel']:,}")
        c3.metric("Dengesizlik oranı (domates)", f"{ozet['dengesizlik_orani']}x")
        c4.metric("Bozuk/açılamayan görsel", ozet["bozuk_gorsel"]["toplam"])

        st.subheader("🍅 Domates alt kümesi — sınıf dağılımı")
        df_sinif = pd.DataFrame(
            [{"Sınıf": k, "Görsel sayısı": v} for k, v in ozet["domates_sayilar"].items()]
        )
        st.bar_chart(df_sinif.set_index("Sınıf"))
        st.dataframe(df_sinif, use_container_width=True, hide_index=True)

        gb = ozet.get("gorsel_boyutu", {})
        st.subheader("📐 Görsel boyutu")
        if gb.get("sabit"):
            st.success(f"Tüm görseller aynı boyutta: {gb['genislik']}×{gb['yukseklik']}px")
        else:
            st.info("Görsel boyutları değişken (model eğitimi zaten hepsini 224×224'e ölçekliyor).")

        st.subheader("🧹 Bozuk / açılamayan görsel (\"boş veri\" kontrolü)")
        df_bozuk = pd.DataFrame(
            [{"Sınıf": k, "Bozuk görsel": v} for k, v in ozet["bozuk_gorsel"]["sinif_bazli"].items()]
        )
        if ozet["bozuk_gorsel"]["toplam"] > 0:
            st.error(f"Toplam {ozet['bozuk_gorsel']['toplam']} bozuk/açılamayan görsel bulundu — "
                     "eğitimden önce temizlenmesi önerilir.")
        else:
            st.success("Bozuk/açılamayan görsel bulunamadı.")
        st.dataframe(df_bozuk, use_container_width=True, hide_index=True)

        st.subheader("🔁 Tekrar eden (duplicate) görsel kontrolü")
        st.caption(
            "Ham veri havuzunda aynı/çok benzer görsel var mı — varsa split sırasında "
            "biri train'e biri valid'e düşüp yapay bir sızıntı yaratabilir."
        )
        df_tekrar = pd.DataFrame([
            {"Sınıf": k, "Örneklenen": v["orneklenen"], "Tekrar eden çift": v["tekrar_cift"],
             "Oran (%)": v["oran_yuzde"]}
            for k, v in ozet.get("tekrar_eden", {}).items()
        ])
        st.dataframe(df_tekrar, use_container_width=True, hide_index=True)
        yuksek_tekrar = df_tekrar[df_tekrar["Oran (%)"] > 5] if not df_tekrar.empty else df_tekrar
        if not yuksek_tekrar.empty:
            st.warning(f"⚠️ {len(yuksek_tekrar)} sınıfta %5'in üzerinde tekrar oranı — "
                       "split öncesi tekilleştirme (deduplication) düşünülebilir.")
        else:
            st.success("Tüm sınıflarda tekrar oranı düşük (%5 altı).")

        st.subheader("🖼️ Örnek görseller ve grafikler")
        for dosya, baslik in [
            ("sinif_dagilimi.png", "Sınıf dağılımı — tüm veri seti"),
            ("ornek_gorseller_izgara.png", "Domates 5 sınıftan örnekler"),
            ("boyut_dagilimi.png", "Görsel boyutu dağılımı"),
        ]:
            yol = os.path.join(EDA_DIR, dosya)
            if os.path.exists(yol):
                st.image(yol, caption=baslik, use_container_width=True)

        with st.expander("📄 Ham özet (veri_ozeti.txt)"):
            txt_yolu = os.path.join(EDA_DIR, "veri_ozeti.txt")
            if os.path.exists(txt_yolu):
                with open(txt_yolu, "r", encoding="utf-8") as f:
                    st.text(f.read())

# =====================================================================
# SEKME 3 — MODEL KARŞILAŞTIRMA (TÜBİTAK: MobileNetV2 vs MobileNetV3Small vs EfficientNetB0)
# =====================================================================
with tab_karsilastirma:
    st.caption(
        "Aynı fotoğraf `/predict_compare` üzerinden üç modele birden (MobileNetV2, "
        "MobileNetV3Small, EfficientNetB0 — `notebooks/01_train_model_colab.py`'nin "
        "TÜBİTAK sürümünde aynı split/koşullarla eğitilir) gönderilir. Model "
        "uzlaşması ve %70 güven eşiği **kural tabanlı** yorumlanır — bu bir ML "
        "tahmini değil, üç sonucu birleştiren açıklanabilir bir karardır."
    )

    karsilastirma_foto = st.file_uploader(
        "Yaprak fotoğrafı yükle", type=["jpg", "jpeg", "png"], key="karsilastirma_uploader"
    )
    if karsilastirma_foto:
        st.image(karsilastirma_foto, caption="Yüklenen fotoğraf", width=220)

    karsilastir_tiklandi = st.button(
        "🔬 Üç Modelle Karşılaştır", type="primary", disabled=karsilastirma_foto is None
    )

    if karsilastir_tiklandi and karsilastirma_foto is not None:
        with st.spinner("Üç model de çalıştırılıyor..."):
            try:
                img = Image.open(karsilastirma_foto)
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG")
                buf.seek(0)
                r = requests.post(
                    f"{INFERENCE_URL}/predict_compare",
                    files={"file": ("yaprak.jpg", buf, "image/jpeg")},
                    timeout=60,
                )
                r.raise_for_status()
                karsilastirma = r.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Inference servisine ulaşılamadı ({INFERENCE_URL}).\n\n"
                    "Önce şunu ayrı bir terminalde çalıştır:\n"
                    "`.venv\\Scripts\\python.exe -m uvicorn inference.app:app --port 8000`"
                )
                st.stop()
            except Exception as e:
                st.error(f"❌ Hata: {e}")
                st.stop()

        sonuclar = karsilastirma["sonuclar"]
        uzlasma = karsilastirma["uzlasma"]

        if any(s["demo_mode"] for s in sonuclar):
            eksikler = [s["model"] for s in sonuclar if s["demo_mode"]]
            st.warning(
                f"⚠️ **DEMO MODU:** şu modeller henüz `model/tubitak/` içinde yok, "
                f"sonuçları RASTGELE üretildi: {', '.join(eksikler)}. TÜBİTAK notebook'u "
                f"çalıştırılıp çıktı oraya kopyalanınca gerçek sonuca döner."
            )

        st.divider()
        st.subheader("📊 Üç modelin sonucu")
        df_karsilastirma = pd.DataFrame([
            {
                "Model": s["model"],
                "Tahmin": s["hastalik_tr"],
                "Güven (%)": s["guven"],
                "Eşik (%70) durumu": "⚠️ Altında" if s["uzmana_yonlendir"] else "✅ Üstünde",
                "Mod": "Demo" if s["demo_mode"] else "Gerçek",
            }
            for s in sonuclar
        ])
        st.dataframe(df_karsilastirma, use_container_width=True, hide_index=True)

        st.subheader("🤝 Model uzlaşması")
        durum_etiket = {"tam": "Tam uzlaşma", "kismi": "Kısmi uzlaşma", "yok": "Uzlaşma yok"}
        c1, c2, c3 = st.columns(3)
        c1.metric("Uzlaşma durumu", durum_etiket.get(uzlasma["durum"], uzlasma["durum"]))
        c2.metric("Çoğunluk tahmini", f"{uzlasma['cogunluk_sinif_tr']} ({uzlasma['cogunluk_adet']}/{uzlasma['toplam_model']})")
        c3.metric("En düşük güven", f"%{uzlasma['en_dusuk_guven']}")

        if uzlasma["durum"] == "tam" and uzlasma["en_dusuk_guven"] >= uzlasma["esik_yuzde"]:
            st.success(f"✅ {uzlasma['tavsiye']}")
        elif uzlasma["durum"] == "yok":
            st.error(f"❌ {uzlasma['tavsiye']}")
        else:
            st.warning(f"⚠️ {uzlasma['tavsiye']}")

        with st.expander("🔧 Ham API çıktısı (debug)"):
            st.json(karsilastirma)
    elif not karsilastirma_foto:
        st.info("👆 Başlamak için bir yaprak fotoğrafı yükle.")
