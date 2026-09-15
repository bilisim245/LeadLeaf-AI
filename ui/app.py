"""
Gradio Demo Arayüzü — "Tarla 360" (jüriye/hızlı teste yerel görselleştirme)

GÜNCELLEME (2026-09-15): Tek fotoğraf → tek statik rapor yerine, referans alınan
bir müşteri analitiği dashboard'undaki yapı uyarlandı: geçmiş TREND'i, hava
durumu bazlı RİSK paneli, bölgesel kümelenme ve müdahale SENARYO analizi.
Bunun için "bonus" aşamasına ertelenen `bot/db.py` (tarla defteri) ve
`agent/weather.py` (hava durumu) MVP'ye çekildi — ikisi de zaten yazılıp test
edilmişti, sadece bağlı değildi.

⚠️ ÖNEMLİ DÜRÜSTLÜK NOTU: "Senaryo analizi" tablosu EĞİTİLMİŞ BİR ML MODELİNİN
ÇIKTISI DEĞİLDİR — kural tabanlı, açıkça etiketlenmiş bir örnekleyici
simülasyondur (bkz. README "Etik/sınırlılık notu"). Amaç, farklı müdahalelerin
GÖRECELİ etkisini göstermek; kesin/kalibre edilmiş bir sayı iddiası yoktur.

Akış: çiftçi/tarla bilgisi + görsel → /predict (CNN) → agent/report.py (LLM) →
      db.py'ye kaydet → geçmiş trend + hava durumu riski + senaryo tablosu.

Çalıştırma (önce inference servisini ayrı bir terminalde başlat):
    .venv\\Scripts\\python.exe -m uvicorn inference.app:app --port 8000
    .venv\\Scripts\\python.exe ui/app.py
"""
from __future__ import annotations

import hashlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv

from agent.report import generate_report
from agent.weather import weather_summary
from bot.db import DB

load_dotenv()

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8000")
db = DB()  # bot/tarla_defteri.sqlite — yerel dosya, ek kurulum yok

RISK_CARPANI = {"dusuk": 0.9, "orta": 1.0, "yuksek": 1.15, "bilinmiyor": 1.0}


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


def _senaryo_tablosu(risk: float):
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
        satirlar.append([ad, f"%{yeni_risk}", f"{fark:+.1f}", not_])
    return satirlar


def _trend_grafigi(gecmis: list[dict]):
    fig, ax = plt.subplots(figsize=(5, 3))
    if not gecmis:
        ax.text(0.5, 0.5, "Henüz geçmiş kayıt yok\n(ilk analiz bu olacak)",
                ha="center", va="center", fontsize=10)
        ax.axis("off")
        return fig

    gecmis = list(reversed(gecmis))  # eskiden yeniye
    tarihler = [g["ts"][:10] for g in gecmis]
    guvenler = [g["guven"] * 100 if g["guven"] <= 1 else g["guven"] for g in gecmis]
    ax.plot(tarihler, guvenler, marker="o", color="#c0392b")
    ax.set_ylabel("Güven / risk (%)")
    ax.set_title("Bu tarlada geçmiş gözlemler")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def analiz_et(isim, il, ilce, urun, image):
    if image is None:
        return ("Lütfen bir yaprak fotoğrafı yükleyin.", {}, None, "", [], {})
    isim = (isim or "Çiftçi").strip()
    il = (il or "Antalya").strip()
    ilce = (ilce or "Serik").strip()

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
        hata = (
            f"❌ Inference servisine ulaşılamadı ({INFERENCE_URL}).\n\n"
            "Önce şunu ayrı bir terminalde çalıştırın:\n"
            "  .venv\\Scripts\\python.exe -m uvicorn inference.app:app --port 8000"
        )
        return (hata, {}, None, "", [], {})
    except Exception as e:
        return (f"❌ Hata: {e}", {}, None, "", [], {})

    rapor = generate_report(cnn["hastalik"], cnn["hastalik_tr"], cnn["guven"])

    # --- Tarla defteri: kullanıcı/alan + gözlem kaydı (bot/db.py, artık entegre) ---
    uid = _demo_user_id(isim, il, ilce)
    db.upsert_user(uid, isim, il=il, ilce=ilce)
    fid = db.get_or_create_default_field(uid, crop=urun or "domates")
    db.add_observation(uid, fid, hastalik=cnn["hastalik"], guven=cnn["guven"] / 100,
                        baglam={"il": il, "ilce": ilce}, ozet=rapor.get("aciklama", ""))
    gecmis = db.history(uid, fid, limit=10)
    kume_sayisi = db.recent_cluster(cnn["hastalik"], ilce, gun=7) if cnn["hastalik"] != "Tomato___healthy" else 0

    # --- Hava durumu bazlı mantar riski (agent/weather.py, artık entegre) ---
    try:
        hava = weather_summary(f"{ilce}, {il}")
    except Exception as e:
        hava = {"bulundu": False, "ozet_metni": f"Hava durumu alınamadı: {e}", "mantar_riski": "bilinmiyor"}

    risk = _risk_skoru(cnn["hastalik"], cnn["guven"], hava.get("mantar_riski", "bilinmiyor"), kume_sayisi)
    senaryo = _senaryo_tablosu(risk)
    trend_fig = _trend_grafigi(gecmis)

    etiketler = {it["sinif_tr"]: it["olasilik"] / 100 for it in cnn["ilk3"]}

    demo_uyarisi = (
        "\n\n> ⚠️ **DEMO MODU:** Gerçek model henüz `model/` klasörüne konmadı "
        "(Kaggle/Colab eğitimi tamamlanmadı). Sınıflandırma sonucu RASTGELE üretildi.\n"
        if cnn.get("demo_mode") else ""
    )
    kaynak_notu = (
        "\n\n*(Rapor: yerel şablon — `ANTHROPIC_API_KEY` .env'de tanımlı değil.)*"
        if rapor.get("_kaynak") == "sablon" else ""
    )

    kume_notu = (
        f"\n\n📍 **Bölgesel not:** Son 7 günde {ilce}'de aynı hastalığı bildiren "
        f"**{kume_sayisi} farklı çiftçi** daha var." if kume_sayisi > 0 else ""
    )

    md = f"""{demo_uyarisi}
## 🔎 Tespit: {cnn['hastalik_tr']}
**Güven:** %{cnn['guven']} · **Risk skoru (hava + bölge dahil):** %{risk}

{"### ⚠️ Bu sonuç kesin değil — bir ziraat mühendisine danışmanızı öneririz." if cnn['uzmana_yonlendir'] else ""}

### Açıklama
{rapor.get('aciklama', '-')}

### Önerilen kültürel/biyolojik önlemler
{rapor.get('onlem', '-')}
{kume_notu}

---
*{rapor.get('uyari', 'Bu bir ön değerlendirmedir, kesin teşhis değildir.')}*
{kaynak_notu}
"""

    hava_md = f"""### 🌦️ Hava durumu — mantar hastalığı riski
{hava.get('ozet_metni', '-')}
"""

    return md, etiketler, trend_fig, hava_md, senaryo, cnn


with gr.Blocks(title="LeadLeaf AI — Tarla 360") as demo:
    gr.Markdown(
        "# 🍅 LeadLeaf AI — Tarla 360 (Yerel Demo)\n"
        "Çiftçi/tarla bilgisi + yaprak fotoğrafı → CNN sınıflandırma + LLM raporu + "
        "**geçmiş trend** + **hava durumu riski** + **müdahale senaryo analizi**.\n\n"
        "*Üretimde bu akış n8n (Telegram bot) üzerinden çalışır — bu ekran yerel "
        "görselleştirme/test amaçlıdır.*"
    )
    with gr.Row():
        isim_in = gr.Textbox(label="Çiftçi adı", value="Ahmet")
        il_in = gr.Textbox(label="İl", value="Antalya")
        ilce_in = gr.Textbox(label="İlçe", value="Serik")
        urun_in = gr.Textbox(label="Ürün", value="domates")

    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Yaprak fotoğrafı")
            btn = gr.Button("Analiz Et", variant="primary")
        with gr.Column():
            lbl_out = gr.Label(label="İlk 3 tahmin")
            md_out = gr.Markdown(label="Rapor")

    with gr.Row():
        trend_out = gr.Plot(label="Geçmiş trend (bu tarla)")
        hava_out = gr.Markdown(label="Hava durumu riski")

    gr.Markdown("### 🎯 Riski düşürmek için ne yapmalı? (kural tabanlı senaryo — ML tahmini DEĞİL)")
    senaryo_out = gr.Dataframe(
        headers=["Senaryo", "Modellenen risk", "Fark", "Not"],
        label="Senaryo analizi",
    )

    with gr.Accordion("Ham CNN çıktısı (debug)", open=False):
        json_out = gr.JSON()

    btn.click(
        analiz_et,
        inputs=[isim_in, il_in, ilce_in, urun_in, img_in],
        outputs=[md_out, lbl_out, trend_out, hava_out, senaryo_out, json_out],
    )

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
