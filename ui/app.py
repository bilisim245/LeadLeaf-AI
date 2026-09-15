"""
Gradio Demo Arayüzü — "Tarla 360" (jüriye/hızlı teste yerel görselleştirme)

TASARIM NOTU (2026-09-15): Arayüz, düz metin listesi yerine RENK KODLU, KARTLI bir
dashboard olarak tasarlandı — risk seviyesine göre yeşil/sarı/kırmızı rozet, ayrı
kartlarda trend/hava/senaryo/benzer-görsel blokları. Amaç: kod bilmeyen birinin bile
tek bakışta "durum iyi mi kötü mü, ne yapmalı" sorusuna cevap bulabilmesi.

Akış: çiftçi/tarla bilgisi + görsel → /predict (CNN, + varsa benzer referans görseller)
      → agent/report.py (LLM + RAG) → db.py'ye kaydet → trend + hava riski + senaryo.

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
PROJE_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_IMAGES_DIR = os.path.join(PROJE_KOKU, "model", "demo_images")
db = DB()  # bot/tarla_defteri.sqlite — yerel dosya, ek kurulum yok

RISK_CARPANI = {"dusuk": 0.9, "orta": 1.0, "yuksek": 1.15, "bilinmiyor": 1.0}

# --- Renkler (risk seviyesine göre) --------------------------------------------------
YESIL = {"bg": "#ecfdf5", "border": "#10b981", "text": "#065f46"}
SARI = {"bg": "#fffbeb", "border": "#f59e0b", "text": "#92400e"}
KIRMIZI = {"bg": "#fef2f2", "border": "#ef4444", "text": "#991b1b"}


def _risk_rengi(risk: float) -> dict:
    if risk < 30:
        return YESIL
    if risk < 60:
        return SARI
    return KIRMIZI


CUSTOM_CSS = """
.baslik-banner {
    background: linear-gradient(135deg, #059669 0%, #10b981 60%, #34d399 100%);
    color: white; padding: 24px 28px; border-radius: 16px; margin-bottom: 8px;
}
.baslik-banner h1 { margin: 0 0 6px 0; font-size: 1.5rem; }
.baslik-banner p { margin: 0; opacity: 0.92; font-size: 0.92rem; }
.kart { border-radius: 14px !important; }
.tanidashboard { font-family: inherit; }
"""


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
    fig.patch.set_facecolor("#fafafa")
    if not gecmis:
        ax.text(0.5, 0.5, "Henüz geçmiş kayıt yok\n(ilk analiz bu olacak)",
                ha="center", va="center", fontsize=10, color="#6b7280")
        ax.axis("off")
        return fig

    gecmis = list(reversed(gecmis))  # eskiden yeniye
    tarihler = [g["ts"][:10] for g in gecmis]
    guvenler = [g["guven"] * 100 if g["guven"] <= 1 else g["guven"] for g in gecmis]
    ax.plot(tarihler, guvenler, marker="o", color="#059669", linewidth=2)
    ax.fill_between(range(len(tarihler)), guvenler, alpha=0.08, color="#059669")
    ax.set_ylabel("Güven / risk (%)")
    ax.set_title("Bu tarlada geçmiş gözlemler", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _rapor_html(cnn: dict, rapor: dict, risk: float, kume_sayisi: int, ilce: str) -> str:
    renk = _risk_rengi(risk)
    demo_banner = (
        f'<div style="background:#fef3c7;border-left:4px solid #f59e0b;padding:10px 14px;'
        f'border-radius:8px;margin-bottom:14px;font-size:0.9rem;color:#78350f;">'
        f'⚠️ <b>DEMO MODU:</b> Gerçek model henüz yüklenmedi (Colab eğitimi tamamlanmadı). '
        f'Sınıflandırma sonucu RASTGELE üretildi — gösterim amaçlıdır.</div>'
        if cnn.get("demo_mode") else ""
    )
    uzman_uyarisi = (
        f'<div style="background:#fee2e2;border-left:4px solid #ef4444;padding:10px 14px;'
        f'border-radius:8px;margin:12px 0;font-size:0.9rem;color:#7f1d1d;">'
        f'⚠️ Bu sonuç kesin değil — bir <b>ziraat mühendisine danışmanız</b> önerilir.</div>'
        if cnn.get("uzmana_yonlendir") else ""
    )
    kume_notu = (
        f'<p style="font-size:0.88rem;color:#4b5563;margin-top:10px;">📍 Son 7 günde '
        f'<b>{ilce}</b> ilçesinde aynı hastalığı bildiren <b>{kume_sayisi} farklı çiftçi</b> '
        f'daha var.</p>' if kume_sayisi > 0 else ""
    )
    rag_notu = (
        '<p style="font-size:0.82rem;color:#059669;margin-top:10px;">🔗 Bu açıklama, '
        'doğrulanmış kaynak dokümandan RAG (vektör arama) ile getirilen bağlama dayanıyor.</p>'
        if rapor.get("_rag_kullanildi") else
        '<p style="font-size:0.82rem;color:#9ca3af;margin-top:10px;">RAG bağlamı bulunamadı — '
        '<code>rag/build_index.py</code> çalıştırılmamış olabilir.</p>'
    )
    kaynak_notu = (
        '<p style="font-size:0.82rem;color:#9ca3af;">(Rapor: yerel şablon — '
        '<code>ANTHROPIC_API_KEY</code> tanımlı değil.)</p>'
        if rapor.get("_kaynak") == "sablon" else ""
    )

    return f"""
    {demo_banner}
    <div style="background:{renk['bg']};border:1.5px solid {renk['border']};border-radius:14px;
                padding:20px 22px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <h2 style="margin:0;color:{renk['text']};">🔎 {cnn['hastalik_tr']}</h2>
        <span style="background:{renk['border']};color:white;padding:5px 14px;border-radius:999px;
                     font-weight:600;font-size:0.95rem;">Risk: %{risk}</span>
      </div>
      <p style="color:{renk['text']};opacity:0.85;margin:6px 0 0 0;font-size:0.9rem;">
        Model güveni: %{cnn['guven']}
      </p>
      {uzman_uyarisi}
      <h4 style="margin-bottom:4px;">Açıklama</h4>
      <p style="margin-top:0;">{rapor.get('aciklama', '-')}</p>
      <h4 style="margin-bottom:4px;">Önerilen kültürel/biyolojik önlemler</h4>
      <p style="margin-top:0;white-space:pre-line;">{rapor.get('onlem', '-')}</p>
      {kume_notu}
      <hr style="border:none;border-top:1px solid {renk['border']}44;margin:14px 0 8px 0;">
      <p style="font-size:0.85rem;font-style:italic;color:{renk['text']};opacity:0.8;">
        {rapor.get('uyari', 'Bu bir ön değerlendirmedir, kesin teşhis değildir.')}
      </p>
      {kaynak_notu}
      {rag_notu}
    </div>
    """


def _hava_html(hava: dict) -> str:
    risk_map = {"dusuk": ("Düşük", YESIL), "orta": ("Orta", SARI),
                "yuksek": ("Yüksek", KIRMIZI), "bilinmiyor": ("Bilinmiyor", {"bg": "#f3f4f6", "border": "#9ca3af", "text": "#374151"})}
    etiket, renk = risk_map.get(hava.get("mantar_riski", "bilinmiyor"), risk_map["bilinmiyor"])
    return f"""
    <div style="background:{renk['bg']};border:1.5px solid {renk['border']};border-radius:14px;
                padding:16px 18px;height:100%;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <h4 style="margin:0;">🌦️ Hava durumu — mantar riski</h4>
        <span style="background:{renk['border']};color:white;padding:3px 12px;border-radius:999px;
                     font-size:0.85rem;font-weight:600;">{etiket}</span>
      </div>
      <pre style="white-space:pre-wrap;font-family:inherit;font-size:0.85rem;color:{renk['text']};
                  margin:10px 0 0 0;">{hava.get('ozet_metni', '-')}</pre>
    </div>
    """


def analiz_et(isim, il, ilce, urun, image):
    bos_gallery = []
    if image is None:
        uyari_html = '<p style="color:#ef4444;">⚠️ Lütfen bir yaprak fotoğrafı yükleyin.</p>'
        return uyari_html, {}, None, "", [], bos_gallery, {}

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
            f'<p style="color:#ef4444;">❌ Inference servisine ulaşılamadı ({INFERENCE_URL}).<br>'
            "Önce şunu ayrı bir terminalde çalıştırın:<br>"
            "<code>.venv\\Scripts\\python.exe -m uvicorn inference.app:app --port 8000</code></p>"
        )
        return hata, {}, None, "", [], bos_gallery, {}
    except Exception as e:
        return f'<p style="color:#ef4444;">❌ Hata: {e}</p>', {}, None, "", [], bos_gallery, {}

    rapor = generate_report(cnn["hastalik"], cnn["hastalik_tr"], cnn["guven"])

    # --- Tarla defteri: kullanıcı/alan + gözlem kaydı (bot/db.py) ---
    uid = _demo_user_id(isim, il, ilce)
    db.upsert_user(uid, isim, il=il, ilce=ilce)
    fid = db.get_or_create_default_field(uid, crop=urun or "domates")
    db.add_observation(uid, fid, hastalik=cnn["hastalik"], guven=cnn["guven"] / 100,
                        baglam={"il": il, "ilce": ilce}, ozet=rapor.get("aciklama", ""))
    gecmis = db.history(uid, fid, limit=10)
    kume_sayisi = db.recent_cluster(cnn["hastalik"], ilce, gun=7) if cnn["hastalik"] != "Tomato___healthy" else 0

    # --- Hava durumu bazlı mantar riski (agent/weather.py) ---
    try:
        hava = weather_summary(f"{ilce}, {il}")
    except Exception as e:
        hava = {"bulundu": False, "ozet_metni": f"Hava durumu alınamadı: {e}", "mantar_riski": "bilinmiyor"}

    risk = _risk_skoru(cnn["hastalik"], cnn["guven"], hava.get("mantar_riski", "bilinmiyor"), kume_sayisi)
    senaryo = _senaryo_tablosu(risk)
    trend_fig = _trend_grafigi(gecmis)

    etiketler = {it["sinif_tr"]: it["olasilik"] / 100 for it in cnn["ilk3"]}

    rapor_html = _rapor_html(cnn, rapor, risk, kume_sayisi, ilce)
    hava_html = _hava_html(hava)

    # --- Görsel RAG: benzer referans görseller (gerçek model gelince dolu gelir) ---
    galeri = []
    for it in cnn.get("benzer_gorseller", []):
        yol = os.path.join(DEMO_IMAGES_DIR, it["dosya"])
        if os.path.exists(yol):
            galeri.append((yol, f"{it['sinif']} — %{it['benzerlik']} benzer"))

    return rapor_html, etiketler, trend_fig, hava_html, senaryo, galeri, cnn


THEME = gr.themes.Soft(primary_hue="emerald", secondary_hue="amber", neutral_hue="slate")

with gr.Blocks(title="LeadLeaf AI — Tarla 360", theme=THEME, css=CUSTOM_CSS) as demo:
    gr.HTML("""
        <div class="baslik-banner">
          <h1>🍅 LeadLeaf AI — Tarla 360</h1>
          <p>Çiftçi/tarla bilgisi + yaprak fotoğrafı → CNN sınıflandırma + RAG destekli LLM raporu +
          geçmiş trend + hava durumu riski + müdahale senaryo analizi.</p>
        </div>
        <p style="font-size:0.85rem;color:#6b7280;margin-top:6px;">
          Üretimde bu akış n8n (Telegram bot) üzerinden çalışır — bu ekran yerel görselleştirme/test amaçlıdır.
        </p>
    """)

    with gr.Group(elem_classes="kart"):
        gr.Markdown("### 👤 Çiftçi ve tarla bilgisi")
        with gr.Row():
            isim_in = gr.Textbox(label="Çiftçi adı", value="Ahmet")
            il_in = gr.Textbox(label="İl", value="Antalya")
            ilce_in = gr.Textbox(label="İlçe", value="Serik")
            urun_in = gr.Textbox(label="Ürün", value="domates")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="kart"):
                gr.Markdown("### 📸 Fotoğraf")
                img_in = gr.Image(type="pil", label="Yaprak fotoğrafı", height=280)
                btn = gr.Button("🔍 Analiz Et", variant="primary", size="lg")
                lbl_out = gr.Label(label="İlk 3 tahmin")
        with gr.Column(scale=2):
            rapor_out = gr.HTML(label="Rapor")

    with gr.Row():
        with gr.Column():
            trend_out = gr.Plot(label="📈 Geçmiş trend (bu tarla)")
        with gr.Column():
            hava_out = gr.HTML(label="Hava durumu riski")

    with gr.Group(elem_classes="kart"):
        gr.Markdown("### 🎯 Riski düşürmek için ne yapmalı? "
                    "<span style='font-size:0.8rem;color:#9ca3af;font-weight:normal;'>"
                    "(kural tabanlı senaryo — ML tahmini DEĞİL)</span>")
        senaryo_out = gr.Dataframe(
            headers=["Senaryo", "Modellenen risk", "Fark", "Not"],
            label=None,
        )

    with gr.Group(elem_classes="kart"):
        gr.Markdown("### 🖼️ Benzer referans görseller "
                    "<span style='font-size:0.8rem;color:#9ca3af;font-weight:normal;'>"
                    "(görsel RAG — modelin öğrendiği özniteliklerle en yakın örnekler)</span>")
        benzer_out = gr.Gallery(label=None, columns=3, height=180,
                                 object_fit="cover",
                                 show_label=False)

    with gr.Accordion("🔧 Ham CNN çıktısı (debug)", open=False):
        json_out = gr.JSON()

    btn.click(
        analiz_et,
        inputs=[isim_in, il_in, ilce_in, urun_in, img_in],
        outputs=[rapor_out, lbl_out, trend_out, hava_out, senaryo_out, benzer_out, json_out],
    )

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
