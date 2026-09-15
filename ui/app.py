"""
Gradio Demo Arayüzü — jüriye/hızlı teste yerel görselleştirme (README'de "opsiyonel").

Akış: görsel yükle → inference/app.py'ye (/predict) HTTP ile gönder → CNN sonucu al →
agent/report.py ile (yerel demo LLM çağrısı) raporu üret → ekranda göster.

n8n (local ya da n8n Cloud) ile YARIŞMAZ — o, Telegram bot üretim akışıdır. Bu arayüz,
n8n/Telegram henüz bağlanmadan projenin uçtan uca çalıştığını GÖRSEL olarak kanıtlamak
ve inference servisini n8n olmadan da test edebilmek içindir.

Çalıştırma (önce inference servisini ayrı bir terminalde başlat):
    .venv\\Scripts\\python.exe -m uvicorn inference.app:app --port 8000
    .venv\\Scripts\\python.exe ui/app.py
"""
from __future__ import annotations

import os
import sys

# proje kökünü sys.path'e ekle (dosya doğrudan `python ui/app.py` ile çalıştırılınca
# `agent` paketi bulunamıyor — `python -m ui.app` ile çalıştırılırsa buna gerek kalmaz)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import requests
from dotenv import load_dotenv

from agent.report import generate_report

load_dotenv()

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8000")


def analiz_et(image):
    if image is None:
        return "Lütfen bir yaprak fotoğrafı yükleyin.", {}, ""

    try:
        # PIL Image -> bytes (JPEG)
        import io

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
        return hata, {}, ""
    except Exception as e:
        return f"❌ Hata: {e}", {}, ""

    rapor = generate_report(cnn["hastalik"], cnn["hastalik_tr"], cnn["guven"])

    etiketler = {it["sinif_tr"]: it["olasilik"] / 100 for it in cnn["ilk3"]}

    demo_uyarisi = (
        "\n\n> ⚠️ **DEMO MODU:** Gerçek model henüz `model/` klasörüne konmadı "
        "(Kaggle eğitimi tamamlanmadı). Sınıflandırma sonucu RASTGELE üretildi; "
        "aşağıdaki LLM raporu bu rastgele sonuca göre yazılmıştır.\n"
        if cnn.get("demo_mode") else ""
    )

    kaynak_notu = (
        "\n\n*(Rapor: yerel şablon — `ANTHROPIC_API_KEY` .env'de tanımlı değil.)*"
        if rapor.get("_kaynak") == "sablon" else ""
    )

    md = f"""{demo_uyarisi}
## 🔎 Tespit: {cnn['hastalik_tr']}
**Güven:** %{cnn['guven']}

{"### ⚠️ Bu sonuç kesin değil — bir ziraat mühendisine danışmanızı öneririz." if cnn['uzmana_yonlendir'] else ""}

### Açıklama
{rapor.get('aciklama', '-')}

### Önerilen kültürel/biyolojik önlemler
{rapor.get('onlem', '-')}

---
*{rapor.get('uyari', 'Bu bir ön değerlendirmedir, kesin teşhis değildir.')}*
{kaynak_notu}
"""

    return md, etiketler, cnn


with gr.Blocks(title="LeadLeaf AI — Demo") as demo:
    gr.Markdown(
        "# 🍅 LeadLeaf AI — Bitki Hastalığı Ön Değerlendirme (Yerel Demo)\n"
        "Bir domates yaprağı fotoğrafı yükleyin. CNN modeli hastalığı sınıflandırır, "
        "LLM-Agent bunu anlaşılır bir öneriye çevirir.\n\n"
        "*Üretimde bu akış n8n (Telegram bot) üzerinden çalışır — bu ekran sadece "
        "yerel görselleştirme/test amaçlıdır.*"
    )
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Yaprak fotoğrafı")
            btn = gr.Button("Analiz Et", variant="primary")
        with gr.Column():
            lbl_out = gr.Label(label="İlk 3 tahmin")
            md_out = gr.Markdown(label="Rapor")
    with gr.Accordion("Ham CNN çıktısı (debug)", open=False):
        json_out = gr.JSON()

    btn.click(analiz_et, inputs=img_in, outputs=[md_out, lbl_out, json_out])

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
