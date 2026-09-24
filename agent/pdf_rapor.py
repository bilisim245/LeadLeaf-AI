"""
PDF rapor üretimi — Gün 7 (n8n, Claude raporunu PDF'e çevirip Telegram'dan
"belge" olarak gönderiyor).

Neden fpdf2: sistemde ekstra bir kurulum (Pango/Cairo gibi) gerektirmeyen, saf
Python bir kütüphane — Windows'ta sorunsuz çalışır. Türkçe karakterler (ğ, ş,
ı, ç, ö, ü) için sistemin Arial TTF fontu embed ediliyor; fpdf2'nin varsayılan
"core" fontları (Helvetica/Times) bu karakterleri desteklemiyor.
"""
from __future__ import annotations

import os

from fpdf import FPDF

# Windows'un kendi Arial fontu — Türkçe karakterleri tam destekliyor, ekstra
# font dosyası indirmeye/lisanslamaya gerek yok (bu proje sadece bu makinede
# demo edileceği için taşınabilirlik burada öncelik değil).
_FONT_REGULAR = r"C:\Windows\Fonts\arial.ttf"
_FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"


_FONT_ADI = "Helvetica"


def _pdf_hazirla() -> FPDF:
    global _FONT_ADI
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    if os.path.exists(_FONT_REGULAR):
        pdf.add_font("Arial", "", _FONT_REGULAR)
        pdf.add_font("Arial", "B", _FONT_BOLD if os.path.exists(_FONT_BOLD) else _FONT_REGULAR)
        _FONT_ADI = "Arial"
    else:
        # Font bulunamazsa (farklı bir işletim sisteminde çalıştırılırsa) çökmek yerine
        # Türkçe karaktersiz bir çekirdek fonta düş — sistem yine de bir PDF üretir.
        _FONT_ADI = "Helvetica"
    pdf.set_font(_FONT_ADI, "", 11)
    return pdf


def _baslik(pdf: FPDF, metin: str) -> None:
    pdf.set_font(_FONT_ADI, "B", 13)
    pdf.multi_cell(0, 8, metin, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(_FONT_ADI, "", 11)
    pdf.ln(1)


def _govde(pdf: FPDF, metin: str) -> None:
    pdf.multi_cell(0, 6, metin, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def rapor_pdf_olustur(rapor: dict, hastalik_tr: str | None = None, tarih: str | None = None) -> bytes:
    """`rapor` = Claude'un ürettiği JSON (hastalik, guven, neden, aciklama, onlem,
    uzmana_yonlendir, uyari). `hastalik_tr`/`tarih` verilirse başlıkta ayrıca gösterilir
    (n8n'den CNN'in ham çıktısı ve zaman damgasıyla birlikte çağrılabilir)."""
    pdf = _pdf_hazirla()

    _baslik(pdf, "LeadLeaf AI — Bitki Hastalığı Ön Değerlendirme Raporu")
    if tarih:
        _govde(pdf, f"Tarih: {tarih}")

    _baslik(pdf, f"Tespit: {rapor.get('hastalik', hastalik_tr or '-')}")
    _govde(pdf, f"Güven düzeyi: %{rapor.get('guven', '-')}")

    if rapor.get("neden"):
        _baslik(pdf, "Neden Oluyor?")
        _govde(pdf, rapor["neden"])

    if rapor.get("aciklama"):
        _baslik(pdf, "Değerlendirme")
        _govde(pdf, rapor["aciklama"])

    onlem = rapor.get("onlem") or []
    if onlem:
        _baslik(pdf, "Önerilen Önlemler")
        for madde in onlem:
            pdf.multi_cell(0, 6, f"- {madde}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    if rapor.get("uzmana_yonlendir"):
        _baslik(pdf, "Uzmana Danışın")
        _govde(pdf, "Bu sonuç kesin değil, bir ziraat mühendisine danışmanızı öneririz.")

    if rapor.get("uyari"):
        pdf.set_font(_FONT_ADI, "", 9)
        _govde(pdf, rapor["uyari"])
        pdf.set_font(_FONT_ADI, "", 11)

    return bytes(pdf.output())
