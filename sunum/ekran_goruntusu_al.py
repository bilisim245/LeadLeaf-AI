"""Streamlit panosunu sayfa sayfa, ekranda görüldüğü gibi kaydırarak çeker (sol menü kırpılır).
Önce pano açık olmalı (streamlit run ui/sunum.py). Çalıştırma: .venv\Scripts\python sunum\ekran_goruntusu_al.py
(Yüklü Chrome kullanılır; pip install playwright gerekir.)"""
import os
import shutil
import time

from playwright.sync_api import sync_playwright

CIKTI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ekran_goruntuleri")
shutil.rmtree(CIKTI, ignore_errors=True)
os.makedirs(CIKTI)
URL = "http://localhost:8501"

PLAN = [
    ("", None, "01_ozet"),
    ("veri_seti", None, "02_veri_seti"),
    ("veri_analizi", "Sınıf dağılımı", "03a_dagilim"),
    ("veri_analizi", "Eğitim / Doğrulama / Test", "03b_bolme"),
    ("veri_analizi", "Görsel özellikleri", "03c_ozellik"),
    ("veri_analizi", "Veri kalitesi", "03d_kalite"),
    ("cnn", "Evrişim (convolution)", "04a_evrisim"),
    ("cnn", "Neden transfer learning?", "04b_transfer"),
    ("cnn", "Projedeki model", "04c_model"),
    ("cnn", "Katmanlar ne görüyor?", "04d_katman"),
    ("model_secimi", None, "05a_secim"),
    ("model_secimi", "Karışıklık matrisi (test)", "05b_cm"),
    ("fine_tuning", None, "06_ft"),
    ("test_analizi", "Karışıklık matrisi", "07a_cm"),
    ("test_analizi", "Sınıf bazında", "07b_sinif"),
    ("test_analizi", "Güven ve eşik", "07c_guven"),
    ("test_analizi", "Yanlış bilinenler", "07d_yanlis"),
    ("aciklanabilirlik", None, "08_gradcam"),
    ("rag", "Bilgi tabanı", "09_rag"),
]
YUKSEKLIK, ADIM = 1000, 850


def bekle(page, ek=3):
    page.wait_for_load_state("networkidle")
    for _ in range(90):
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        time.sleep(1)
    time.sleep(ek)


with sync_playwright() as p:
    tarayici = p.chromium.launch(channel="chrome", headless=True)
    page = tarayici.new_page(viewport={"width": 1500, "height": YUKSEKLIK}, device_scale_factor=1.4)
    for yol, sekme, onek in PLAN:
        page.goto(f"{URL}/{yol}")
        bekle(page, 5)
        if sekme:
            page.get_by_role("tab", name=sekme).click()
            bekle(page)
        sol = page.locator('[data-testid="stSidebar"]').bounding_box()
        x0 = int(sol["x"] + sol["width"]) if sol else 0
        scroll = page.locator('[data-testid="stMain"]')
        onceki, i = -1, 0
        while i < 8:
            konum = page.evaluate("() => document.querySelector('[data-testid=\"stMain\"]').scrollTop")
            if konum == onceki:
                break
            page.screenshot(path=os.path.join(CIKTI, f"{onek}_{i}.png"),
                            clip={"x": x0, "y": 0, "width": 1500 - x0, "height": YUKSEKLIK})
            onceki = konum
            page.evaluate(f"() => document.querySelector('[data-testid=\"stMain\"]').scrollBy(0, {ADIM})")
            time.sleep(1.5)
            i += 1
        print(f"{onek}: {i} ekran", flush=True)
    tarayici.close()
