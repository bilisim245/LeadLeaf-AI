"""
LeadLeaf AI — sunum panosu.

Çalıştırma (önce inference servisi açık olmalı, Canlı Demo sayfası onu kullanıyor):
    .venv\\Scripts\\python -m uvicorn inference.app:app --port 8000
    .venv\\Scripts\\python -m streamlit run ui/sunum.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st  # noqa: E402

from ortak import SAYFALAR, STIL, uretim_modeli  # noqa: E402

st.set_page_config(page_title="LeadLeaf AI", page_icon="🌿", layout="wide")
st.session_state["sunum_modu"] = True  # app.py (Canlı Demo) sadece analiz akışını göstersin
st.logo(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png"), size="large")
st.markdown(STIL, unsafe_allow_html=True)
st.sidebar.caption("Miuul Bootcamp · DL + LLM-Agent + n8n · 2026")

# Model bir kez, pano açılırken yüklensin; sunum ortasında CNN sayfasında beklemeyelim
with st.spinner("Model yükleniyor..."):
    uretim_modeli()

sayfa = {yol: st.Page(yol, title=baslik, icon=ikon) for yol, baslik, ikon in SAYFALAR}
gruplar = {
    "Proje": ["sayfalar/01_ozet.py"],
    "Veri": ["sayfalar/02_veri_seti.py", "sayfalar/03_veri_analizi.py"],
    "Model": ["sayfalar/04_cnn.py", "sayfalar/05_model_secimi.py", "sayfalar/06_fine_tuning.py",
              "sayfalar/07_test_analizi.py", "sayfalar/08_aciklanabilirlik.py"],
    "Sistem": ["sayfalar/09_rag.py", "app.py"],
    "Sonuç": ["sayfalar/10_sinirliliklar.py"],
}
st.navigation({g: [sayfa[y] for y in yollar] for g, yollar in gruplar.items()}, expanded=True).run()
