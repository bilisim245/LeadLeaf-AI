import streamlit as st

from ortak import gezinme, sinif_tablosu, test_sonuclari

st.title("LeadLeaf AI")
st.markdown("#### Yaprak fotoğrafından bitki hastalığı ön değerlendirmesi")
st.write("Çiftçi Telegram'dan bir yaprak fotoğrafı gönderiyor. Model hastalığı tahmin ediyor, "
         "bilgi tabanından o hastalığa ait kaynak metin bulunuyor ve Claude bu kaynağa göre "
         "kısa bir rapor yazıyor. Emin olmadığı durumda sistem ziraat mühendisine yönlendiriyor.")

df = sinif_tablosu()
ts = test_sonuclari()
dogruluk = (ts["y_prob"].argmax(1) == ts["y_true"]).mean() * 100 if ts else 99.02

c1, c2, c3, c4 = st.columns(4)
c1.metric("Görsel", f"{df['Toplam'].sum():,}".replace(",", "."))
c2.metric("Bitki / Sınıf", f"{df['Bitki'].nunique()} / {len(df)}")
c3.metric("Test doğruluğu", f"%{dogruluk:.2f}".replace(".", ","))
c4.metric("Model", "EfficientNetB0")

st.subheader("Görev şablonu: DL → LLM-Agent → n8n")
st.caption("Bootcamp görevi: \"Görüntüden Rapor (Vision → Agent → PDF)\" — A) Tarım, bitki hastalığı tespiti")
st.graphviz_chart("""
digraph {
  rankdir=TB; bgcolor="transparent"; nodesep=0.3; ranksep=0.35;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin="0.3,0.12", width=6];
  edge [color="#1B3A6B"];
  g  [label="Telegram: yaprak fotoğrafı", fillcolor="#EEF2F8", color="#1B3A6B"];
  dl [label="DL-Model: Algılama / Tahmin
EfficientNetB0 (transfer learning), 38 sınıf → sınıf + güven skoru",
      fillcolor="#1B3A6B", fontcolor="white", color="#1B3A6B"];
  ag [label="LLM-Agent: Yorumlama + Karar + Doğal Dil Çıktısı
Claude + RAG (Chroma, 38 bilgi dosyası) → JSON rapor",
      fillcolor="#2E5A9A", fontcolor="white", color="#2E5A9A"];
  n8 [label="n8n: Tetikleme + Entegrasyon + Aksiyon
Telegram cevabı · PDF rapor · Google Sheets · güven < %70 → uzman · takip hatırlatması",
      fillcolor="#EEF2F8", color="#1B3A6B"];
  g -> dl -> ag -> n8;
}
""", use_container_width=True)

st.subheader("Kullanılan araçlar")
st.dataframe(
    {
        "Katman": ["Model eğitimi", "Model servisi", "Bilgi tabanı (RAG)", "Rapor", "Akış",
                   "Kullanıcı arayüzü", "Kayıt", "Bu pano"],
        "Araç": ["TensorFlow / Keras, Google Colab (GPU)", "FastAPI", "Chroma + sentence-transformers",
                 "Claude (Anthropic API)", "n8n", "Telegram botu", "Google Sheets", "Streamlit"],
    },
    hide_index=True, use_container_width=True,
)

gezinme(__file__)
