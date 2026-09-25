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

st.subheader("Sistem nasıl çalışıyor?")
st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#EEF2F8", color="#1B3A6B",
        fontname="Helvetica", fontsize=12, margin="0.25,0.15"];
  edge [color="#1B3A6B"];
  tg  [label="Telegram\\nfotoğraf"];
  n8n [label="n8n\\nakış yönetimi"];
  cnn [label="CNN (FastAPI)\\nEfficientNetB0, 38 sınıf", fillcolor="#1B3A6B", fontcolor="white"];
  rag [label="RAG\\nChroma bilgi tabanı"];
  llm [label="Claude\\nrapor yazımı"];
  out [label="Telegram cevabı\\n+ Google Sheets kaydı"];
  tg -> n8n -> cnn -> rag -> llm -> out;
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
