import os

import altair as alt
import pandas as pd
import streamlit as st

from ortak import LACIVERT, TUBITAK_DIR, gezinme

st.title("Model Karşılaştırma")
st.write("Hangi modeli kullanacağımıza tahminle karar vermedik. Üç hazır modeli **aynı veriyle, "
         "aynı ayarlarla** eğitip test setinde karşılaştırdık. Bu deneyi hız için domatesin 5 "
         "sınıfıyla yaptık, kazananı sonra 38 sınıfa taşıdık.")

df = pd.read_csv(os.path.join(TUBITAK_DIR, "model_comparison.csv"))
df["model"] = df["model"].replace({"EfficientNetB0": "EfficientNetB0 (ilk tarif)"})
gel = pd.read_csv(os.path.join(TUBITAK_DIR, "karsilastirma_gelismis.csv"))
gel["model"] = "EfficientNetB0 (fine-tuning sonrası)"
tum = pd.concat([df, gel], ignore_index=True)

st.subheader("Neden bu üç model?")
c1, c2, c3 = st.columns(3)
c1.markdown("**MobileNetV2**  \nTelefonlar için tasarlanmış, hafif ve hızlı. Sık kullanılan bir başlangıç noktası.")
c2.markdown("**MobileNetV3Small**  \nMobileNet'in daha da küçük hali. \"En küçük model ne kadar iyi olur?\" "
            "sorusunun cevabı için.")
c3.markdown("**EfficientNetB0**  \nAz parametreyle yüksek doğruluk için tasarlanmış. Literatürde bitki "
            "hastalığı sınıflandırmada iyi sonuç veriyor.")
st.caption("Üçü de ücretsiz Colab GPU'sunda makul sürede eğitilebilecek kadar küçük modeller.")

metrikler = {"Doğruluk": "dogruluk", "Macro F1": "macro_f1", "Macro Precision": "macro_precision",
             "Macro Recall": "macro_recall", "Macro AUC": "macro_auc"}
secilen = st.radio("Metrik", list(metrikler), horizontal=True)
kolon = metrikler[secilen or "Doğruluk"]
sira = tum.sort_values(kolon)["model"].tolist()
st.altair_chart(
    alt.Chart(tum).mark_bar().encode(
        x=alt.X(f"{kolon}:Q", title=secilen, scale=alt.Scale(domain=[0.8, 1.0]), axis=alt.Axis(format="%")),
        y=alt.Y("model:N", sort=sira, title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.condition(alt.datum.model == "EfficientNetB0 (fine-tuning sonrası)",
                            alt.value("#2F7D4A"), alt.value(LACIVERT)),
        tooltip=["model", alt.Tooltip(f"{kolon}:Q", format=".2%")],
    ).properties(height=230)
    + alt.Chart(tum).mark_text(align="left", dx=6, color="#0F2347").encode(
        x=f"{kolon}:Q", y=alt.Y("model:N", sort=sira), text=alt.Text(f"{kolon}:Q", format=".2%")),
    use_container_width=True,
)

sol, sag = st.columns([3, 2])
with sol:
    st.markdown("**Boyut, hız ve doğruluk birlikte**")
    st.altair_chart(alt.Chart(tum).mark_circle(opacity=0.8).encode(
        x=alt.X("model_boyutu_mb:Q", title="Model boyutu (MB)"),
        y=alt.Y("dogruluk:Q", title="Doğruluk", scale=alt.Scale(domain=[0.86, 0.99]), axis=alt.Axis(format="%")),
        size=alt.Size("ort_inference_ms:Q", title="Süre (ms)", scale=alt.Scale(range=[200, 900])),
        color=alt.Color("model:N", legend=alt.Legend(orient="bottom", columns=2, title=None)),
        tooltip=["model", alt.Tooltip("dogruluk:Q", format=".2%"), "model_boyutu_mb", "ort_inference_ms"],
    ).properties(height=320), use_container_width=True)
with sag:
    st.markdown("**Tablo**")
    goster = tum[["model", "dogruluk", "macro_f1", "macro_auc", "model_boyutu_mb", "ort_inference_ms"]].copy()
    goster.columns = ["Model", "Doğruluk", "Macro F1", "AUC", "MB", "ms/görsel"]
    st.dataframe(goster.style.format({"Doğruluk": "{:.2%}", "Macro F1": "{:.3f}", "AUC": "{:.3f}",
                                      "MB": "{:.1f}", "ms/görsel": "{:.0f}"}),
                 hide_index=True, use_container_width=True)
    st.success("EfficientNetB0 en yüksek doğruluk ve F1'i verdi. Boyutu biraz daha büyük ama modeli "
               "sunucuda çalıştırdığımız için bu bir sorun değil.")

st.subheader("Eğitim grafikleri (Colab çıktıları)")
secenek = {"MobileNetV2": "MobileNetV2", "MobileNetV3Small": "MobileNetV3Small",
           "EfficientNetB0 (ilk tarif)": "EfficientNetB0", "EfficientNetB0 (fine-tuning sonrası)": "EfficientNetB0_gelismis"}
m = st.selectbox("Model", list(secenek), index=2)
k1, k2 = st.columns(2)
for kolon_, on_ek, baslik in ((k1, "ogrenme_egrisi", "Öğrenme eğrisi: eğitim ve doğrulama"),
                              (k2, "confusion_matrix", "Karışıklık matrisi (test)")):
    yol = os.path.join(TUBITAK_DIR, f"{on_ek}_{secenek[m]}.png")
    if os.path.exists(yol):
        kolon_.image(yol, caption=baslik, use_container_width=True)
st.caption("Öğrenme eğrisinde eğitim ve doğrulama çizgileri birbirine yakın gidiyorsa model ezberlemiyor demektir. "
           "MobileNetV3Small'da fine-tuning başlayınca doğrulama düştü: küçük model bu ayara iyi tepki vermedi.")

gezinme(__file__)
