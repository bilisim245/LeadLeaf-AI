import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ortak import LACIVERT, TUBITAK_DIR, gezinme, nasil_okunur

st.title("Model Karşılaştırma")
st.write("Model seçimi tahmine göre yapılmamıştır. Üç hazır model **aynı veriyle, "
         "aynı ayarlarla** eğitilip test setinde karşılaştırılmıştır. Bu deney hız için domatesin 5 "
         "sınıfıyla yapılmış, kazanan model daha sonra 38 sınıfa taşınmıştır.")

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
tum["Vurgu"] = np.where(tum["model"] == "EfficientNetB0 (fine-tuning sonrası)", "Seçilen", "Diğer")
taban = alt.Chart(tum).encode(
    y=alt.Y("model:N", sort=sira, title=None, axis=alt.Axis(labelLimit=320)),
    x=alt.X(f"{kolon}:Q", title=secilen, scale=alt.Scale(domain=[0.8, 1.0], clamp=True),
            axis=alt.Axis(format="%")),
    tooltip=["model", alt.Tooltip(f"{kolon}:Q", format=".2%")],
)
st.altair_chart(
    alt.layer(
        taban.mark_bar(clip=True).encode(
            color=alt.Color("Vurgu:N", scale=alt.Scale(domain=["Seçilen", "Diğer"], range=["#2F7D4A", LACIVERT]),
                            legend=None)),
        taban.mark_text(align="right", dx=-8, color="white", fontWeight="bold").encode(
            text=alt.Text(f"{kolon}:Q", format=".2%")),
    ).properties(height=240),
    use_container_width=True,
)
st.caption("Eksen %80'den başlıyor; farklar daha net görünsün diye. Yeşil çubuk üretimde kullanılan eğitim tarifi.")
nasil_okunur(
    "Üç modelin ve fine-tuning sonrası EfficientNetB0'ın test setindeki başarısı. Üstteki seçenekle metrik değiştirilebilir.",
    "Çubuk ne kadar uzunsa model o kadar başarılıdır. Eksen %80'den başladığı için küçük farklar büyük görünür; gerçek değerler çubuk içindeki yüzdelerdir.",
    "Aynı veri ve aynı ayarlarla en iyi sonucu EfficientNetB0 vermiştir (%94,68). Fine-tuning tarifinin geliştirilmesiyle %97,62'ye çıkmıştır; bu yüzden üretimde bu model kullanılmaktadır.")

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
    nasil_okunur(
        "Her balon bir model. Yatay eksen boyut (MB), dikey eksen doğruluk, balonun büyüklüğü bir görselin işlenme süresi.",
        "Sol üst köşe hem küçük hem doğru demektir. Sağ üst doğru ama büyük, sol alt küçük ama daha az doğrudur.",
        "MobileNetV3Small en küçük (9,7 MB) ama en az doğru modeldir. EfficientNetB0 biraz daha büyüktür ama en doğrusudur; sunucuda çalıştığı için boyut farkı önemli değildir. Süreler birbirine yakındır (~90 ms).")
with sag:
    st.markdown("**Tablo**")
    goster = tum[["model", "dogruluk", "macro_f1", "macro_auc", "model_boyutu_mb", "ort_inference_ms"]].copy()
    goster.columns = ["Model", "Doğruluk", "Macro F1", "AUC", "MB", "ms/görsel"]
    st.dataframe(goster.style.format({"Doğruluk": "{:.2%}", "Macro F1": "{:.3f}", "AUC": "{:.3f}",
                                      "MB": "{:.1f}", "ms/görsel": "{:.0f}"}),
                 hide_index=True, use_container_width=True)
    st.success("EfficientNetB0 en yüksek doğruluk ve F1'i verdi. Boyutu biraz daha büyük ama modeli "
               "sunucuda çalıştığı için bu bir sorun oluşturmamaktadır.")

st.subheader("Eğitim grafikleri (Colab çıktıları)")
secenek = {"MobileNetV2": "MobileNetV2", "MobileNetV3Small": "MobileNetV3Small",
           "EfficientNetB0 (ilk tarif)": "EfficientNetB0", "EfficientNetB0 (fine-tuning sonrası)": "EfficientNetB0_gelismis"}
m = st.selectbox("Model", list(secenek), index=2)
egri = os.path.join(TUBITAK_DIR, f"ogrenme_egrisi_{secenek[m]}.png")
cm = os.path.join(TUBITAK_DIR, f"confusion_matrix_{secenek[m]}.png")
t_egri, t_cm = st.tabs(["Öğrenme eğrisi (eğitim ve doğrulama)", "Karışıklık matrisi (test)"])
with t_egri:
    if os.path.exists(egri):
        st.image(egri, use_container_width=True)
with t_cm:
    if os.path.exists(cm):
        orta = st.columns([1, 2, 1])[1]
        orta.image(cm, use_container_width=True)
st.caption("Öğrenme eğrisinde eğitim ve doğrulama çizgilerinin birbirine yakın gitmesi, modelin ezberlemediğini "
           "gösterir. MobileNetV3Small'da fine-tuning başlayınca doğrulama doğruluğu düşmüştür: küçük model bu ayara "
           "iyi tepki vermemiştir.")
nasil_okunur(
    "Öğrenme eğrisi: her epoch sonunda eğitim (mavi) ve doğrulama (turuncu) doğruluğu ile kaybı. Karışıklık matrisi: test görsellerinin hangi sınıfa tahmin edildiği.",
    "Eğrideki kesikli çizgi fine-tuning'in başladığı yerdir. İki çizginin birbirine yakın gitmesi ezberlemenin olmadığını gösterir. Matriste koyu köşegen doğru tahminlerdir; köşegen dışındaki renkler karışmalardır.",
    "EfficientNetB0'da fine-tuning başlayınca eğitim doğruluğu kısa bir süre düşüp toparlanmış, doğrulama çizgisi eğitim çizgisinden ayrılmamıştır. MobileNetV3Small'da ise fine-tuning sonrası doğrulama düşmüştür.")

gezinme(__file__)
