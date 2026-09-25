import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ortak import LACIVERT, M38_DIR, TUBITAK_DIR, gezinme, kod_parcasi, uretim_modeli

st.title("Fine-Tuning (İnce Ayar)")

st.write("Transfer learning'de önce hazır modelin gövdesini **donduruyoruz**, yani ağırlıklarına "
         "dokunmuyoruz ve sadece en sona eklediğimiz sınıflandırma katmanını eğitiyoruz. Fine-tuning, "
         "bundan sonra gövdenin **son katmanlarını da açıp** çok küçük adımlarla kendi verimize göre "
         "ayarlamak demek.")

c1, c2 = st.columns(2)
c1.markdown("**Neden son katmanlar?**  \nİlk katmanlar kenar ve renk gibi her görselde işe yarayan şeyler "
            "öğreniyor, onlara dokunmaya gerek yok. Son katmanlar ise ImageNet'in sınıflarına göre "
            "şekillenmiş; yaprak hastalıklarına uyması gereken kısım orası.")
c2.markdown("**Neden küçük öğrenme oranı?**  \nİlk aşamada öğrenme oranı 0,001. Fine-tuning'de 0,00001, "
            "yani 100 kat küçük. Büyük adımlarla gidersek modelin ImageNet'ten getirdiği bilgiyi birkaç "
            "adımda bozarız.")

st.subheader("Nasıl yaptık: kademeli açma")
asamalar = pd.DataFrame([
    {"Aşama": "0 · Kafa eğitimi", "Açık gövde (%)": 0, "Öğrenme oranı": "0,001", "En fazla epoch": 8},
    {"Aşama": "1 · Son %15 açık", "Açık gövde (%)": 15, "Öğrenme oranı": "0,00001", "En fazla epoch": 6},
    {"Aşama": "2 · Son %30 açık", "Açık gövde (%)": 30, "Öğrenme oranı": "0,00001", "En fazla epoch": 6},
    {"Aşama": "3 · Son %40 açık", "Açık gövde (%)": 40, "Öğrenme oranı": "0,00001", "En fazla epoch": 8},
])
model = uretim_modeli()
govde = model.get_layer("efficientnetb0")
kafa = model.count_params() - govde.count_params()
egitilen = []
for oran in (0, 0.15, 0.30, 0.40):
    n = len(govde.layers)
    acik = govde.layers[int(n * (1 - oran)):] if oran else []
    # eğitim kodundaki parametre_ozeti() ile aynı: açık katmanların eğitilebilir ağırlıkları
    # (BatchNormalization'ın hareketli ortalama/varyansı zaten eğitilebilir değil)
    egitilen.append(kafa + sum(int(np.prod(w.shape)) for l in acik for w in l.weights if w.trainable))
asamalar["Eğitilen parametre"] = egitilen

sol, sag = st.columns([3, 2])
with sol:
    st.altair_chart(alt.Chart(asamalar).mark_bar(color=LACIVERT).encode(
        x=alt.X("Eğitilen parametre:Q", title="Eğitilen parametre sayısı"),
        y=alt.Y("Aşama:N", sort=None, title=None),
        tooltip=["Aşama", "Açık gövde (%)", "Öğrenme oranı", alt.Tooltip("Eğitilen parametre:Q", format=",")],
    ).properties(height=220), use_container_width=True)
with sag:
    st.dataframe(asamalar.style.format({"Eğitilen parametre": "{:,.0f}"}), hide_index=True, use_container_width=True)
st.caption("Parametre sayıları üretim modelinin katmanlarından hesaplandı. Toplam parametre: "
           f"{model.count_params():,}".replace(",", ".") + ". Son aşamada bile gövdenin büyük kısmı sabit kalıyor.")

st.markdown("""
Her aşamada iki yardımcı var:
- **ReduceLROnPlateau:** doğrulama kaybı 2 epoch düzelmezse öğrenme oranını yarıya indiriyor.
- **EarlyStopping:** doğrulama doğruluğu 5 epoch artmazsa duruyor ve en iyi epoch'un ağırlıklarına geri dönüyor.

Bir de **BatchNormalization** tuzağı var: gövdeyi açınca bu katmanlar kendi istatistiklerini küçük
batch'lerimize göre değiştirmeye başlıyor ve model sessizce bozuluyor. Gövdeyi `training=False` ile
çağırarak bunu engelledik.
""")

st.subheader("Sonuç: aynı model, iki farklı eğitim")
ilk = pd.read_csv(os.path.join(TUBITAK_DIR, "model_comparison.csv")).query("model == 'EfficientNetB0'").iloc[0]
son = pd.read_csv(os.path.join(TUBITAK_DIR, "karsilastirma_gelismis.csv")).iloc[0]
kars = pd.DataFrame([
    {"Metrik": ad, "Eğitim": e, "Değer": v}
    for ad, k in (("Doğruluk", "dogruluk"), ("Macro F1", "macro_f1"), ("Macro Recall", "macro_recall"))
    for e, v in (("İlk tarif (son %25, sabit oran)", ilk[k]), ("Kademeli fine-tuning", son[k]))
])
k1, k2, k3 = st.columns(3)
k1.metric("Doğruluk", f"%{son['dogruluk'] * 100:.2f}".replace(".", ","), f"+{(son['dogruluk'] - ilk['dogruluk']) * 100:.2f} puan".replace(".", ","))
k2.metric("Macro F1", f"{son['macro_f1']:.3f}".replace(".", ","), f"+{son['macro_f1'] - ilk['macro_f1']:.3f}".replace(".", ","))
k3.metric("Model boyutu", f"{son['model_boyutu_mb']:.1f} MB".replace(".", ","), f"{son['model_boyutu_mb'] - ilk['model_boyutu_mb']:.1f} MB".replace(".", ","), delta_color="inverse")
st.altair_chart(alt.Chart(kars).mark_bar().encode(
    x=alt.X("Değer:Q", scale=alt.Scale(domain=[0.9, 1.0]), axis=alt.Axis(format="%"), title=None),
    y=alt.Y("Eğitim:N", title=None, axis=alt.Axis(labelLimit=260)),
    color=alt.Color("Eğitim:N", scale=alt.Scale(range=["#8FB4E8", "#2F7D4A"]), legend=None),
    row=alt.Row("Metrik:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
    tooltip=["Metrik", "Eğitim", alt.Tooltip("Değer:Q", format=".2%")],
).properties(height=70), use_container_width=False)
st.caption("Domates 5 sınıf deneyi. Bu tarifi daha sonra 38 sınıfa aynen uyguladık.")

k1, k2 = st.columns(2)
for kolon, yol, baslik in (
    (k1, os.path.join(TUBITAK_DIR, "ogrenme_egrisi_EfficientNetB0_gelismis.png"), "Kademeli fine-tuning (5 sınıf)"),
    (k2, os.path.join(M38_DIR, "ogrenme_egrisi_EfficientNetB0_38sinif.png"), "Aynı tarif, 38 sınıf"),
):
    if os.path.exists(yol):
        kolon.image(yol, caption=baslik, use_container_width=True)
st.caption("Kesikli çizgiler aşama geçişleri. Her geçişte küçük bir sıçrama var ama eğitim ve doğrulama "
           "çizgileri birbirinden ayrılmıyor.")

with st.expander("Kod: kademeli fine-tuning döngüsü"):
    st.code(kod_parcasi("notebooks/03_efficientnetb0_38_sinif.py", "for i, (oran, ust_sinir_epoch)",
                        "# %% 8)"), language="python")

gezinme(__file__)
