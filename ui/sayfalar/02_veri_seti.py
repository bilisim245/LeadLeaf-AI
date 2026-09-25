import os

import streamlit as st

from ortak import (VERI_DIR, etiket, gezinme, ornek_dosyalar, sinif_tablosu, tr_sirala, veri_var,
                   veri_yok_uyarisi)

st.title("Veri Seti: PlantVillage")
st.write("Laboratuvarda, sade bir arka plan önünde çekilmiş yaprak fotoğrafları. "
         "Her klasör bir sınıf: bitki adı + hastalık adı (ya da sağlıklı).")

df = sinif_tablosu()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Toplam görsel", f"{df['Toplam'].sum():,}".replace(",", "."))
c2.metric("Bitki", df["Bitki"].nunique())
c3.metric("Sınıf", len(df))
c4.metric("Hastalık sınıfı", int((df["Durum"] == "Hastalıklı").sum()))

if not veri_var():
    veri_yok_uyarisi()
    st.stop()

st.subheader("Görsellere bakalım")
sol, sag = st.columns([1, 3])
with sol:
    bitkiler = sorted(df["Bitki"].unique(), key=tr_sirala)
    secili_bitki = st.selectbox("Bitki", bitkiler, index=bitkiler.index("Domates"))
    alt = df[df["Bitki"] == secili_bitki].sort_values("Toplam", ascending=False)
    # key bitkiye bağlı: bitki değişince önceki bitkinin sınıf seçimi taşınmasın
    secili = st.radio("Sınıf", alt["Sınıf"].tolist(), format_func=lambda s: etiket(s).split(" — ")[1],
                      key=f"sinif_{secili_bitki}")
    if secili not in alt["Sınıf"].values:
        secili = alt["Sınıf"].iloc[0]
    adet = int(alt.loc[alt["Sınıf"] == secili, "Toplam"].iloc[0])
    st.metric("Bu sınıftaki görsel", f"{adet:,}".replace(",", "."))
    if "tohum" not in st.session_state:
        st.session_state.tohum = 0
    if st.button("Başka örnekler göster", use_container_width=True):
        st.session_state.tohum += 1
with sag:
    dosyalar = ornek_dosyalar(secili, 12, st.session_state.tohum)
    for satir in range(3):
        kolonlar = st.columns(4)
        for kolon, yol in zip(kolonlar, dosyalar[satir * 4:(satir + 1) * 4]):
            kolon.image(yol, use_container_width=True)
    st.caption(f"Klasör: {secili} · Görseller 256×256 piksel")

st.subheader("İki sınıfı yan yana karşılaştır")
st.write("Bazı hastalıklar birbirine çok benziyor. Modelin işini zorlaştıran da bu.")
siniflar = df.sort_values("Etiket")["Sınıf"].tolist()
k1, k2 = st.columns(2)
a = k1.selectbox("Birinci sınıf", siniflar, index=siniflar.index("Tomato___Early_blight"), format_func=etiket)
b = k2.selectbox("İkinci sınıf", siniflar, index=siniflar.index("Tomato___Target_Spot"), format_func=etiket)
for kolon, sinif in ((k1, a), (k2, b)):
    ic = kolon.columns(3)
    for k, yol in zip(ic, ornek_dosyalar(sinif, 3, 7)):
        k.image(yol, use_container_width=True)

with st.expander("Tüm sınıflar ve görsel sayıları"):
    st.dataframe(df[["Etiket", "Sınıf", "Toplam"]].sort_values("Toplam", ascending=False),
                 hide_index=True, use_container_width=True)
    st.caption(f"Kaynak: {os.path.relpath(VERI_DIR)} (GitHub: spMohanty/PlantVillage-Dataset, raw/color)")

gezinme(__file__)
