import json
import os

import altair as alt
import pandas as pd
import streamlit as st

from ortak import (ACIK, DURUM_RENK, LACIVERT, M38_DIR, VERI_DIR, etiket, gezinme, manifest, nasil_okunur,
                   sinif_tablosu)

st.title("Keşifsel Veri Analizi")

df = sinif_tablosu()
yol = os.path.join(M38_DIR, "veri_analizi.json")
va = json.load(open(yol, encoding="utf-8")) if os.path.exists(yol) else None

t1, t2, t3, t4 = st.tabs(["Sınıf dağılımı", "Eğitim / Doğrulama / Test", "Görsel özellikleri", "Veri kalitesi"])

with t1:
    en_kucuk, en_buyuk = df.loc[df["Toplam"].idxmin()], df.loc[df["Toplam"].idxmax()]
    c1, c2, c3 = st.columns(3)
    c1.metric("En büyük sınıf", f"{en_buyuk['Toplam']:,}".replace(",", "."), en_buyuk["Etiket"], delta_color="off")
    c2.metric("En küçük sınıf", en_kucuk["Toplam"], en_kucuk["Etiket"], delta_color="off")
    c3.metric("Oran", f"{en_buyuk['Toplam'] / en_kucuk['Toplam']:.0f} kat")

    sol, sag = st.columns([2, 3])
    with sol:
        st.markdown("**Bitkilere göre**")
        bdf = df.groupby(["Bitki", "Durum"], as_index=False)["Toplam"].sum()
        sira = bdf.groupby("Bitki")["Toplam"].sum().sort_values(ascending=False).index.tolist()
        st.altair_chart(alt.Chart(bdf).mark_bar().encode(
            x=alt.X("Toplam:Q", title="Görsel"), y=alt.Y("Bitki:N", sort=sira, title=None),
            color=alt.Color("Durum:N", scale=DURUM_RENK, legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Bitki", "Durum", alt.Tooltip("Toplam:Q", format=",")],
        ).properties(height=430), use_container_width=True)
        pasta = df.groupby("Durum", as_index=False)["Toplam"].sum()
        st.altair_chart(alt.Chart(pasta).mark_arc(innerRadius=55).encode(
            theta="Toplam:Q", color=alt.Color("Durum:N", scale=DURUM_RENK, legend=alt.Legend(orient="right", title=None)),
            tooltip=["Durum", alt.Tooltip("Toplam:Q", format=",")],
        ).properties(height=200), use_container_width=True)
        nasil_okunur(
            "Her bitkinin veri setindeki görsel sayısı. Koyu lacivert hastalıklı, açık mavi sağlıklı yaprakları "
            "gösterir. Halka grafik tüm veride hastalıklı ve sağlıklı oranını verir.",
            "Çubuk ne kadar uzunsa o bitkiden o kadar çok görsel vardır. Tamamı açık mavi olan bitkilerde "
            "(soya, ahududu, yaban mersini) hastalık sınıfı yoktur.",
            "Veri bitkiler arasında eşit dağılmamıştır: domates 18.160 görselle en büyük bitkidir, ahududu yalnızca "
            "371 görseldir. Görsellerin %72'si hastalıklı, %28'i sağlıklı yapraktır.")
    with sag:
        st.markdown("**Sınıflara göre** (bir bitkiye tıklandığında o bitkinin sınıfları vurgulanır)")
        secim = alt.selection_point(fields=["Bitki"])
        st.altair_chart(alt.Chart(df).mark_bar().encode(
            x=alt.X("Toplam:Q", title="Görsel"),
            y=alt.Y("Etiket:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.condition(secim, alt.Color("Durum:N", scale=DURUM_RENK, legend=None), alt.value("#E3E8F0")),
            tooltip=["Etiket", "Sınıf", alt.Tooltip("Toplam:Q", format=",")],
        ).add_params(secim).properties(height=760), use_container_width=True)
        nasil_okunur(
            "38 sınıfın her birindeki görsel sayısı, büyükten küçüğe sıralı.",
            "En üstteki en kalabalık, en alttaki en az görsele sahip sınıftır. Soldaki grafikte bir bitkiye "
            "tıklanınca yalnızca o bitkinin sınıfları renkli kalır.",
            "En büyük sınıf (portakal HLB, 5.507) en küçükten (sağlıklı patates, 152) yaklaşık 36 kat büyüktür. "
            "Bu dengesizlik yüzünden yalnızca doğruluğa değil, her sınıfı eşit sayan macro F1 skoruna da bakılmıştır.")
    st.info("Sınıflar dengesiz. Bu yüzden sadece doğruluğa değil, her sınıfı eşit sayan "
            "**macro F1** skoru da raporlanmıştır.")

with t2:
    m = manifest()
    st.write(f"Veri, eğitimden önce **bir kez** bölünmüştür: her sınıfın kendi içinde %70 eğitim, "
             f"%15 doğrulama, %15 test. Rastgelelik sabit (seed = {m['seed']}), yani aynı bölme "
             "tekrar üretilebilmektedir. Hangi dosyanın hangi kümeye gittiği bir dosyada kayıtlıdır.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Eğitim", f"{df['Eğitim'].sum():,}".replace(",", "."), "modelin öğrendiği", delta_color="off")
    c2.metric("Doğrulama", f"{df['Doğrulama'].sum():,}".replace(",", "."), "eğitim sırasında kontrol", delta_color="off")
    c3.metric("Test", f"{df['Test'].sum():,}".replace(",", "."), "en sonda, bir kez", delta_color="off")
    uzun = df.melt(id_vars=["Etiket"], value_vars=["Eğitim", "Doğrulama", "Test"], var_name="Küme", value_name="Adet")
    st.altair_chart(alt.Chart(uzun).mark_bar().encode(
        x=alt.X("Adet:Q", stack="normalize", title="Pay", axis=alt.Axis(format="%")),
        y=alt.Y("Etiket:N", title=None, sort=df.sort_values("Toplam", ascending=False)["Etiket"].tolist(),
                axis=alt.Axis(labelLimit=260)),
        color=alt.Color("Küme:N", scale=alt.Scale(domain=["Eğitim", "Doğrulama", "Test"],
                                                  range=[LACIVERT, "#2E5A9A", ACIK]),
                        legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("Küme:N", sort="descending"),
        tooltip=["Etiket", "Küme", "Adet"],
    ).properties(height=760), use_container_width=True)
    st.caption("Her sınıfta oranlar aynı (stratified bölme). Küçük sınıflar da testte temsil ediliyor.")
    nasil_okunur(
        "Her sınıfın görsellerinin eğitim, doğrulama ve test kümelerine nasıl paylaştırıldığı (yüzde olarak).",
        "Her çubuk bir sınıftır ve %100'e tamamlanır. Renklerin sınırları bütün çubuklarda aynı hizadaysa bölme "
        "her sınıfta aynı oranda yapılmış demektir.",
        "Bütün sınıflar %70 eğitim, %15 doğrulama, %15 test olarak bölünmüştür. Böylece 152 görsellik en küçük "
        "sınıf bile testte temsil edilmektedir.")

with t3:
    if not va:
        st.info("Önce `notebooks/05_yerel_veri_analizi.py` çalıştırılmalı.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Görsel boyutları (54.305 görselin hepsi)**")
            st.dataframe(pd.DataFrame([{"Boyut": k, "Adet": v} for k, v in va["boyutlar"].items()]),
                         hide_index=True, use_container_width=True)
            st.caption("Hepsi aynı boyutta. Modele girmeden önce 224×224'e küçültülüyor.")
        with c2:
            renk = pd.DataFrame(va["renk_ornegi"])
            renk["Etiket"] = renk["sinif"].map(etiket)
            renk["Durum"] = renk["sinif"].str.endswith("___healthy").map({True: "Sağlıklı", False: "Hastalıklı"})
            st.markdown("**Parlaklık dağılımı** (her sınıftan 60 görsel)")
            st.altair_chart(alt.Chart(renk).mark_boxplot(size=10).encode(
                x=alt.X("parlaklik:Q", title="Ortalama parlaklık (0-255)", scale=alt.Scale(zero=False)),
                y=alt.Y("Etiket:N", title=None, sort=alt.EncodingSortField("parlaklik", op="median"),
                        axis=alt.Axis(labelLimit=260)),
                color=alt.Color("Durum:N", scale=DURUM_RENK, legend=alt.Legend(orient="top", title=None)),
            ).properties(height=760), use_container_width=True)
            nasil_okunur(
                "Her sınıftan 60 görselin ortalama parlaklığı (0 siyah, 255 beyaz).",
                "Kutu, görsellerin ortadaki yarısını; kutunun içindeki çizgi ortanca değeri; bıyıklar geri kalanını "
                "gösterir. Kutular birbirinden uzaksa sınıfların ışık koşulları farklıdır.",
                "Sınıflar arasında parlaklık farkı vardır: en koyu mısır pası (ortanca 89), en parlak sağlıklı mısır "
                "(ortanca 142). Model rengi ve ışığı da öğrenebileceği için tarla fotoğraflarında bu fark sorun "
                "yaratabilir.")
        st.markdown("**Renk: yeşil kanal ile kırmızı kanal**")
        st.altair_chart(alt.Chart(renk).mark_circle(size=40, opacity=0.55).encode(
            x=alt.X("R:Q", title="Ortalama kırmızı", scale=alt.Scale(zero=False)),
            y=alt.Y("G:Q", title="Ortalama yeşil", scale=alt.Scale(zero=False)),
            color=alt.Color("Durum:N", scale=DURUM_RENK, legend=alt.Legend(orient="top", title=None)),
            tooltip=["Etiket", "R", "G", "B"],
        ).properties(height=380).interactive(), use_container_width=True)
        nasil_okunur(
            "Her nokta bir görsel. Yatay eksen ortalama kırmızı, dikey eksen ortalama yeşil değeridir.",
            "Sağ üste gidildikçe görsel daha parlak, yukarı gidildikçe daha yeşildir. Grafik yakınlaştırılabilir.",
            "Hastalıklı yapraklar ortalamada daha koyu ve daha az yeşildir: yeşil kanal 122'ye karşı 137, kırmızı "
            "kanal 117'ye karşı 127. Renk tek başına hastalığı ayırmaya yetmez; iki grup büyük ölçüde iç içedir.")

with t4:
    if not va:
        st.info("Önce `notebooks/05_yerel_veri_analizi.py` çalıştırılmalı.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Birebir aynı görsel grubu", va["tekrar_grubu"])
        c2.metric("Fazladan kopya", va["tekrar_eden_fazla_kopya"])
        c3.metric("Farklı sınıfta aynı görsel", va["farkli_sinifta_tekrar"])
        st.write("Her dosyanın içeriğinden bir özet (MD5) çıkarılıp karşılaştırılmıştır. Aynı özete sahip "
                 "iki dosya birebir aynı fotoğraftır. Aynı fotoğrafın bir kopyası eğitimde, "
                 "diğeri testte olursa model onu ezberden bilir ve sonuç olduğundan iyi görünür.")
        if "farkli_kumede_tekrar" in va:
            n_test = va["egitimde_kopyasi_olan_test"]
            st.warning(
                f"**Bulgu:** {va['tekrar_grubu']} tekrar grubundan {va['farkli_kumede_tekrar']} tanesinin "
                f"kopyaları farklı kümelere düşmüş. Bunlardan **{n_test} test görselinin** birebir aynısı "
                f"eğitim kümesinde var. Yani model bu {n_test} görseli aslında eğitimde gördü. 8.146 test "
                f"görselinin içinde bu en fazla %{n_test / 8146 * 100:.2f}'lik bir etki; sonucu "
                "değiştirmiyor ama bölmeden önce tekrarları temizlemek daha doğru olurdu."
            )
            with st.expander("Farklı kümelere düşen kopyalar"):
                st.dataframe(
                    [{"Grup": i + 1, "Dosya": x["dosya"], "Küme": x["kume"]}
                     for i, g in enumerate(va["farkli_kumede_ornekler"]) for x in g],
                    hide_index=True, use_container_width=True,
                )
        st.caption("Bozuk ya da açılamayan görsel yok: 54.305 görselin hepsi tam olarak açılıp okundu.")
        farkli = [f"{c}/{d}" for c in sorted(os.listdir(VERI_DIR)) if os.path.isdir(os.path.join(VERI_DIR, c))
                  for d in os.listdir(os.path.join(VERI_DIR, c))
                  if not d.lower().endswith((".jpg", ".jpeg"))] if os.path.isdir(VERI_DIR) else []
        if farkli:
            st.warning(f"**Bulgu:** {len(farkli)} dosya fotoğraf değil, ekran görüntüsü (PNG): "
                       f"`{farkli[0]}`. Veri setine yanlışlıkla girmiş ve test kümesine düşmüş. Model bunu %69,1 "
                       "güvenle 'domates erken yanıklık' sandı; güven %70'in altında olduğu için bot uzmana "
                       "yönlendirirdi.")
            st.image(os.path.join(VERI_DIR, farkli[0]), width=260)

gezinme(__file__)
