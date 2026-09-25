import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy import ndimage

from ortak import (KOK, etiket, gezinme, kod_parcasi, nasil_okunur, ornek_dosyalar, sinif_tablosu, uretim_modeli,
                   veri_var)

st.title("CNN ve Transfer Learning")

siniflar = sinif_tablosu().sort_values("Etiket")["Sınıf"].tolist()


def gorsel_sec(anahtar: str, varsayilan: str = "Tomato___Early_blight") -> Image.Image:
    if veri_var():
        sinif = st.selectbox("Yaprak", siniflar, index=siniflar.index(varsayilan), format_func=etiket, key=anahtar)
        return Image.open(ornek_dosyalar(sinif, 1, 3)[0]).convert("RGB")
    return Image.open(os.path.join(KOK, "model", "demo_images", "Tomato___Early_blight.jpg")).convert("RGB")


t1, t2, t3, t4 = st.tabs(["Evrişim (convolution)", "Neden transfer learning?", "Projedeki model", "Katmanlar ne görüyor?"])

with t1:
    st.write("CNN'in temel işlemi: küçük bir filtre (3×3 sayı) görselin üzerinde kaydırılır. Her konumda "
             "filtre ile altındaki pikseller çarpılıp toplanır. Filtre neyi arıyorsa o bölgeler parlar. "
             "Aşağıda bir filtre seçilerek ya da sayılar değiştirilerek sonuç görülebilir.")
    hazir = {
        "Dikey kenar": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        "Yatay kenar": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        "Keskinleştirme": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        "Bulanıklaştırma": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    }
    sol, orta, sag = st.columns([1, 1, 1])
    with sol:
        img = gorsel_sec("evrisim")
        secim = st.radio("Hazır filtre", list(hazir), horizontal=False)
        filtre = st.data_editor(pd.DataFrame(hazir[secim], columns=["1", "2", "3"]),
                                hide_index=True, key=f"filtre_{secim}", use_container_width=True)
    gri = np.asarray(img.convert("L"), dtype=np.float32)
    k = filtre.to_numpy(dtype=np.float32)
    if secim == "Bulanıklaştırma":
        k = k / max(k.sum(), 1)
    sonuc = ndimage.convolve(gri, k, mode="reflect")
    if secim in ("Dikey kenar", "Yatay kenar"):
        sonuc = np.abs(sonuc)
    sonuc = (sonuc - sonuc.min()) / (np.ptp(sonuc) + 1e-6) * 255
    orta.image(img, caption="Orijinal", use_container_width=True)
    sag.image(sonuc.astype(np.uint8), caption=f"Filtre sonrası: {secim}", use_container_width=True)
    st.caption("Gerçek bir CNN'de bu filtreler elle yazılmaz; model eğitim sırasında bunları kendisi öğrenir. "
               "İlk katmanlar kenar ve renk geçişi gibi basit şeyler, son katmanlar leke ve doku gibi "
               "daha karmaşık şeyler öğreniyor.")

    nasil_okunur(
        "Seçilen yaprak (ortada) ve üzerine 3×3'lük bir filtre uygulanmış hâli (sağda).",
        "Sağdaki görselde parlak yerler filtrenin aradığı desenin bulunduğu yerlerdir. Dikey kenar filtresi dikey çizgileri, yatay kenar filtresi yatay çizgileri parlatır.",
        "CNN'in temel işlemi budur: küçük bir filtre görselin üzerinde gezer ve belirli desenleri öne çıkarır. Gerçek modelde bu filtreler elle yazılmaz, eğitim sırasında öğrenilir.")
with t2:
    st.write("İki yol vardı: sıfırdan bir CNN yazıp eğitmek ya da daha önce milyonlarca görselle "
             "eğitilmiş bir modeli alıp bu veriye uyarlamak (transfer learning). Projede ikincisi tercih edilmiştir.")
    st.table(pd.DataFrame({
        "": ["Başlangıç", "Gereken veri", "Eğitim süresi", "Ezberleme riski", "Projenin koşulları"],
        "Sıfırdan CNN": ["Rastgele ağırlıklar, hiçbir şey bilmiyor", "Çok fazla", "Uzun",
                         "Yüksek", "Colab'ın ücretsiz GPU'su ve kısıtlı süre"],
        "Transfer learning": ["ImageNet'te (1,28 milyon görsel, 1000 sınıf) eğitilmiş",
                              "Daha az yeter", "Kısa", "Daha düşük",
                              "Kenar, doku, şekil bilgisi hazır geliyor"],
    }))
    st.write("Yaprak da sonuçta bir görsel. Kenar, doku, renk geçişi gibi temel özellikleri ImageNet'te "
             "öğrenmiş bir model bunları yaprakta da kullanabilir. Bu projede yalnızca son kısım 38 "
             "sınıfa göre eğitilmiştir.")
    st.info("Not: Sıfırdan eğitilen bir CNN ile ayrıca karşılaştırma yapılmamıştır. Bu bir sonraki adım olarak "
            "yapılabilir; literatürde PlantVillage'da transfer learning'in sıfırdan eğitime göre "
            "daha iyi sonuç verdiği biliniyor.")

with t3:
    model = uretim_modeli()
    govde = model.get_layer("efficientnetb0")
    kafa_param = model.count_params() - govde.count_params()
    c1, c2, c3 = st.columns(3)
    c1.metric("Gövde (EfficientNetB0) katmanı", len(govde.layers))
    c2.metric("Gövde parametresi", f"{govde.count_params():,}".replace(",", "."), "ImageNet'ten hazır", delta_color="off")
    c3.metric("Eklenen sınıflandırma katmanı", f"{kafa_param:,}".replace(",", "."), "38 sınıf için", delta_color="off")
    st.graphviz_chart("""
    digraph {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#EEF2F8", color="#1B3A6B", fontname="Helvetica", fontsize=11];
      a [label="Girdi\\n224×224×3"];
      b [label="Veri artırma\\n(çevirme, döndürme,\\nyakınlaştırma, kontrast)\\nsadece eğitimde"];
      c [label="EfficientNetB0 gövdesi\\n238 katman\\nImageNet ağırlıkları", fillcolor="#1B3A6B", fontcolor="white"];
      d [label="Global Average\\nPooling\\n7×7×1280 → 1280"];
      e [label="Dropout 0.2"];
      f [label="Dense 38\\nsoftmax\\n(38 olasılık)"];
      a -> b -> c -> d -> e -> f;
    }
    """, use_container_width=True)
    st.write("Softmax her sınıf için bir olasılık veriyor, toplamları 1. En yüksek olan tahmin, "
             "o olasılık da **güven** değeri.")
    with st.expander("Kod: modelin kurulduğu bölüm (notebooks/03_efficientnetb0_38_sinif.py)"):
        st.code(kod_parcasi("notebooks/03_efficientnetb0_38_sinif.py", "inputs = keras.Input", "def parametre_ozeti"),
                language="python")

with t4:
    st.write("Bir yaprak modele verilip gövdenin farklı derinliklerindeki çıktılar gösterilmektedir. "
             "Her küçük kare bir filtrenin tepkisi: parlak yerler, o filtrenin \"bir şey bulduğu\" yerler.")
    img4 = gorsel_sec("katman", "Tomato___Late_blight")
    katmanlar = {"Başta (stem)": "stem_activation", "Ortada (blok 3)": "block3b_activation",
                 "Derinde (blok 6)": "block6d_activation"}

    @st.cache_resource
    def ara_model():
        import keras
        g = uretim_modeli().get_layer("efficientnetb0")
        return keras.Model(g.input, [g.get_layer(n).output for n in katmanlar.values()])

    x = np.expand_dims(np.asarray(img4.resize((224, 224)), dtype=np.float32), 0)
    ciktilar = ara_model().predict(x, verbose=0)
    st.image(img4, width=180)
    for (ad, katman), cikti in zip(katmanlar.items(), ciktilar):
        cikti = cikti[0]
        kanallar = np.argsort(cikti.mean(axis=(0, 1)))[::-1][:8]
        st.markdown(f"**{ad}** · `{katman}` · çıktı boyutu {cikti.shape[0]}×{cikti.shape[1]}, {cikti.shape[2]} filtre")
        kolonlar = st.columns(8)
        for kolon, kanal in zip(kolonlar, kanallar):
            h = cikti[..., kanal]
            h = (h - h.min()) / (np.ptp(h) + 1e-6)
            kolon.image(Image.fromarray((h * 255).astype(np.uint8)).resize((112, 112), Image.NEAREST),
                        use_container_width=True)
    st.caption("Başta çıktı büyük (112×112) ve yaprağın şekli seçiliyor. Derine indikçe çıktı küçülüyor "
               "(14×14), filtre sayısı artıyor ve neyi gördüğü insan gözüyle zor anlaşılıyor: model "
               "artık şekil yerine leke, doku gibi soyut özelliklere bakıyor.")
    nasil_okunur(
        "Aynı yaprağın modelin başındaki, ortasındaki ve derinindeki katmanlarda nasıl göründüğü. Her kare bir filtrenin çıktısıdır.",
        "Parlak yerler filtrenin bir şey bulduğu yerlerdir. Aşağı inildikçe kareler bulanıklaşır, çünkü çözünürlük düşer (112 → 28 → 14).",
        "Model ilk katmanlarda yaprağın kenarını ve şeklini, derin katmanlarda leke ve doku gibi hastalığa özgü desenleri yakalamaktadır. Karar en derindeki bu desenlere göre verilir.")

gezinme(__file__)
