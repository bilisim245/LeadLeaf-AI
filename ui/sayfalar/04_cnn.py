import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy import ndimage

from ortak import (KOK, etiket, gezinme, kod_parcasi, ornek_dosyalar, sinif_tablosu, uretim_modeli,
                   veri_var)

st.title("CNN ve Transfer Learning")

siniflar = sinif_tablosu().sort_values("Etiket")["Sınıf"].tolist()


def gorsel_sec(anahtar: str, varsayilan: str = "Tomato___Early_blight") -> Image.Image:
    if veri_var():
        sinif = st.selectbox("Yaprak", siniflar, index=siniflar.index(varsayilan), format_func=etiket, key=anahtar)
        return Image.open(ornek_dosyalar(sinif, 1, 3)[0]).convert("RGB")
    return Image.open(os.path.join(KOK, "model", "demo_images", "Tomato___Early_blight.jpg")).convert("RGB")


t1, t2, t3, t4 = st.tabs(["Evrişim (convolution)", "Neden transfer learning?", "Bizim model", "Katmanlar ne görüyor?"])

with t1:
    st.write("CNN'in temel işlemi: küçük bir filtre (3×3 sayı) görselin üzerinde kaydırılır. Her konumda "
             "filtre ile altındaki pikseller çarpılıp toplanır. Filtre neyi arıyorsa o bölgeler parlar. "
             "Aşağıda filtreyi seçip ya da sayıları değiştirip sonucu görebilirsiniz.")
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
    st.caption("Gerçek bir CNN'de bu filtreleri biz yazmıyoruz; model eğitim sırasında kendisi öğreniyor. "
               "İlk katmanlar kenar ve renk geçişi gibi basit şeyler, son katmanlar leke ve doku gibi "
               "daha karmaşık şeyler öğreniyor.")

with t2:
    st.write("İki yol vardı: sıfırdan bir CNN yazıp eğitmek ya da daha önce milyonlarca görselle "
             "eğitilmiş bir modeli alıp kendi verimize uyarlamak (transfer learning). İkinciyi seçtik.")
    st.table(pd.DataFrame({
        "": ["Başlangıç", "Gereken veri", "Eğitim süresi", "Ezberleme riski", "Bizim durumumuz"],
        "Sıfırdan CNN": ["Rastgele ağırlıklar, hiçbir şey bilmiyor", "Çok fazla", "Uzun",
                         "Yüksek", "Colab'ın ücretsiz GPU'su ve kısıtlı süre"],
        "Transfer learning": ["ImageNet'te (1,28 milyon görsel, 1000 sınıf) eğitilmiş",
                              "Daha az yeter", "Kısa", "Daha düşük",
                              "Kenar, doku, şekil bilgisi hazır geliyor"],
    }))
    st.write("Yaprak da sonuçta bir görsel. Kenar, doku, renk geçişi gibi temel özellikleri ImageNet'te "
             "öğrenmiş bir model bunları yaprakta da kullanabiliyor. Biz sadece son kısmı kendi 38 "
             "sınıfımıza göre eğitiyoruz.")
    st.info("Not: Sıfırdan bir CNN'i ayrıca eğitip karşılaştırmadık. Bu bir sonraki adım olarak "
            "yapılabilir; literatürde PlantVillage'da transfer learning'in sıfırdan eğitime göre "
            "daha iyi sonuç verdiği biliniyor.")

with t3:
    model = uretim_modeli()
    govde = model.get_layer("efficientnetb0")
    kafa_param = model.count_params() - govde.count_params()
    c1, c2, c3 = st.columns(3)
    c1.metric("Gövde (EfficientNetB0) katmanı", len(govde.layers))
    c2.metric("Gövde parametresi", f"{govde.count_params():,}".replace(",", "."), "ImageNet'ten hazır", delta_color="off")
    c3.metric("Bizim eklediğimiz kafa", f"{kafa_param:,}".replace(",", "."), "38 sınıf için", delta_color="off")
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
    with st.expander("Kod: modeli kurduğumuz yer (notebooks/03_efficientnetb0_38_sinif.py)"):
        st.code(kod_parcasi("notebooks/03_efficientnetb0_38_sinif.py", "inputs = keras.Input", "def parametre_ozeti"),
                language="python")

with t4:
    st.write("Bir yaprağı modele verip gövdenin farklı derinliklerindeki çıktılara bakıyoruz. "
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

gezinme(__file__)
