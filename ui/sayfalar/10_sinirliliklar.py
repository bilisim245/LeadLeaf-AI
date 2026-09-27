import streamlit as st

from ortak import gezinme

st.title("Sınırlılıklar ve Sonraki Adım")

st.subheader("Gerçek bir test: şeftali yaprak kıvırcıklığı")
c1, c2, c3 = st.columns(3)
c1.metric("Modelin cevabı", "Domates geç yanıklığı", "%89,1 güven", delta_color="inverse")
c2.metric("Şeftali sınıflarına verdiği olasılık", "%0,1")
c3.metric("Bitki filtresiyle", "Tanımlı olmayan belirti", "uzmana yönlendir", delta_color="off")
st.write("Bu hastalık veri setinde bulunmamaktadır. Model \"bilmiyorum\" diyemediği için görüntü, bilinen "
         "sınıflardan en çok benzeyene yüksek güvenle yakıştırılmıştır. Çözüm olarak bitki bilgisi "
         "eklenmiştir: bitki biliniyorsa tahmin yalnızca o bitkinin sınıfları arasından yapılır; hiçbiri "
         "uymuyorsa teşhis konulmaz ve uzmana yönlendirilir.")

st.subheader("Bilinen sınırlılıklar")
st.dataframe({
    "Konu": ["Laboratuvar verisi", "Eksik hastalıklar", "Bitki bilgisi", "Rapor değerlendirmesi", "Altyapı",
             "Tekrar eden görseller", "İlaç dozu (bilinçli karar)"],
    "Durum": [
        "Görseller sade arka planda çekilmiştir. Grad-CAM'de bazı örneklerde arka plana odaklanıldığı görülmüştür. "
        "Tarladan çekilen fotoğraflarda başarı ayrıca ölçülmeli.",
        "Üzüm mildiyösü, şeftali yaprak kıvırcıklığı gibi yaygın hastalıklar ve zeytin, fındık gibi bitkiler veri setinde bulunmamaktadır.",
        "Bitki filtresi şu an kullanıcının bitki adını belirtmesine bağlıdır.",
        "Claude'un raporları bir ziraat mühendisi tarafından sistematik olarak incelenmemiştir.",
        "Sistem tek bir bilgisayarda çalışmaktadır; bir sunucuya taşınmalıdır.",
        "3 test görselinin birebir aynısı eğitim kümesinde bulunmaktadır (etkisi %0,04). Tekrarlar bölmeden önce temizlenmelidir.",
        "Görevde istenen doz ve bekleme süresi verilmemektedir; karar ruhsatlı ziraat mühendisine bırakılmıştır.",
    ],
}, hide_index=True, use_container_width=True)

st.subheader("Sonraki adım")
k1, k2, k3 = st.columns(3)
k1.markdown("**Kısa vade**\n- Bitkiyi yapraktan otomatik tanıma\n- Tarladan etiketli test seti\n- Uzmanla rapor değerlendirmesi")
k2.markdown("**Orta vade**\n- Türkiye'de yaygın hastalıkları eklemek\n- \"Bilmiyorum\" diyebilen bir katman\n- Kullanıcı geri bildirimiyle yeniden eğitim")
k3.markdown("**Uzun vade**\n- Konum ve mevsim bilgisi\n- Sunucuya taşıma\n- Mobil kullanım")

gezinme(__file__)
