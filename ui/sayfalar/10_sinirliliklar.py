import streamlit as st

from ortak import gezinme

st.title("Sınırlılıklar ve Sonraki Adım")

st.subheader("Gerçek bir test: şeftali yaprak kıvırcıklığı")
c1, c2, c3 = st.columns(3)
c1.metric("Modelin cevabı", "Domates geç yanıklığı", "%89,1 güven", delta_color="inverse")
c2.metric("Şeftali sınıflarına verdiği olasılık", "%0,1")
c3.metric("Bitki filtresiyle", "Tanımlı olmayan belirti", "uzmana yönlendir", delta_color="off")
st.write("Bu hastalık veri setinde yok. Model \"bilmiyorum\" diyemediği için bildiği sınıflardan en "
         "çok benzeyene yakıştırdı ve üstelik emindi. Çözüm olarak bitki bilgisi eklendi: bitki "
         "biliniyorsa tahmin sadece o bitkinin sınıfları arasından yapılıyor; hiçbiri uymuyorsa sistem "
         "teşhis uydurmuyor.")

st.subheader("Bilinen sınırlılıklar")
st.dataframe({
    "Konu": ["Laboratuvar verisi", "Eksik hastalıklar", "Bitki bilgisi", "Rapor değerlendirmesi", "Altyapı",
             "Tekrar eden görseller"],
    "Durum": [
        "Görseller sade arka planda çekilmiş. Grad-CAM'de model bazen arka plana bakıyor. "
        "Tarladan çekilen fotoğraflarda başarı ayrıca ölçülmeli.",
        "Üzüm mildiyösü, şeftali yaprak kıvırcıklığı gibi yaygın hastalıklar ve zeytin, fındık gibi bitkiler yok.",
        "Bitki filtresi şu an kullanıcının bitki adını yazmasına bağlı.",
        "Claude'un raporlarını bir ziraat mühendisi sistematik olarak incelemedi.",
        "Sistem tek bilgisayarda çalışıyor; sunucuya taşınmalı.",
        "3 test görselinin birebir aynısı eğitimde var (etkisi %0,04). Bölmeden önce tekrarlar temizlenmeli.",
    ],
}, hide_index=True, use_container_width=True)

st.subheader("Sonraki adım")
k1, k2, k3 = st.columns(3)
k1.markdown("**Kısa vade**\n- Bitkiyi yapraktan otomatik tanıma\n- Tarladan etiketli test seti\n- Uzmanla rapor değerlendirmesi")
k2.markdown("**Orta vade**\n- Türkiye'de yaygın hastalıkları eklemek\n- \"Bilmiyorum\" diyebilen bir katman\n- Kullanıcı geri bildirimiyle yeniden eğitim")
k3.markdown("**Uzun vade**\n- Konum ve mevsim bilgisi\n- Sunucuya taşıma\n- Mobil kullanım")

gezinme(__file__)
