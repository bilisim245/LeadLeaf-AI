"""
Streamlit panosuyla sunum senaryosu: hangi sırayla, nereye tıklanacak, ne gösterilecek, ne söylenecek.

Çalıştırma:  .venv\\Scripts\\python sunum\\streamlit_senaryosu_pdf.py
Çıktı:       sunum/streamlit_senaryosu.pdf
"""
import os

from fpdf import FPDF

CIKTI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_senaryosu.pdf")
LACIVERT, GRI, METIN, YESIL = (15, 35, 71), (90, 100, 120), (30, 30, 30), (47, 125, 74)

HAZIRLIK = [
    "Telefon internetine bağlanılır (okul ağında bot çalışmaz).",
    "FastAPI, ngrok, n8n ve Streamlit açılır (başlatma komutları: konusma_notlari.md).",
    "Tarayıcıda http://localhost:8501 açılır; 'Model yükleniyor' yazısı kaybolana kadar beklenir.",
    "Bütün sayfalara bir kez tıklanır (ilk açılış yavaştır, sonra hızlanır).",
    "Canlı Demo'da bir fotoğrafla bir kez deneme yapılır. Telefonda Telegram açık tutulur.",
    "Tarayıcı tam ekran yapılır (F11) ve yakınlaştırma %100'e alınır (Ctrl+0).",
    "Demo için 2 fotoğraf hazır tutulur: bir hastalıklı domates yaprağı ve şeftali yaprak kıvırcıklığı fotoğrafı.",
]

# (süre, sayfa, tıkla, göster, söyle)
ADIMLAR = [
    ("0:00", "Proje Özeti", "Sayfa açık gelir.",
     "Dört sayıyı ve görev şablonu şemasını (DL → LLM-Agent → n8n).",
     "LeadLeaf AI, yaprak fotoğrafından bitki hastalığı ön değerlendirmesi yapan bir sistemdir. Görüntü modeli "
     "hastalığı bulur, Claude bilgi tabanına dayanarak rapor yazar, n8n bunu Telegram'dan çiftçiye iletir."),
    ("1:00", "Proje Özeti", "Aşağı kaydırılır.",
     "Görevin karşılanma tablosunu.",
     "Bootcamp görevinin Temel ve Orta seviyesi tamamen, İleri seviyesi kısmen karşılanmıştır. İlaç dozu bilinçli "
     "olarak verilmemektedir; bu karar ruhsatlı ziraat mühendisine bırakılmıştır."),
    ("1:40", "Veri Seti", "Bitki: Domates → sınıf: Geç Yanıklık; 'Başka örnekler göster'e bir kez basılır.",
     "Gerçek yaprak görsellerini.",
     "Veri seti PlantVillage'dır: 14 bitki, 38 sınıf, 54.305 görsel. Görseller laboratuvarda, düz zemin önünde çekilmiştir."),
    ("2:20", "Veri Seti", "Aşağıdaki 'İki sınıfın karşılaştırılması' gösterilir.",
     "Erken yanıklık ve hedef lekeyi yan yana.",
     "Bazı hastalıklar göz için bile çok benzerdir; modelin en çok zorlandığı yer burasıdır."),
    ("2:50", "Keşifsel Veri Analizi", "'Sınıf dağılımı' sekmesi; sağdaki grafikte bir bitkiye tıklanır.",
     "Dengesizliği (36 kat).",
     "Veri dengesizdir: en büyük sınıf 5.507, en küçük 152 görseldir. Bu yüzden her sınıfı eşit sayan macro F1 de raporlanmıştır."),
    ("3:30", "Keşifsel Veri Analizi", "'Veri kalitesi' sekmesi.",
     "Tekrar eden görselleri ve ekran görüntüsü bulgusunu.",
     "54.305 görselin hepsi kontrol edilmiştir: 3 test görselinin aynısı eğitimde bulunmuş, bir klasörde de ekran "
     "görüntüsü çıkmıştır. Etkileri çok küçüktür ama açıkça belirtilmektedir."),
    ("4:10", "CNN ve Transfer Learning", "'Evrişim' sekmesinde 'Dikey kenar' seçilir.",
     "Filtrenin yapraktaki kenarları bulmasını.",
     "CNN'in temel işlemi budur: küçük filtreler görselde gezip desenleri bulur. Modelde bu filtreler eğitimle öğrenilir."),
    ("4:50", "CNN ve Transfer Learning", "'Neden transfer learning?' sonra 'Katmanlar ne görüyor?' sekmesi.",
     "Hazır modelin avantajlarını ve katman çıktılarını.",
     "ImageNet'te eğitilmiş EfficientNetB0 alınmış, yalnızca son kısmı 38 sınıfa göre eğitilmiştir. Başta yaprağın "
     "şekli, derinde leke ve doku desenleri görülür."),
    ("5:40", "Model Karşılaştırma", "Metrik 'Doğruluk' açık; aşağıda balon grafik.",
     "Üç modelin karşılaştırmasını.",
     "Model seçimi tahmine göre yapılmamıştır: üç model aynı koşullarda karşılaştırılmış, en iyisi EfficientNetB0 olmuştur."),
    ("6:20", "Fine-Tuning", "Parametre grafiği ve önce/sonra grafiği gösterilir.",
     "%94,7'den %97,6'ya çıkışı.",
     "Model aynı kalmış, yalnızca eğitim şekli değişmiştir: gövde kademeli açılmış ve öğrenme oranı düşürülmüştür."),
    ("7:00", "Test Sonuçları", "Üstteki sayılar; 'Karışıklık matrisi' sekmesi ('Sadece hataları göster' açık).",
     "8.146 test görselini ve karışan sınıfları.",
     "Test görselleri modelin hiç görmediği görsellerdir ve yerel olarak yeniden çalıştırılmıştır: doğruluk %98,98. "
     "83 hatanın 70'i aynı bitkinin hastalıkları arasındadır."),
    ("7:50", "Test Sonuçları", "'Güven ve eşik' sekmesi; kaydırıcı 70'ten 90'a götürülür.",
     "Eşiğin etkisini.",
     "Model emin değilse sistem uzmana yönlendirir. %70 eşiğinde 83 hatanın 42'si bu şekilde yakalanmaktadır."),
    ("8:30", "Model Nereye Bakıyor?", "Doğru bilinenler → Domates — Geç Yanıklık; sonra Mısır — Yaygın Pas.",
     "Grad-CAM ısı haritasını.",
     "Geç yanıklıkta model lekeye bakmaktadır. Mısır pasında ise doğru bildiği hâlde arka plana bakmaktadır; "
     "bu, tarla fotoğraflarında başarının düşebileceğinin işaretidir."),
    ("9:20", "RAG ve Rapor", "'Canlı arama' sekmesinde 'Bilgi tabanında ara'ya basılır.",
     "Claude'a giden kaynak metni.",
     "Rapor ezberden değil, kaynaktan yazılır. İlaç markası ve doz verilmez; güven düşükse uzmana yönlendirilir."),
    ("10:00", "Canlı Demo", "Domates fotoğrafı yüklenir, bitki 'Domates', konum 'Antalya'; 'Analiz Et'.",
     "Tespit, rapor, en yakın 3 olasılık, 'bu sonuç nasıl oluştu', hava durumu, PDF indirme.",
     "Telegram'daki akışın aynısı burada adım adım görülmektedir. Rapor PDF olarak da indirilebilir."),
    ("10:50", "Telegram (telefon)", "Telefondan bota aynı fotoğraf gönderilir; PDF sorusunda 'Evet'.",
     "Adım adım durum mesajını, raporu ve gelen PDF'i.",
     "Çiftçinin kullandığı arayüz Telegram'dır. Fotoğraf alındığı andan itibaren her adım mesajda güncellenir."),
    ("11:40", "Sınırlılıklar ve Sonraki Adım", "Sayfa açılır.",
     "Şeftali örneğini ve sınırlılık tablosunu.",
     "Veri setinde olmayan bir hastalık yanlış ama emin şekilde tahmin edilmiştir; bitki filtresiyle bu risk "
     "azaltılmıştır. Öncelik tarladan fotoğrafla test ve bitkinin otomatik tanınmasıdır. Teşekkürler."),
]

IPUCLARI = [
    "Her sayfada önce ekrana bakılır, sonra konuşulur; jüri ne gösterildiğini görmeden anlatılmaz.",
    "Fare, anlatılan grafiğin üzerinde tutulur; jüri nereye bakacağını bilir.",
    "Bir grafik anlatılırken takılınırsa altındaki 'Bu grafik nasıl okunur?' kutusu açılıp 'Söylenecek' okunabilir.",
    "Sayılar ezberlenmez: ekrandaki kutularda yazar, oradan okunur.",
    "Canlı demo aksarsa (ağ, bot) yedek video gösterilir ve Streamlit'teki Canlı Demo ile devam edilir.",
    "Bilinmeyen bir soruda: 'Bu konu denenmemiştir, sonraki aşamada incelenebilir.'",
]


class Belge(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Arial", "", 8)
        self.set_text_color(*GRI)
        self.cell(0, 6, f"LeadLeaf AI · Streamlit sunum senaryosu · {self.page_no()}", align="C")


pdf = Belge()
pdf.add_font("Arial", "", r"C:\Windows\Fonts\arial.ttf")
pdf.add_font("Arial", "B", r"C:\Windows\Fonts\arialbd.ttf")
pdf.set_margins(16, 16, 16)
pdf.set_auto_page_break(True, margin=16)
pdf.add_page()


def yaz(metin, boyut=10, kalin=False, renk=METIN, ara=5.2):
    pdf.set_font("Arial", "B" if kalin else "", boyut)
    pdf.set_text_color(*renk)
    pdf.multi_cell(0, ara, metin, new_x="LMARGIN", new_y="NEXT")


yaz("LeadLeaf AI — Streamlit Sunum Senaryosu", 20, True, LACIVERT, 10)
yaz("Yaklaşık 12 dakika. Her adımda: süre, sayfa, tıklanacak yer, gösterilecek şey ve söylenecek cümle.", 10, renk=GRI)
pdf.ln(3)
yaz("Sunumdan 10 dakika önce", 13, True, LACIVERT, 8)
for i, h in enumerate(HAZIRLIK, 1):
    yaz(f"{i}. {h}")
pdf.ln(3)
yaz("Adım adım akış", 13, True, LACIVERT, 8)
pdf.set_draw_color(*LACIVERT)
for sure, sayfa, tikla, goster, soyle in ADIMLAR:
    if pdf.get_y() > 245:
        pdf.add_page()
    pdf.line(16, pdf.get_y(), 194, pdf.get_y())
    pdf.ln(1.5)
    yaz(f"{sure}  ·  {sayfa}", 11, True, LACIVERT, 6)
    yaz("Tıkla: " + tikla)
    yaz("Göster: " + goster)
    yaz("Söyle: " + soyle, 10, False, YESIL)
    pdf.ln(2)
if pdf.get_y() > 220:
    pdf.add_page()
pdf.ln(2)
yaz("İpuçları", 13, True, LACIVERT, 8)
for ip in IPUCLARI:
    yaz("• " + ip)
pdf.output(CIKTI)
print("Kaydedildi:", CIKTI, "·", pdf.page_no(), "sayfa")
