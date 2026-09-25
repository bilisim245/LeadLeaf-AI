"""
Streamlit panosunun ekran görüntüleriyle grafik rehberi: her ekranın altında ne gösterdiği,
nasıl okunduğu ve ne anlama geldiği. Önce ekran görüntüleri alınmalı (sunum/ekran_goruntusu_al.py).

Çalıştırma:  .venv\\Scripts\\python sunum\\grafik_rehberi_pdf.py
Çıktı:       sunum/grafik_rehberi.pdf
"""
import glob
import os

import numpy as np
from fpdf import FPDF
from PIL import Image

KLASOR = os.path.dirname(os.path.abspath(__file__))
EKRAN = os.path.join(KLASOR, "ekran_goruntuleri")
CIKTI = os.path.join(KLASOR, "grafik_rehberi.pdf")
LACIVERT, GRI, METIN, YESIL = (15, 35, 71), (90, 100, 120), (30, 30, 30), (47, 125, 74)

TEMEL = [
    ("Ağırlık (parametre) nedir?",
     "Bir yapay sinir ağı, içinde binlerce hatta milyonlarca sayı tutan bir hesap makinesidir. Bu sayılara "
     "ağırlık ya da parametre denir. Bir görüntü modele girince, piksellerin değerleri bu ağırlıklarla çarpılıp "
     "toplanır; sonuçta her hastalık için bir puan çıkar. Eğitim, bu sayıların doğru cevabı verecek şekilde "
     "yavaş yavaş ayarlanmasıdır. Yani modelin 'öğrendiği' her şey bu sayıların içindedir."),
    ("Evrişim ekranındaki 3x3 sayı tablosu (kenardaki ağırlıklar)",
     "Evrişim ekranında solda gördüğünüz tablo (-1 0 1 / -2 0 2 / -1 0 1) tek bir filtrenin 9 ağırlığıdır. Bu "
     "filtre görselin üzerinde 3x3 piksellik bir pencere gibi kaydırılır. Her konumda altındaki 9 piksel bu 9 "
     "sayıyla çarpılıp toplanır. Soldaki sütun eksi, sağdaki artı olduğu için: pencerenin solu ile sağı "
     "arasında parlaklık farkı varsa (yani dikey bir kenar varsa) sonuç büyük çıkar ve o nokta parlar; fark "
     "yoksa sonuç sıfıra yakındır ve karanlık kalır. Sağdaki siyah görselde yaprağın kenarlarının ve "
     "damarlarının parlamasının nedeni budur. Tablodaki sayılar değiştirilirse filtrenin aradığı desen de değişir."),
    ("Modelde bu filtreler nereden geliyor?",
     "Ekrandaki filtreyi elle yazdık, ama gerçek modelde kimse filtre yazmaz. EfficientNetB0'ın içinde "
     "binlerce filtre vardır ve hepsinin sayıları eğitim sırasında kendiliğinden bulunmuştur. Bazıları kenar, "
     "bazıları renk, derindekiler leke ve doku gibi karmaşık desenler arar."),
    ("4.049.571 ve 48.678 sayıları ne?",
     "'Projedeki model' ekranındaki 4.049.571, EfficientNetB0 gövdesindeki toplam ağırlık sayısıdır. Bunlar "
     "ImageNet adlı 1,28 milyon görsellik veriyle önceden eğitilmiş hâlde hazır gelir. 48.678 ise projede "
     "eklenen son katmandaki ağırlık sayısıdır: gövdenin çıkardığı 1.280 özelliği 38 hastalığa bağlayan "
     "katman (1.280 x 38 + 38 = 48.678). Önce yalnızca bu 48.678 sayı eğitilmiş, sonra fine-tuning ile gövdenin "
     "son kısmındaki ağırlıklar da azar azar ayarlanmıştır."),
    ("Transfer learning bir benzetmeyle",
     "Sıfırdan model eğitmek, hiç okuma bilmeyen birine hem okumayı hem tıbbı öğretmek gibidir. Transfer "
     "learning ise okumayı zaten bilen birine yalnızca tıbbı öğretmektir. ImageNet'te eğitilmiş model 'görmeyi' "
     "(kenar, renk, doku, şekil) biliyor; biz ona yalnızca bu desenleri yaprak hastalıklarına bağlamayı öğrettik. "
     "Bu yüzden daha az veri ve daha kısa sürede yüksek başarı elde edildi."),
]

# (dosya öneki, başlık, ekranda ne var, nasıl okunur, ne anlama geliyor)
BOLUMLER = [
    ("01_ozet", "Proje Özeti",
     "Dört metrik kartı (görsel sayısı, bitki/sınıf, test doğruluğu, model adı), bootcamp görev şablonunun "
     "şeması ve kullanılan araçlar tablosu.",
     "Şema yukarıdan aşağı okunur: fotoğraf → görüntü modeli (tahmin) → LLM-Agent (yorum ve rapor) → n8n "
     "(Telegram, PDF, Sheets gibi eylemler).",
     "Proje, bootcamp'in 'DL → LLM-Agent → n8n' şablonuna birebir uyar. %98,98 doğruluk, modelin hiç görmediği "
     "8.146 test görselinde ölçülmüştür."),
    ("02_veri_seti", "Veri Seti",
     "Seçilen bitki ve hastalığa ait 12 gerçek yaprak fotoğrafı; altta iki hastalığın yan yana karşılaştırması.",
     "Soldan bitki ve sınıf seçilir; sağda o sınıfın örnekleri görünür.",
     "Fotoğrafların hepsi laboratuvarda, gri zemin önünde, tek yaprak olarak çekilmiştir. Karşılaştırmada "
     "erken yanıklık ile hedef lekenin ne kadar benzediği görülür; model en çok bu tür benzer hastalıklarda zorlanır."),
    ("03a_dagilim", "Keşifsel Veri Analizi — Sınıf dağılımı",
     "Üstte en büyük ve en küçük sınıf; solda bitkilere göre görsel sayısı (koyu: hastalıklı, açık: sağlıklı) ve "
     "halka grafik; sağda 38 sınıfın görsel sayısı.",
     "Çubuk ne kadar uzunsa o bitkiden ya da sınıftan o kadar çok görsel vardır. Halka grafikte koyu kısım "
     "hastalıklı, açık kısım sağlıklı yaprakların payıdır.",
     "Veri dengesizdir: domates 18.160 görselle en büyük, ahududu 371 görselle en küçük bitkidir; en büyük sınıf "
     "en küçükten 36 kat büyüktür. Bu yüzden yalnızca doğruluğa bakmak yetmez; her sınıfı eşit sayan macro F1 "
     "de raporlanmıştır. Görsellerin %72'si hastalıklıdır."),
    ("03b_bolme", "Keşifsel Veri Analizi — Eğitim / Doğrulama / Test",
     "Üç kümenin görsel sayısı ve her sınıfın bu üç kümeye nasıl paylaştırıldığını gösteren yatay çubuklar.",
     "Her çubuk bir sınıftır ve %100'e tamamlanır. Koyu lacivert eğitim, orta mavi doğrulama, açık mavi test payıdır.",
     "Eğitim kümesi (%70) modelin öğrendiği görsellerdir. Doğrulama (%15) eğitim sırasında 'model iyi gidiyor "
     "mu?' diye bakılan, test (%15) ise en sonda bir kez kullanılan ve modelin hiç görmediği görsellerdir. "
     "Bütün çubuklarda renk sınırları aynı hizada olduğu için bölme her sınıfta aynı oranda yapılmıştır."),
    ("03c_ozellik", "Keşifsel Veri Analizi — Görsel özellikleri",
     "Görsel boyutları tablosu, her sınıfın parlaklık kutu grafiği ve kırmızı-yeşil renk dağılımı.",
     "Kutu grafikte kutu görsellerin ortadaki yarısını, kutunun içindeki çizgi ortanca değeri gösterir. Renk "
     "grafiğinde her nokta bir görseldir: sağa gittikçe kırmızı, yukarı gittikçe yeşil artar.",
     "Bütün görseller 256x256 pikseldir. Sınıflar arasında parlaklık farkı vardır (en koyu mısır pası, en "
     "parlak sağlıklı mısır). Hastalıklı yapraklar ortalamada daha koyu ve daha az yeşildir; ama renk tek başına "
     "hastalığı ayırmaya yetmez, iki grup iç içedir."),
    ("03d_kalite", "Keşifsel Veri Analizi — Veri kalitesi",
     "Birebir aynı görsel sayıları, farklı kümelere düşen kopyalar ve veri setinde bulunan ekran görüntüsü.",
     "Her dosyanın içeriğinden bir özet çıkarılıp karşılaştırılmıştır; özeti aynı olan iki dosya aynı fotoğraftır.",
     "21 fotoğraf veri setinde iki kez vardır; bunlardan 3 test görselinin aynısı eğitimde bulunur. Etkisi en fazla "
     "%0,04'tür, sonucu değiştirmez. Biber klasöründe bir ekran görüntüsü bulunmuştur; model onu %69 güvenle "
     "yanlış tahmin etmiş, güven %70'in altında olduğu için sistem uzmana yönlendirirdi."),
    ("04a_evrisim", "CNN — Evrişim (filtre) denemesi",
     "Solda yaprak seçimi, filtre seçimi ve filtrenin 3x3 ağırlık tablosu; ortada orijinal yaprak, sağda "
     "filtre uygulanmış hâli.",
     "Sağdaki görselde parlak yerler filtrenin aradığı desenin bulunduğu yerlerdir. Filtre değiştirildikçe ya "
     "da tablodaki sayılar düzenlendikçe sağdaki görsel değişir (ağırlıkların anlamı için rehberin başına bakınız).",
     "CNN'in yaptığı en temel iş budur: küçük filtreler görselin üzerinde gezer ve kenar, çizgi, leke gibi "
     "desenleri öne çıkarır. Gerçek modelde binlerce filtre vardır ve sayıları eğitimle bulunur."),
    ("04b_transfer", "CNN — Neden transfer learning?",
     "Sıfırdan CNN eğitimi ile transfer learning'in karşılaştırma tablosu.",
     "Tablo satır satır okunur: başlangıç noktası, gereken veri, süre, ezberleme riski ve projenin koşulları.",
     "Sıfırdan eğitim çok veri ve uzun GPU süresi ister; proje Colab'ın ücretsiz GPU'suyla sınırlıydı. Hazır "
     "modelin 'görme' bilgisi kullanıldığı için daha az veriyle yüksek başarıya ulaşılmıştır."),
    ("04c_model", "CNN — Projedeki model",
     "Üç metrik (gövde katman sayısı, gövde parametresi, eklenen katman parametresi) ve modelin katman şeması.",
     "Şema soldan sağa okunur: 224x224 görsel girer → eğitimde veri artırma uygulanır → EfficientNetB0 gövdesi "
     "özellik çıkarır → 1.280 sayıya indirilir (Global Average Pooling) → Dropout → 38 sınıf için olasılık (softmax).",
     "Modelin büyük kısmı (4 milyon ağırlık) hazır gelmiştir; projede eklenen kısım 48.678 ağırlıktır. Softmax "
     "her hastalığa bir olasılık verir, en yüksek olan tahmin ve o olasılık da 'güven' değeridir."),
    ("04d_katman", "CNN — Katmanlar ne görüyor?",
     "Seçilen yaprak ve modelin başındaki, ortasındaki ve derinindeki katmanlardan 8'er filtre çıktısı.",
     "Her küçük kare bir filtrenin yaprağa verdiği tepkidir; parlak yerler filtrenin bir şey bulduğu yerlerdir. "
     "Aşağı indikçe kareler bulanıklaşır, çünkü çözünürlük düşer (112 → 28 → 14 piksel).",
     "Model başta yaprağın şeklini ve kenarlarını görür; derine indikçe leke ve doku gibi hastalığa özgü "
     "desenlere odaklanır. Son karar en derindeki bu desenlere göre verilir."),
    ("05a_secim", "Model Karşılaştırma",
     "Üç modelin test başarısı (çubuk grafik), boyut-doğruluk-hız karşılaştırması (balon grafik), sonuç tablosu ve "
     "öğrenme eğrisi.",
     "Çubukta uzun olan daha başarılıdır (eksen %80'den başlar). Balon grafikte sol üst hem küçük hem doğru "
     "demektir; balonun büyüklüğü hızı gösterir. Öğrenme eğrisinde mavi eğitim, turuncu doğrulama doğruluğudur; "
     "kesikli çizgi fine-tuning'in başladığı yerdir.",
     "Aynı veri ve ayarlarla en iyi sonucu EfficientNetB0 vermiştir (%94,7); eğitim tarifi geliştirilince "
     "%97,6'ya çıkmıştır. Eğitim ve doğrulama çizgilerinin birbirine yakın gitmesi modelin ezberlemediğini gösterir."),
    ("05b_cm", "Model Karşılaştırma — Karışıklık matrisi (5 sınıf)",
     "Domatesin 5 sınıfındaki test sonuçlarının tablosu.",
     "Satırlar gerçek sınıf, sütunlar modelin tahminidir. Köşegendeki koyu kareler doğru tahminler, köşegen "
     "dışındaki renkli kareler karışmalardır.",
     "Köşegen koyu, dışı açık olduğu için model çoğunlukla doğru tahmin etmektedir; en çok karışan sınıf erken yanıklıktır."),
    ("06_ft", "Fine-Tuning",
     "Her aşamada eğitilen parametre sayısı, aynı modelin iki farklı eğitim tarifiyle sonucu ve öğrenme eğrileri.",
     "Parametre grafiğinde aşama ilerledikçe çubuk uzar, çünkü gövdenin daha fazlası eğitime açılır. Önce/sonra "
     "grafiğinde açık mavi ilk tarif, yeşil kademeli fine-tuning'dir.",
     "Fine-tuning, hazır modelin son katmanlarını açıp çok küçük adımlarla yeniden eğitmektir. Kademeli yapıldığı "
     "için modelin hazır bilgisi bozulmamış; aynı model, yalnızca eğitim şekli değiştirilerek %94,7'den %97,6'ya çıkmıştır."),
    ("07a_cm", "Test Sonuçları — Karışıklık matrisi (38 sınıf)",
     "Üstte test sonuçlarının özeti; ortada 38x38'lik karışıklık matrisi (yalnızca hatalar); altta en sık karışan sınıflar.",
     "Satır gerçek sınıf, sütun tahmindir. Her renkli kare 'bu hastalık şu hastalıkla karıştırıldı' demektir; "
     "renk koyulaştıkça karışma artar. Örnek: gerçekte gri yaprak lekesi olan 9 yaprağa kuzey yaprak yanıklığı denmiştir.",
     "8.146 test görselinde 83 hata vardır, 70'i aynı bitkinin kendi hastalıkları arasındadır. Model bitkiyi "
     "neredeyse hiç karıştırmaz; zorlandığı yer benzer lekeli hastalıklardır."),
    ("07b_sinif", "Test Sonuçları — Sınıf bazında başarı",
     "38 sınıfın her birinin F1 (veya seçilen ölçü) değeri ve tablo.",
     "Çubuk ne kadar kısaysa model o sınıfta o kadar zorlanıyor demektir; turuncu çubuklar %97'nin altındadır. "
     "Precision: model 'bu hastalık' dediğinde ne kadar haklı. Recall: gerçekten o hastalıkta olanların ne kadarını buldu.",
     "Beş sınıf %97'nin altında kalmıştır; en zayıfları mısır gri yaprak lekesi (0,889) ve domates erken yanıklıktır "
     "(0,899). Diğer sınıfların çoğu %99'un üzerindedir."),
    ("07c_guven", "Test Sonuçları — Güven ve eşik",
     "Doğru ve yanlış tahminlerde modelin güven dağılımı ve %70 eşiğinin etkisini gösteren eğri.",
     "Soldaki grafikte lacivert doğru, turuncu yanlış tahminlerdir. Sağdaki kaydırıcı eşiği değiştirir; eğride "
     "lacivert çizgi cevap verilen tahminlerin doğruluğu, yeşil çizgi uzmana gönderilen oranıdır.",
     "Model yanıldığında genellikle daha az emindir (ortalama %71, doğrularda %99). %70 eşiğinde tahminlerin "
     "%98,8'ine cevap verilir, bunların doğruluğu %99,5'tir; 83 hatanın 42'si uzmana gönderilerek yakalanır."),
    ("07d_yanlis", "Test Sonuçları — Yanlış bilinenler",
     "Modelin yanıldığı gerçek test görselleri; her birinin altında gerçek sınıf, tahmin ve güven.",
     "Varsayılan sıralamada en üstte güveni en yüksek hatalar vardır.",
     "Güveni yüksek hatalar en tehlikelileridir: model yanlış olduğu hâlde emindir. Bu yüzden güven eşiği tek "
     "başına yeterli değildir; uzman kontrolü önemlidir."),
    ("08_gradcam", "Model Nereye Bakıyor? (Grad-CAM)",
     "Seçilen yaprak, üzerine modelin kararını en çok etkileyen bölgeleri gösteren ısı haritası ve ilk 3 tahmin.",
     "Kırmızı ve sarı bölgeler kararı en çok etkileyen, mavi bölgeler en az etkileyen yerlerdir.",
     "Kırmızı bölge lekenin üzerindeyse model doğru şeye bakıyordur. Bazı örneklerde (ör. mısır pası) model "
     "doğru bildiği hâlde arka plana bakar; bu, tarla fotoğraflarında başarının düşebileceğinin işaretidir."),
    ("09_rag", "RAG ve Rapor — Bilgi tabanı",
     "Bilgi tabanının özeti, her hastalık için metin parçası sayısı ve açılabilen bir bilgi dosyası.",
     "Her çubuk bir hastalığın bilgi dosyasının kaç parçaya bölündüğünü gösterir.",
     "38 hastalığın hepsi için bilgi dosyası vardır (toplam 197 parça). Model bir hastalık bulduğunda o hastalığın "
     "en ilgili 2 parçası Claude'a verilir ve rapor bu metne dayanarak yazılır; buna RAG denir."),
]


def kirp(yol):
    """Alttaki boş beyaz alanı ve sayfa altındaki gezinme çubuğunu kırpar."""
    im = Image.open(yol).convert("RGB")
    a = np.asarray(im).astype(int)
    dolu = np.where((255 - a).sum(axis=(1, 2)) > a.shape[1] * 6)[0]
    alt = dolu.max() + 20 if len(dolu) else a.shape[0]
    return im.crop((0, 0, im.width, min(alt, im.height)))


class Belge(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Arial", "", 8)
        self.set_text_color(*GRI)
        self.cell(0, 6, f"LeadLeaf AI · Grafik rehberi · {self.page_no()}", align="C")


pdf = Belge()
pdf.add_font("Arial", "", r"C:\Windows\Fonts\arial.ttf")
pdf.add_font("Arial", "B", r"C:\Windows\Fonts\arialbd.ttf")
pdf.set_margins(16, 14, 16)
pdf.set_auto_page_break(True, margin=16)


def yaz(metin, boyut=10, kalin=False, renk=METIN, ara=5.2):
    pdf.set_font("Arial", "B" if kalin else "", boyut)
    pdf.set_text_color(*renk)
    pdf.multi_cell(0, ara, metin, new_x="LMARGIN", new_y="NEXT")


pdf.add_page()
yaz("LeadLeaf AI — Grafik Rehberi", 20, True, LACIVERT, 10)
yaz("Streamlit panosundaki her ekranın görüntüsü ve altında ne gösterdiği, nasıl okunduğu ve ne anlama geldiği.",
    10, renk=GRI)
pdf.ln(4)
yaz("Önce temel kavramlar", 14, True, LACIVERT, 8)
for baslik, metin in TEMEL:
    yaz(baslik, 11, True, LACIVERT, 6)
    yaz(metin)
    pdf.ln(2.5)

gecici = os.path.join(KLASOR, "_gecici")
os.makedirs(gecici, exist_ok=True)
for onek, baslik, ne, nasil, anlam in BOLUMLER:
    dosyalar = sorted(glob.glob(os.path.join(EKRAN, f"{onek}_*.png")))
    if not dosyalar:
        continue
    pdf.add_page()
    yaz(baslik, 15, True, LACIVERT, 8)
    pdf.ln(1)
    for i, yol in enumerate(dosyalar):
        im = kirp(yol)
        gp = os.path.join(gecici, os.path.basename(yol).replace(".png", ".jpg"))
        if im.width > 1400:
            im = im.resize((1400, int(im.height * 1400 / im.width)), Image.LANCZOS)
        im.save(gp, "JPEG", quality=82, optimize=True)
        genislik = 150
        yukseklik = genislik * im.height / im.width
        if pdf.get_y() + yukseklik > 280:
            pdf.add_page()
        pdf.image(gp, x=(210 - genislik) / 2, w=genislik)
        pdf.ln(2)
    if pdf.get_y() > 230:
        pdf.add_page()
    pdf.set_draw_color(*LACIVERT)
    pdf.line(16, pdf.get_y(), 194, pdf.get_y())
    pdf.ln(2)
    yaz("Ekranda ne var?", 11, True, LACIVERT, 6)
    yaz(ne)
    pdf.ln(1.5)
    yaz("Nasıl okunur?", 11, True, LACIVERT, 6)
    yaz(nasil)
    pdf.ln(1.5)
    yaz("Ne anlama geliyor?", 11, True, YESIL, 6)
    yaz(anlam)

pdf.output(CIKTI)
for f in glob.glob(os.path.join(gecici, "*.jpg")):
    os.remove(f)
os.rmdir(gecici)
print("Kaydedildi:", CIKTI, "·", pdf.page_no(), "sayfa")
