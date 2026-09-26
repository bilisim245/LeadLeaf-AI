"""
Streamlit panosundaki her sayfanın ne gösterdiğini sade bir dille anlatan PDF.

Çalıştırma:  .venv\\Scripts\\python sunum\\pano_rehberi_pdf.py
Çıktı:       sunum/pano_rehberi.pdf
"""
import os

from fpdf import FPDF

CIKTI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pano_rehberi.pdf")
LACIVERT, GRI, METIN = (15, 35, 71), (90, 100, 120), (30, 30, 30)

GIRIS = ("Bu rehber, Streamlit panosundaki her sayfanın ne için var olduğunu, ekranda ne göründüğünü ve bunun "
         "ne anlama geldiğini anlatır. Sunumdan önce bir kez baştan sona okunması, sonra panonun yanında açık "
         "tutulması önerilir. Sayılar projenin gerçek sonuçlarından alınmıştır.")

BUYUK_RESIM = [
    "Proje üç parçadan oluşur ve bir fabrikadaki üç istasyon gibi çalışır:",
    "1) Görüntü modeli (CNN): Fotoğrafa bakar, 'bu yaprak domates geç yanıklığı, %97 eminim' der. Yalnızca "
    "sınıf adı ve yüzde üretir, açıklama yapamaz.",
    "2) LLM-Agent (Claude + RAG): Modelin sonucunu alır, bilgi tabanından o hastalıkla ilgili metni bulur ve "
    "çiftçinin anlayacağı bir rapor yazar. Model emin değilse 'uzmana danışın' kararını verir.",
    "3) n8n: Her şeyi birbirine bağlayan otomasyon. Telegram'dan fotoğrafı alır, modele ve Claude'a gönderir, "
    "cevabı çiftçiye iletir, kaydı Google Sheets'e yazar, istenirse PDF gönderir.",
    "Streamlit panosu bu sistemin nasıl kurulduğunu ve ne kadar iyi çalıştığını gösteren vitrindir. Çiftçi "
    "panoyu kullanmaz; çiftçinin kullandığı yer Telegram'dır.",
]

SAYFALAR = [
    ("1. Proje Özeti", "Projenin tek bakışta anlaşılması.", [
        ("Dört kutu (sayılar)", "54.305 görsel, 14 bitki / 38 sınıf, test doğruluğu, kullanılan model.",
         "Veri büyüktür ve model çok başarılıdır. Ama bu başarı laboratuvar fotoğraflarında ölçülmüştür."),
        ("Görev şablonu şeması", "Bootcamp'in istediği üç katman: DL-Model → LLM-Agent → n8n ve projedeki karşılıkları.",
         "Proje, bootcamp görevinin 'Görüntüden Rapor' şablonuna birebir uymaktadır."),
        ("Karşılanma tablosu", "Görevin Temel, Orta ve İleri seviye maddeleri ve projedeki durumu.",
         "Temel ve Orta tamamen, İleri kısmen karşılanmaktadır. İlaç dozunun neden verilmediği altta yazar."),
    ]),
    ("2. Veri Seti", "Modelin neyle eğitildiğinin görülmesi. Jüri veriyi gerçek görsellerle burada görür.", [
        ("Bitki ve sınıf seçimi, 12 görsel", "Seçilen hastalığa ait gerçek fotoğraflar. 'Başka örnekler göster' yeni görseller getirir.",
         "Görseller laboratuvarda, düz zemin önünde, tek yaprak olarak çekilmiştir. Tarladaki fotoğraflar böyle değildir."),
        ("İki sınıfın karşılaştırılması", "İki hastalığın görselleri yan yana (ör. erken yanıklık ve hedef leke).",
         "Bazı hastalıklar göz için bile çok benzer. Modelin en çok zorlandığı yer burasıdır."),
    ]),
    ("3. Keşifsel Veri Analizi", "Veriyi modele vermeden önce sayılarla tanımak (EDA).", [
        ("Bitkilere göre grafik", "Her bitkiden kaç görsel olduğu; koyu hastalıklı, açık sağlıklı.",
         "Veri dengesizdir: domates 18.160, ahududu yalnızca 371 görsel."),
        ("Sınıflara göre grafik", "38 sınıfın görsel sayıları büyükten küçüğe.",
         "En büyük sınıf en küçükten 36 kat büyüktür. Bu yüzden her sınıfı eşit sayan 'macro F1' ölçüsü de kullanılmıştır."),
        ("Eğitim / Doğrulama / Test", "Her sınıfın görsellerinin %70 / %15 / %15 paylaştırılması.",
         "Eğitim: modelin öğrendiği görseller. Doğrulama: eğitim sırasında 'iyi gidiyor mu?' kontrolü. "
         "Test: en sonda bir kez bakılan, modelin hiç görmediği görseller. Başarı yalnızca testte ölçülür."),
        ("Görsel özellikleri", "Görsel boyutu, parlaklık ve renk.",
         "Bütün görseller 256x256. Hastalıklı yapraklar ortalamada daha koyu ve daha az yeşildir."),
        ("Veri kalitesi", "Aynı fotoğrafın iki kez olup olmadığı ve bozuk dosya kontrolü.",
         "21 fotoğraf iki kez var; 3 test görselinin aynısı eğitimde. Biber klasöründe bir ekran görüntüsü "
         "bulundu. Bunlar kontrol edilip açıkça söylenmektedir; sonucu değiştirmez."),
    ]),
    ("4. CNN ve Transfer Learning", "Modelin fotoğrafa nasıl 'baktığının' anlaşılması.", [
        ("Evrişim (filtre) denemesi", "Küçük bir 3x3 filtre yaprağa uygulanır; sağda sonucu görünür.",
         "CNN'in temel işi budur: küçük filtreler görselin üstünde gezip kenar, leke gibi desenleri bulur."),
        ("Neden transfer learning?", "Sıfırdan model eğitmek ile hazır model uyarlamanın karşılaştırması.",
         "ImageNet'te 1,28 milyon görselle eğitilmiş hazır bir model alınmış, yalnızca son kısmı 38 sınıfa göre "
         "eğitilmiştir. Böylece daha az veri ve sürede yüksek başarı elde edilmiştir."),
        ("Projedeki model", "Modelin katmanları: girdi → veri artırma → EfficientNetB0 gövdesi → sınıflandırma.",
         "Gövde 4 milyon parametre ve hazır gelir; eklenen kısım yaklaşık 48 bin parametredir."),
        ("Katmanlar ne görüyor?", "Aynı yaprağın modelin başındaki, ortasındaki ve derinindeki hâli.",
         "Başta yaprağın şekli, derinde leke ve doku desenleri görülür. Karar derindeki desenlere göre verilir."),
    ]),
    ("5. Model Karşılaştırma", "Neden EfficientNetB0 seçildiğinin gösterilmesi.", [
        ("Çubuk grafik", "Üç modelin test başarısı; yeşil üretimde kullanılan.",
         "Aynı veri ve ayarlarla en iyisi EfficientNetB0'dır (%94,7); fine-tuning ile %97,6'ya çıkmıştır."),
        ("Balon grafik", "Model boyutu, doğruluk ve hız birlikte.",
         "En küçük model en az doğru olandır. EfficientNetB0 biraz büyük ama en doğrudur; sunucuda çalıştığı için sorun değildir."),
        ("Eğitim grafikleri", "Öğrenme eğrisi ve karışıklık matrisi (Colab çıktıları).",
         "Eğitim ve doğrulama çizgileri birbirine yakınsa model ezberlememiştir."),
    ]),
    ("6. Fine-Tuning", "Başarıyı %94,7'den %97,6'ya çıkaran eğitim tekniğinin anlatılması.", [
        ("Fine-tuning nedir?", "Hazır modelin son katmanlarını açıp küçük adımlarla yeniden eğitmek.",
         "Önce yalnızca son katman eğitilmiş, sonra gövdenin son %15, %30 ve %40'ı kademeli açılmıştır."),
        ("Parametre grafiği", "Her aşamada eğitilen parametre sayısı.",
         "Kademeli açıldığı için model hazır bilgisini bir anda kaybetmemiştir."),
        ("Önce / sonra", "Aynı modelin iki eğitim şekliyle sonucu.",
         "Model değişmemiş, yalnızca eğitim şekli değişmiştir; başarı yaklaşık 3 puan artmıştır."),
    ]),
    ("7. Test Sonuçları (38 Sınıf)", "Modelin gerçek başarısının ayrıntılı gösterilmesi. En önemli sayfadır.", [
        ("Üstteki sayılar", "8.146 test görseli, doğruluk %98,98, 83 yanlış.",
         "Model test görsellerinin yerel olarak yeniden çalıştırılmasıyla ölçülmüştür. Colab'daki %99,02 ile fark 3 görseldir."),
        ("Karışıklık matrisi", "Satır: gerçek sınıf, sütun: modelin tahmini. Hatalar renkli kareler.",
         "83 hatanın 70'i aynı bitkinin hastalıkları arasındadır. En çok mısırda gri leke ile kuzey yanıklığı karışır."),
        ("Sınıf bazında", "Her sınıfın ayrı başarısı; turuncular %97'nin altı.",
         "Beş sınıf zayıftır; hepsi birbirine benzeyen lekeli hastalıklardır."),
        ("Güven ve eşik", "Modelin ne kadar emin olduğu ve %70 eşiğinin etkisi. Kaydırıcıyla değiştirilebilir.",
         "Model yanıldığında genelde daha az emindir. %70'in altı uzmana gider; bu 83 hatanın 42'sini yakalar."),
        ("Yanlış bilinenler", "Modelin yanıldığı gerçek görseller.",
         "Güveni yüksek olan hatalar en tehlikelidir; model yanlış olduğu hâlde emindir."),
    ]),
    ("8. Model Nereye Bakıyor?", "Modelin kararını neye göre verdiğinin gösterilmesi (Grad-CAM).", [
        ("Isı haritası", "Kırmızı yerler kararı en çok etkileyen bölgeler.",
         "Çoğu örnekte model lekeye bakar. Bazılarında (ör. mısır pası) doğru bildiği hâlde arka plana bakar; "
         "bu, tarla fotoğraflarında başarının düşebileceğinin işaretidir."),
    ]),
    ("9. RAG ve Rapor", "Raporun ezberden değil kaynaktan yazıldığının gösterilmesi.", [
        ("Bilgi tabanı", "38 hastalık dosyası, 197 metin parçası; bir dosya açılıp okunabilir.",
         "Her hastalık için etken, belirti, uygun koşullar ve önlemler yazılmıştır."),
        ("Canlı arama", "Bir hastalık seçilip 'ara'ya basılınca Claude'a giden metin görünür.",
         "Claude raporu bu metne dayanarak yazar; buna RAG denir."),
        ("Kurallar ve örnek rapor", "Claude'a verilen güvenlik kuralları ve gerçek bir rapor.",
         "İlaç markası ve doz verilmez; güven düşükse uzmana yönlendirilir; kötü niyetli mesajlar uygulanmaz."),
    ]),
    ("10. n8n Akışı", "Telegram botunun arkasındaki akışın tıklanabilir şeması: her düğüm ne yapıyor.", [
        ("Şema (35 kutu)", "n8n'deki akışın aynısı; renkler düğüm türünü, yeşil/kırmızı oklar IF'in evet/hayır çıkışını gösterir.",
         "Mesajın hangi dala gideceğine LLM değil IF düğümleri karar verir; akış öngörülebilirdir."),
        ("Kutuya tıklama", "Tıklanan düğümün ne yaptığı, nereden gelip nereye gittiği ve n8n'deki ayarları (URL, koşul, prompt, kod).",
         "n8n'in 'kod yazmadan' kurulan bir akış olduğu, ama her kutunun arkasında açık bir ayar/ifade bulunduğu görülür."),
        ("Senaryo + ▶", "Fotoğraf, 'Domates' düzeltmesi, sohbet, 'merhaba', PDF butonu için mesajın izlediği yol adım adım.",
         "Bir fotoğrafın yolculuğu: Telegram → CNN → RAG → Claude → rapor, kayıt, PDF, uzman, takip."),
    ]),
    ("11. Canlı Demo", "Sistemin çalışırken gösterilmesi.", [
        ("Fotoğraf yükle → Analiz Et", "Model tahmini, durum kutusu, Claude raporu, en yakın 3 olasılık, 'bu sonuç nasıl oluştu'; konum yazılırsa hava durumu.",
         "Telegram'daki akışın aynısı ekranda adım adım görülür. Rapor PDF olarak indirilebilir."),
        ("'Bitki doğru mu?' sorusu (sonucun altında)", "Bitki önceden seçtirilmez; model önce tahmin eder, sonra "
         "'Model bu yaprağı Mısır yaprağı olarak değerlendirdi. Doğru mu?' diye sorulur. 'Hayır' denip doğru bitki "
         "seçilirse aynı fotoğraf yalnızca o bitkinin hastalıkları arasından yeniden değerlendirilir.",
         "Sonuç önceden yapılan bir seçimle yönlendirilmez. Tanımsız bir hastalıkta sistem teşhis uydurmaz, uzmana "
         "yönlendirir (şeftali örneği). Telegram'da da aynı mantık var: fotoğraftan sonra bitki adı yazılabilir."),
    ]),
    ("12. Sınırlılıklar ve Sonraki Adım", "Sistemin neyi yapamadığının dürüstçe söylenmesi.", [
        ("Şeftali örneği", "Veri setinde olmayan bir hastalığın yanlış ama emin şekilde tahmin edilmesi.",
         "Model 'bilmiyorum' diyemez. Bitki filtresi bu riski azaltır."),
        ("Sınırlılık tablosu ve yol haritası", "Bilinen eksikler ve sonraki adımlar.",
         "Öncelik: tarladan fotoğrafla test, bitkinin otomatik tanınması, uzmanla değerlendirme."),
    ]),
]


class Belge(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Arial", "", 8)
        self.set_text_color(*GRI)
        self.cell(0, 6, f"LeadLeaf AI · Pano rehberi · {self.page_no()}", align="C")


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


yaz("LeadLeaf AI — Streamlit Pano Rehberi", 20, True, LACIVERT, 10)
yaz(GIRIS, 10, renk=GRI)
pdf.ln(3)
yaz("Büyük resim: sistem ne yapıyor?", 13, True, LACIVERT, 8)
for p in BUYUK_RESIM:
    yaz(p)
    pdf.ln(1)
pdf.ln(3)

for baslik, amac, ogeler in SAYFALAR:
    if pdf.get_y() > 235:
        pdf.add_page()
    yaz(baslik, 13, True, LACIVERT, 8)
    pdf.set_draw_color(*LACIVERT)
    pdf.line(16, pdf.get_y(), 194, pdf.get_y())
    pdf.ln(1.5)
    yaz("Bu sayfa neden var: " + amac, 10, renk=GRI)
    pdf.ln(1.5)
    for ne, ekranda, anlam in ogeler:
        if pdf.get_y() > 255:
            pdf.add_page()
        yaz(ne, 10.5, True, LACIVERT, 5.5)
        yaz("Ekranda: " + ekranda)
        yaz("Ne anlama geliyor: " + anlam)
        pdf.ln(2)
    pdf.ln(2)

pdf.output(CIKTI)
print("Kaydedildi:", CIKTI, "·", pdf.page_no(), "sayfa")
