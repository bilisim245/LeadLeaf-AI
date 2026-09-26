# Çalışma Rehberi — Kâğıttaki 16 Soru

26 Eylül 2026'da elle yazılan soru listesinin cevapları. Her cevap: **kısa cevap** (jüriye
söylenecek 1-2 cümle) → **açıklama** → **projede nerede**. Sayılar projenin gerçek çıktılarından
alındı (`model/model_38sinif/`, `model/tubitak/`).

**Çalışma sırası önerisi (sunuma 2 gün var):** önce 2 → 3 → 4 → 2b → 8 (modelin kalbi, jüri en çok
buradan sorar), sonra 15 → 6 (sonuçları okumak), sonra 1 → 7 → 12 → 13 (LLM tarafı), en son
9 → 11 → 14 → 10 → 16 (altyapı ve sınırlılıklar).

---

## 1) RAG ve TR_ADLAR — ikisi aynı şey mi?

**Kısa cevap:** Hayır. `TR_ADLAR` sabit bir **sözlük** (İngilizce sınıf adı → Türkçe ad). RAG ise
hastalık hakkında **bilgi metni getiren** bir arama sistemi. İkisi de CNN'in bulduğu sınıf adını
kullanıyor ama işleri farklı.

**Açıklama:**
- CNN'in çıktısı İngilizce bir sınıf adı: `Tomato___Late_blight`.
- **TR_ADLAR** (`inference/app.py`): bu adı çiftçinin anlayacağı Türkçe ada çevirir →
  "Geç Yanıklık (Late Blight)". 38 sınıfın hepsi elle yazıldı ve kontrol edildi.
  **Neden LLM'e çevirtmedik?** LLM bazen olmayan bir halk adı uydurabilir. Bu yüzden Claude'a Türkçe
  ad hazır veriliyor ve promptta "bu adı değiştirme, aynen kullan" kuralı var.
- **RAG** (Retrieval-Augmented Generation = "getirerek zenginleştirilmiş üretim"):
  `agent/knowledge/` klasöründe 38 sınıfın her biri için elle yazılmış bir bilgi dosyası var
  (etken, belirtiler, uygun koşullar, karıştırılabilecek hastalıklar, önlemler). Bu dosyalar
  parçalara (chunk) bölünüp sayılara (vektör/embedding) çevrildi ve Chroma veritabanına kondu.
  CNN "geç yanıklık" deyince, bu hastalığın en alakalı 2 parçası getirilip Claude'un önüne konuyor.
  Claude artık **ezberinden değil, bizim doğruladığımız metinden** konuşuyor.
- Benzetme: TR_ADLAR = **sözlük**, RAG = **kütüphaneden ilgili kitabın ilgili sayfasını getiren kütüphaneci**.

![RAG'in iki aşaması](sunum/sistem_gorselleri/rag_akisi.png)

**Projede:** `inference/app.py` → `TR_ADLAR`; `rag/build_index.py` (indeksi kurar),
`agent/rag.py` → `retrieve_context()` (sorguda getirir), embedding modeli
`paraphrase-multilingual-MiniLM-L12-v2` (Türkçeyi anladığı için seçildi). Panoda: **RAG** sayfası.

---

## 2) %70 / %15 / %15 — eğitim sırasında hangisi nereye gidiyor? Doğrulama nedir? Test nedir?

**Kısa cevap:** 54.305 fotoğrafı üçe böldük: **eğitim** (model bunlardan öğrenir), **doğrulama**
(eğitim sırasında "iyi gidiyor mu?" diye kontrol edilir), **test** (en sonda bir kere, hiç görmediği
fotoğraflarla gerçek not).

| Bölüm | Oran | Fotoğraf | Ne işe yarar | Benzetme |
|---|---|---|---|---|
| Eğitim (train) | %70 | 38.013 | Model ağırlıklarını bunlarla günceller, **öğrenir** | Ders kitabı + çalışma soruları |
| Doğrulama (valid) | %15 | 8.146 | Her epoch sonunda ölçülür; eğitimi **yönlendirmek** için kullanılır | Deneme sınavları |
| Test | %15 | 8.146 | En sonda **bir kere** ölçülür; rapordaki %99,02 buradan | Gerçek sınav |

**Doğrulama seti neden var, eğitim setinde ölçsek olmaz mı?** Model eğitim fotoğraflarını
ezberleyebilir (**aşırı öğrenme / overfitting**): eğitimde %100, yeni fotoğrafta kötü. Doğrulama
seti modelin hiç öğrenmediği fotoğraflar olduğu için ezberi yakalar. Projede doğrulama setiyle
verilen kararlar:
- **EarlyStopping:** doğrulama doğruluğu artık yükselmiyorsa eğitimi durdur, **en iyi** ağırlıklara dön.
- **ReduceLROnPlateau:** doğrulama kaybı düşmüyorsa öğrenme hızını yarıya indir.

**Test neden ayrı, doğrulama yetmez mi?** Doğrulama setine bakarak karar verdiğimiz için model
dolaylı yoldan ona "uyum" sağlıyor (deneme sınavlarına göre çalışan öğrenci gibi). Dürüst not için
hiçbir karara karışmamış üçüncü bir set gerekiyor → test.

**Önemli ayrıntılar:**
- **Tabakalı (stratified) bölme:** her sınıf ayrı ayrı %70/%15/%15 bölündü. Böylece az fotoğraflı
  sınıflar (ör. sağlıklı patates: 106/23/23) da her bölümde temsil ediliyor.
- **SEED = 42:** rastgele bölme sabitlendi; tekrar çalıştırınca aynı bölme çıkıyor.
- Bölme bir kere yapılıp `split_manifest.json`'a kaydedildi (hangi fotoğraf hangi bölümde).
  Bu sayede test evde yeniden üretildi: %98,98 (Colab'da %99,02; fark 3 fotoğraf).

**Projede:** `notebooks/03_efficientnetb0_38_sinif.py` → `ORANLAR`, `_stratified_uc_yonlu_split()`;
`model/model_38sinif/split_manifest.json`.

---

## 2b) Kademeli açma neden? "Kafa" ne demek?

**Kısa cevap:** Hazır eğitilmiş modelin katmanlarını birden değil, **azar azar** (son %15 → %30 →
%40) eğitime açtık. Birden açarsak önceden öğrendiği değerli bilgiyi bozabiliyor.

**Önce iki terim:**
- **Gövde (base / backbone):** EfficientNetB0'ın ImageNet'te eğitilmiş hazır kısmı. Kenar, doku,
  leke gibi genel görsel özellikleri tanıyor.
- **Kafa (head):** Bizim eklediğimiz son katmanlar: `GlobalAveragePooling2D` → `Dropout(0.2)` →
  `Dense(38, softmax)`. "Bu özellikler hangi hastalığa ait?" kararını veren kısım. Başta rastgele.

**Eğitim aşamaları (38 sınıflı model):**

| Aşama | Ne eğitiliyor | Öğrenme hızı | Neden |
|---|---|---|---|
| 0 | Sadece kafa (gövde tamamen donuk) | 1e-3 (0,001) | Kafa rastgele başlıyor; önce o bir şeyler öğrensin |
| 1 | Kafa + gövdenin son %15'i | 1e-5 | Gövdenin en "özel" katmanlarını hastalıklara uyarla |
| 2 | Kafa + son %30 | 1e-5 (gerekirse yarıya iner) | Biraz daha derine |
| 3 | Kafa + son %40 | 1e-5 (gerekirse yarıya iner) | En derine |

**Neden kademeli?**
1. **Unutma sorunu (catastrophic forgetting):** Kafa henüz rastgeleyken gövdeyi açarsak, kafanın
   büyük ve rastgele hataları gövdeye geri yayılır, ImageNet'ten gelen iyi ağırlıkları bozar.
2. **Gövdenin ilk katmanları zaten evrensel:** kenar/renk bulan filtreler yaprak için de geçerli,
   onlara hiç dokunmuyoruz. Son katmanlar daha özel ("köpek kulağı" gibi) → onları uyarlıyoruz.
3. **Küçük öğrenme hızı (1e-5):** açılan katmanları büyük adımlarla değil, **ince ayar** ile değiştirmek.
   "Fine-tuning" (ince ayar) adı buradan geliyor.

**Projede:** `notebooks/03_efficientnetb0_38_sinif.py` → `UNFREEZE_ASAMALARI = [0.15, 0.30, 0.40]`.
Panoda: **Fine-Tuning** sayfası.

---

## 3) CNN ve evrişim (convolution) işlemi

**Kısa cevap:** CNN, görselin üzerinde küçük bir **filtre** (3×3 sayı) gezdirir. Her konumda filtre
ile altındaki pikselleri çarpıp toplar. Sonuç, filtrenin aradığı desen (kenar, leke…) orada varsa büyük çıkar.

**Adım adım örnek (dikey kenar filtresi):**

```
Görselin 3×3'lük parçası        Filtre (dikey kenar)
  10  10  80                      -1   0   1
  10  10  80                      -2   0   2
  10  10  80                      -1   0   1
```
Karşılıklı çarp ve topla: (-10+0+80) + (-20+0+160) + (-10+0+80) = 70 + 140 + 70 = **280** → büyük
sayı: **burada dikey bir kenar var** (soldan sağa koyu → açık geçiş).

Aynı filtre düz bir bölgede (hepsi 10): (-10+10) + (-20+20) + (-10+10) = **0** → kenar yok.

Filtre bütün görselde kaydırılınca yeni bir görsel (**özellik haritası / feature map**) oluşur.
Parlak yerler = desenin bulunduğu yerler.

**CNN'in katmanları:** İlk katmanlar basit şeyler bulur (kenar, renk) → orta katmanlar bunları
birleştirir (doku, leke şekli) → derin katmanlar daha soyut şeyler ("kahverengi halkalı leke").
Her katmanda çözünürlük düşer, filtre sayısı artar (panoda: 112 → 28 → 14 piksel).

**Projede / panoda:** **CNN ve Transfer Learning** sayfası → evrişim oyun alanı (filtre seç,
sayıları değiştir, sonucu gör) + "katman çıktıları" (aynı yaprağın başta, ortada, derinde nasıl göründüğü).

---

## 4) Transfer learning nedir?

**Kısa cevap:** Sıfırdan model eğitmek yerine, başka bir büyük veri setinde (ImageNet: ~1,2 milyon
fotoğraf, 1000 sınıf) **zaten eğitilmiş** bir modeli alıp kendi problemimize uyarlamak.

**Açıklama:** Benzetme: iyi resim yapabilen birine bitki hastalıklarını öğretmek, hiç kalem
tutmamış birine öğretmekten çok daha kolay. Model "görmeyi" (kenar, doku, şekil) ImageNet'te
öğrenmiş; biz sadece "bu gördüklerin hangi hastalık?" kısmını öğretiyoruz.

- `EfficientNetB0(weights="imagenet", include_top=False)` → ImageNet ağırlıklarıyla yükle, ama
  ImageNet'in 1000 sınıflık **kafasını** alma (`include_top=False`), kendi 38 sınıflık kafamızı ekle.
- **Faydası:** 38.000 fotoğrafla %99 doğruluk. Sıfırdan eğitmek çok daha fazla veri ve GPU zamanı isterdi.

**Projede:** `notebooks/03_...py` 219. satır civarı. Panoda: **CNN ve Transfer Learning** sayfası.

---

## 5) CNN nasıl görür? Ağırlıklar ne? [1 | 2 | 1] ne anlama geliyor? Streamlit'te pencere var mı?

**Kısa cevap:** Filtredeki sayıların her biri bir **ağırlıktır**. Modelin "öğrenmesi" = bu
sayıların eğitimle ayarlanması. Panoda bu sayıların elle değiştirilebildiği bir pencere **var**.

**[1, 2, 1] ne?** Yatay kenar filtresinin satırları: `[-1, -2, -1]`, `[0, 0, 0]`, `[1, 2, 1]`.
- Üst satır eksi, alt satır artı → **üst ile alt arasındaki farkı** ölçer (yatay kenar).
- Ortadaki **2**: ortadaki piksele yanlardakinden **iki kat önem** ver demek (merkeze yakın olan daha önemli).
- Bu hazır filtrenin adı **Sobel filtresi**; görüntü işlemede CNN'den önce de kullanılıyordu.

**Gerçek CNN'de fark:** Bu sayıları **biz yazmıyoruz**. Model başta rastgele sayılarla başlar;
eğitimde her yanlış tahminden sonra sayılar hatayı azaltacak yönde biraz değiştirilir (**geri
yayılım / backpropagation**). Binlerce tekrardan sonra filtreler kendiliğinden "leke bulan",
"damar bulan" filtrelere dönüşür. EfficientNetB0'da ~4 milyon ağırlık var.

**Streamlit'te nerede:** `ui/sayfalar/04_cnn.py` → **CNN ve Transfer Learning** sayfası:
"Hazır filtre" seçimi (Dikey kenar, Yatay kenar, Keskinleştirme, Bulanıklaştırma) ve altında
**3×3'lük düzenlenebilir tablo**. Sayıları değiştirince sağdaki görsel anında değişiyor. Jüriye
canlı gösterilebilir: "yatay kenarı seçip sayıları değiştiriyorum, bakın parlayan yerler değişiyor".

---

## 6) Isı haritaları ne anlama geliyor?

Panoda iki farklı ısı haritası var, karıştırılmamalı:

**a) Karışıklık matrisi (Test Sonuçları sayfası):**
- **Satırlar = gerçek sınıf, sütunlar = modelin tahmini.** Her kare: "gerçekte X olan kaç fotoğrafa
  model Y dedi?" Renk koyulaştıkça sayı büyük.
- **Köşegen** (sol üstten sağ alta) = doğru tahminler. Köşegen dışında kalan her şey hata.
- Bizim modelde: 8.146 test fotoğrafında 83 hata, bunların 70'i **aynı bitki içinde** (ör. mısır gri
  yaprak lekesi ↔ mısır kuzey yaprak yanıklığı: 17 hata). Yani model bitkiyi neredeyse hiç
  karıştırmıyor, benzer görünen hastalıkları karıştırıyor.
- Panoda "Sadece hataları göster" seçeneği var → köşegeni gizleyip hataları öne çıkarıyor.

**b) Grad-CAM (Model Nereye Bakıyor? sayfası):**
- Fotoğrafın üzerine boyanan renkli harita: **kırmızı = modelin kararını en çok etkileyen yerler**.
- Amaç: model doğru şeye mi bakıyor? (**açıklanabilirlik**)
- Bulgumuz: geç yanıklıkta ve esca'da lekeye bakıyor (iyi). Mısır pası ve elma kara çürüklüğünde
  doğru bildiği hâlde bazen **arka plana/sapa** bakıyor → PlantVillage'ın düz arka planından
  öğrendiği bir kısayol olabilir. Bu da tarla fotoğraflarında neden zorlandığını açıklıyor (bkz. 16).

---

## 7) RAG ve LLM nasıl bağlandı? HTTP Request ne?

**Kısa cevap:** n8n akışında sırayla: CNN'e fotoğraf gönderilir → RAG'den o hastalığın bilgisi
alınır → ikisi birlikte Claude'a verilir → Claude raporu yazar.

![Bileşenler ve veri depoları](sunum/sistem_gorselleri/mimari.png)

**HTTP Request nedir?** Bir programın başka bir programa internet (ya da aynı bilgisayar) üzerinden
"şunu yap, cevabı bana ver" demesi. Tarayıcıya adres yazmak da bir HTTP isteği. n8n'deki
"HTTP Request" düğümü bunu yapan kutu.

**Akış (n8n):**
```
Telegram Trigger → Fotoğrafı İndir
  → HTTP Request - Predict CNN   (POST http://127.0.0.1:8000/predict, fotoğraf gönderir)
  → HTTP Request - RAG Context   (GET  http://127.0.0.1:8000/rag-context?hastalik=...)
  → Basic LLM Chain + Anthropic Chat Model   (Claude'a: sistem promptu + CNN sonucu + RAG metni)
  → Rapor JSON'unu Ayrıştır → Telegram cevabı, Google Sheets kaydı, PDF
```
- CNN ve RAG **bizim FastAPI servisimizde** (`inference/app.py`), n8n onlara HTTP Request ile ulaşıyor.
- Claude'a ise **HTTP Request ile değil**, n8n'in LangChain düğümleriyle (Basic LLM Chain + Anthropic
  Chat Model) bağlanıyor. 21 Eylül'e kadar HTTP Request ile doğrudan `api.anthropic.com`'a
  bağlıydık; LangChain düğümlerine geçtik çünkü API anahtarı n8n'in şifreli kimlik bilgisinde
  (credential) duruyor, model değiştirmek tek tık ve bootcamp "prompt geliştirme n8n'de" istiyordu.

---

## 8) İki model arasındaki fark ne, neden yapıldı?

İki karşılaştırma var, ikisi de sorulabilir:

**a) Aynı mimari (EfficientNetB0), iki eğitim tarifi (5 sınıf, domates):**

| | Önceki (`01`) | Gelişmiş (`02`) |
|---|---|---|
| Gövdeden açılan kısım | Tek seferde son %25 | Kademeli: %15 → %30 → %40 |
| Öğrenme hızı | Sabit | Takılınca yarıya iner (ReduceLROnPlateau) |
| Erken durdurma | Az sabırlı | Daha sabırlı |
| **Doğruluk** | 0,9468 | **0,9762** |
| **Macro F1** | 0,940 | **0,9716** |

**Neden yapıldı?** Önce 3 mimari yarıştırıldı (MobileNetV2 0,9095, MobileNetV3Small 0,8794,
EfficientNetB0 0,9468). Kazanan EfficientNetB0 oldu. Sonra "kazananı daha iyi eğitebilir miyiz?"
sorusu için **sadece eğitim tarifini** değiştirip yeniden eğittik. Mimari aynı kaldığı için
iyileşmenin eğitim yönteminden geldiği kesin.

**b) 5 sınıftan 38 sınıfa:** Sunum tarihi 28 Eylül olarak netleşince zaman açıldı; en iyi tarifle
(`02`) 14 bitki / 38 sınıf eğitildi (`03`): doğruluk **0,9902**, macro F1 **0,9859**. Üretimde bu model var.

Panoda: **Model Karşılaştırma** ve **Fine-Tuning** sayfaları.

---

## 9) n8n'i nasıl çalıştırdık? (Bulut / Docker / npx / ngrok farkı)

| Yol | Ne | Neden seçmedik / seçtik |
|---|---|---|
| **n8n Cloud** | n8n'in kendi sunucularında çalışır | 14 gün ücretsiz deneme, sonra aylık ücret → **seçmedik** |
| **Docker** | n8n'i bilgisayarda bir "kutu" (container) içinde çalıştırır | Docker Desktop kurmak/çalıştırmak ağır → **seçmedik** |
| **npx (Node.js)** | `npx n8n start` → n8n doğrudan bilgisayarda çalışır | Ücretsiz, kurulum hafif → **SEÇTİK** |
| **ngrok** | n8n'i çalıştırmaz; bilgisayardaki n8n'e **internetten ulaşılmasını** sağlar | Telegram'ın mesajları bize iletebilmesi için **gerekli** |

**Neden ngrok gerekiyor?** Telegram, bota mesaj gelince bunu bir internet adresine gönderir
(**webhook**). Bizim n8n `localhost:5678`'de; bu adres sadece bizim bilgisayarımızda geçerli,
Telegram oraya ulaşamaz. ngrok ücretsiz planda kalıcı bir adres veriyor
(`enclose-afterglow-sappiness.ngrok-free.dev`) ve oraya gelen istekleri bilgisayarımızdaki 5678'e
aktarıyor (**tünel**).

**Bedeli:** bilgisayar kapanınca veya internet gidince bot durur. Okul (FATİH) ağında SSL denetimi
yüzünden ngrok bağlanamıyor → sunumda telefon hotspot'u. Ayrıntı: `report/kod_notlarim.md` → 26 Eylül bölümü.

### n8n nedir, akışımız nasıl çalışıyor? (sıfırdan)

**n8n**, kod yazmadan "kutuları birbirine bağlayarak" otomasyon kurulan bir araç. Tarayıcıda
`http://localhost:5678` adresinden açılıyor. Beş kavram yeter:

| Kavram | Anlamı | Bizde örnek |
|---|---|---|
| **Akış (workflow)** | Bir işi baştan sona yapan kutular zinciri | "LeadLeaf AI — Bitki Hastalığı Bot (tam v2)", 35 kutu |
| **Düğüm (node)** | Tek bir iş yapan kutu | "Fotoğrafı İndir", "Basic LLM Chain" |
| **Tetikleyici (trigger)** | Akışı başlatan düğüm | "Telegram Trigger": bota bir şey gelince akış başlar |
| **İfade (expression)** | Bir alanın önceki düğümlerin verisinden hesaplanması: `{{ ... }}` | `{{ $json.hastalik }}` = önceki düğümün çıktısındaki hastalık |
| **Kimlik bilgisi (credential)** | Bir hizmete bağlanmak için şifreli anahtar | "Telegram account 2", Anthropic anahtarı, Google hesabı |

Akış **yayınlandığında (Publish)** n8n, Telegram'a "bu bota gelen her şeyi şu adrese gönder" diye
kayıt yapar (webhook). Her mesaj akışı bir kez çalıştırır; buna **çalıştırma (execution)** denir ve
kaydı n8n'in veritabanında tutulur (bkz. 11).

**Veri düğümden düğüme nasıl geçiyor?** Her düğüm bir JSON çıktısı üretir, bir sonraki düğüm onu
`$json` olarak görür. Daha geriye ulaşmak için adla çağrılır: `$('Telegram Trigger').item.json.message.chat.id`
= "tetikleyiciye gelen mesajın sohbet numarası" (cevabı doğru kişiye göndermek için her Telegram düğümü bunu kullanır).

![n8n akışı](sunum/sistem_gorselleri/n8n_akisi.png)

**Dört dal var:** (A) fotoğraf → teşhis ve rapor, (B) yazı → selamlama / bitki düzeltmesi / sohbet,
(C) PDF butonu, (D) rapordan sonra paralel işler (kayıt, uzman, takip). Numaralar şemadakilerle aynı:

| No | Düğüm | Ne yapar |
|---|---|---|
| 1 | Telegram Trigger | Bota gelen mesajı, fotoğrafı ya da buton tıklamasını yakalar; akışı başlatır |
| 2 | Buton mu? | Gelen şey bir buton tıklaması mı ("PDF ister misiniz?")? Evet → C dalı |
| 3 | Fotoğraf var mı? | Mesajda fotoğraf var mı? Evet → A dalı, hayır → B dalı |
| 4 | Telegram - İnceleniyor | "📥 Fotoğrafınız alındı, inceleniyor…" mesajı (düzeltmede "🔁 … yeniden inceleniyor") |
| 5 | Fotoğrafı İndir | Fotoğrafı Telegram sunucusundan kimliğiyle (file_id) indirir |
| 6 | HTTP Request - Predict CNN | Fotoğrafı FastAPI `/predict`'e gönderir (bitki adı, sohbet no, fotoğraf kimliği ile); CNN sonucu gelir |
| 7 | Telegram - Ön Tespit | 4'teki mesajı düzenler: model sonucu + güven + fotoğraf ipucu |
| 8 | HTTP Request - RAG Context | FastAPI `/rag-context`: bu hastalığın 2 bilgi parçası (Chroma) |
| 9 | Basic LLM Chain | Claude'a sistem promptu + CNN sonucu + RAG metnini verir, JSON rapor ister |
| 10 | Anthropic Chat Model | 9'a takılı "model" alt düğümü: hangi Claude modeli (claude-sonnet-5) ve API anahtarı |
| 11 | Rapor JSON'unu Ayrıştır | Kod: Claude'un JSON'unu okur, Telegram mesajını hazırlar; güven düşükse "bitki yanlış mı?" ipucu |
| 12 | Telegram - Durum: Hazır | Durum mesajını "✅ Rapor hazırlandı 👇" yapar |
| 13 | Telegram - Cevap Gönder | Raporu çiftçiye gönderir |
| 14 | Google Sheets - Kaydet | Tabloya bir satır ekler (10 sütun, bkz. 11) |
| 15 | HTTP Request - Raporu Kaydet | FastAPI `/rapor-kaydet`: raporu `data/raporlar/`'a yazar, 12 haneli numara döner |
| 16 | Telegram - PDF Sorusu | "📄 PDF ister misiniz?" + Evet/Hayır butonları (butonda rapor numarası saklı) |
| 17 | Güven < %70 mi? | Model emin değil mi? Evet → 18 |
| 18 | Telegram - Uzmana Bildir | "🔔 Uzman incelemesi gerekiyor" mesajı: çiftçi, tarih, tahmin, güven |
| 19 | Hastalık var mı? | Sonuç "sağlıklı" değilse → takip |
| 20 | Bekle (takip) | Bekler: sunum için 1 dakika (gerçek kullanımda 3 gün) |
| 21 | Telegram - Takip Hatırlatması | "Bitkinizin durumu nasıl? Yeni fotoğraf gönderebilirsiniz" hatırlatması |
| 22 | Selamlaşma mı? | merhaba / selam / /start / yardım mı? Evet → 23 |
| 23 | Telegram - Tanıtım | Sabit tanıtım mesajı: nasıl kullanılır, tanınan 14 bitki |
| 24 | HTTP Request - Son Fotoğraf | FastAPI `/son-foto`: "bu yazı son fotoğraf için bir bitki düzeltmesi mi?" |
| 25 | Bitki düzeltmesi mi? | Evet → 26, hayır → 27 (sohbet) |
| 26 | Yeniden Değerlendirme Hazırla | Kod: yazıyı "son fotoğrafın kimliğini taşıyan fotoğraf mesajına" çevirir → 4 ve 5'e döner |
| 27 | Basic LLM Chain - Sohbet | Genel sohbet (kısa cevap, güvenlik kuralları, marka/doz yok) |
| 28 | Anthropic Chat Model - Sohbet | 27'nin model alt düğümü |
| 29 | Telegram - Sohbet Cevabı | Sohbet cevabını gönderir |
| 30 | Telegram - Butonu Onayla | Telegram'a "butona basıldı" der (butondaki dönen simge durur) |
| 31 | PDF istendi mi? | "Evet, PDF gönder" mi "Hayır" mı? |
| 32 | Telegram - PDF Hazırlanıyor | "📄 PDF raporunuzu hazırlıyorum…" |
| 33 | HTTP Request - PDF Oluştur | FastAPI `/rapor-pdf/{numara}`: kayıtlı rapordan PDF üretir |
| 34 | Telegram - PDF Gönder | PDF'i belge olarak gönderir |
| 35 | Telegram - PDF İstenmedi | "Tamam. Başka bir yaprak fotoğrafı gönderebilirsiniz 🌿" |

**Bir fotoğrafın yolculuğu (A dalı, ~15–20 sn):** 1 → 2 (buton değil) → 3 (fotoğraf var) → 4 ve 5
aynı anda → 6 (CNN) → 7 ve 8 aynı anda → 9+10 (Claude) → 11 → 12–19 aynı anda (cevap, kayıt,
PDF sorusu, uzman kontrolü, takip). En uzun adım Claude (~15 sn).

**n8n'de "kod" var mı?** Evet, iki "Code" düğümü (11 ve 26) JavaScript çalıştırıyor. Kalanı
ayarlardan ve `{{ }}` ifadelerinden oluşuyor. Akışın tamamı `n8n/leadleaf_tam_akis.json`
dosyasında; bu dosya n8n'e "Import from File" ile yüklendi.

---

## 10) Veri setinde kabak (ve bazı bitkiler) — sağlıklı kabak yok

**Kısa cevap:** Doğru, bu PlantVillage veri setinin bir eksikliği ve bizim modelimizin bir sınırı.

5 bitkinin veri setinde **tek bir sınıfı** var:

| Bitki | Tek sınıfı | Sonucu |
|---|---|---|
| Kabak (Squash) | Sadece külleme (powdery mildew) | Model "sağlıklı kabak" **diyemez** |
| Portakal (Orange) | Sadece HLB (citrus greening) | Model "sağlıklı portakal" diyemez |
| Yaban mersini, ahududu, soya | Sadece sağlıklı | Bu bitkilerde **hiçbir hastalığı** tanıyamaz |

**Pratikte ne olur?** Sağlıklı bir kabak yaprağı gelirse:
- Açıklamaya "kabak" yazılmadıysa model onu başka bir bitkinin sağlıklı sınıfına benzetebilir.
- "Kabak" yazıldıysa bitki filtresi sadece kabak sınıflarına bakar (tek sınıf: külleme). Yaprak
  küllemeye benzemiyorsa o sınıfın olasılığı düşük çıkar → "sistemde tanımlı olmayan belirti" +
  uzmana yönlendir. Olasılıkları yeniden ölçeklendirmediğimiz için yanlışlıkla "külleme %100" demiyor.

**Jüriye:** "Veri setinin sınıf yapısından kaynaklanan bir sınırlılık; bitki filtresi ve güven
eşiği sayesinde yanlış kesinlik üretmiyoruz. Sonraki aşamada bu bitkiler için ek veri gerekir."
Panoda: **Sınırlılıklar** sayfası.

---

## 11) n8n'in veritabanı ne?

**Kısa cevap:** n8n kendi ayarlarını bilgisayardaki bir **SQLite** dosyasında tutuyor:
`C:\Users\90539\.n8n\database.sqlite`.

İçinde:
- **Akışlar (workflow):** düğümler, bağlantılar, hangisinin yayında olduğu.
- **Kimlik bilgileri (credentials):** Telegram, Anthropic, Google anahtarları — **şifrelenmiş**
  (şifre anahtarı `.n8n/config` dosyasında).
- **Çalıştırma geçmişi (executions):** her mesajda hangi düğüm çalıştı, ne girdi, ne çıktı.
  "Mısır" hatasını bu kayıtlardan bulduk.

**Karıştırılmaması gerekenler:**

| Nerede | Ne tutuluyor |
|---|---|
| `.n8n/database.sqlite` | n8n'in kendi işleyişi (akış, anahtar, geçmiş) |
| Google Sheets | Bizim **analiz kayıtlarımız** (tarih, hastalık, güven, önlemler…) — n8n oraya yazıyor |
| `data/raporlar/*.json` | PDF butonuna basılınca üretilecek raporlar (FastAPI) |
| `data/son_fotolar.json` | Son fotoğrafın Telegram kimliği, 30 dk (bitki düzeltmesi için) |

**SQLite nedir?** Ayrı bir sunucu gerektirmeyen, tek bir dosyadan oluşan veritabanı. Küçük/orta
uygulamalar için yeterli. Yedek almak = dosyayı kopyalamak (değişikliklerden önce
`database.sqlite.bak_...` yedekleri alındı).

### Projedeki bütün veritabanları ve veri dosyaları (26 Eylül'deki gerçek içerik)

![Bileşenler ve veri depoları](sunum/sistem_gorselleri/mimari.png)

| # | Ne | Türü | Nerede | İçinde ne var (26 Eylül) | Kim yazar / kim okur |
|---|---|---|---|---|---|
| 1 | **Chroma** (RAG bilgi tabanı) | Vektör veritabanı | `rag/chroma_db/` (1,9 MB) | 197 metin parçası, her biri 384 sayılık vektörüyle | `rag/build_index.py` bir kez yazar; FastAPI `/rag-context` okur |
| 2 | **n8n veritabanı** | SQLite | `~/.n8n/database.sqlite` (4,4 MB) | 4 akış, 4 kimlik bilgisi (şifreli), 67 çalıştırma kaydı | n8n'in kendisi |
| 3 | **Google Sheets** | Bulut tablo | Google Drive'da | Her fotoğraf analizi bir satır, 10 sütun | n8n yazar (düğüm 14); sen okursun |
| 4 | **Raporlar** | JSON dosyaları | `data/raporlar/` | 7 rapor (her biri 12 haneli numarayla) | FastAPI yazar; PDF butonuna basılınca okur |
| 5 | **Son fotoğraflar** | JSON dosyası | `data/son_fotolar.json` | Her sohbetin son fotoğraf kimliği, 30 dk | FastAPI `/predict` yazar, `/son-foto` okur |
| 6 | **Tarla defteri** | SQLite | `bot/tarla_defteri.sqlite` | 3 kullanıcı, 2 tarla, 5 gözlem | Streamlit Canlı Demo (eski "Tarla 360" geçmişi) |
| 7 | **CNN modeli** | Model dosyası | `model/model.keras` (31,6 MB) | ~4 milyon öğrenilmiş ağırlık | Colab'da eğitildi; FastAPI okur |
| — | *Görsel RAG indeksi* | *NumPy dosyası* | *`rag/image_embeddings.npz`* | ***Kurulmadı: dosya yok*** | *Kodu hazır (`rag/build_image_index.py`) ama çalıştırılmadı → "benzer referans görseller" özelliği şu an pasif* |

**Google Sheets sütunları:** tarih (İstanbul saati), hastalik (Türkçe), sinif (İngilizce sınıf adı),
guven, onlem (önlemler " | " ile), uzmana_yonlendir, telegram_chat_id, ilk3_tahmin, model_versiyonu, neden.

### Chroma'nın içi — bir kayıt neye benziyor?

Her kayıt 4 şeyden oluşur: **kimlik**, **metin**, **etiket (metadata)** ve **vektör**. İlk 3 kayıt:

| Kimlik | Etiket: sinif | Metin (başı) | Vektör (384 sayının ilk 3'ü) |
|---|---|---|---|
| Apple___Apple_scab__0 | Apple___Apple_scab | "# Elma Karalekesi (Apple Scab) — Venturia inaequalis…" | 0,029 · −0,215 · 0,208 … |
| Apple___Apple_scab__1 | Apple___Apple_scab | "## Belirtiler — Yapraklarda zeytin yeşili/koyu, kadifemsi…" | 0,028 · −0,290 · 0,195 … |
| Apple___Apple_scab__2 | Apple___Apple_scab | "## Uygun koşullar — SERİN ve YAĞMURLU ilkbahar havası…" | 0,145 · 0,014 · 0,189 … |

Parçalar nasıl oluştu: her bilgi dosyası `## ` başlıklarından bölündü (Belirtiler, Uygun koşullar,
Önlemler...). 38 dosya → 197 parça. Her parça, çok dilli embedding modeliyle
(`paraphrase-multilingual-MiniLM-L12-v2`) 384 sayıya çevrildi. Anlamca benzer metinlerin sayıları da
birbirine yakın çıkıyor.

![Chroma'daki parçaların haritası](sunum/sistem_gorselleri/vektor_haritasi.png)

**Haritadan çıkan ders:** Parçalar **bitkiye göre değil, konu başlığına göre** kümeleniyor. Bütün
"Belirtiler" parçaları bir arada, bütün "Kaynak notu" parçaları bir arada. Ölçtük: aynı başlıktaki
parçaların dağınıklığı 0,06–0,27, aynı bitkinin parçalarınınki ortalama 0,86. Bu yüzden aramada
**sınıf filtresi** (`where={"sinif": ...}`) şart. Filtre olmasaydı "geç yanıklık belirtileri"
sorusuna başka bir hastalığın "Belirtiler" parçası gelebilirdi.

### Neden Chroma? SQLite yetmez miydi?

- **SQLite satırları tam eşleşmeyle bulur:** `WHERE sinif = 'Tomato___Late_blight'`. "Anlamca en
  yakın metni bul" diye bir komutu yok.
- **Chroma bir vektör veritabanı:** sorguyu 384 sayıya çevirip kayıtlı vektörler arasından en
  yakınları bulur (**benzerlik araması**). Hızlı arama için kendi indeksini (HNSW) tutar.
- **İşin ilginç yanı:** Chroma içeride SQLite de kullanıyor. `rag/chroma_db/chroma.sqlite3` dosyası
  metinleri ve etiketleri tutuyor, yanındaki klasörler vektör indeksini. Yani **Chroma = SQLite +
  vektör arama katmanı**.
- **Dürüst not:** 197 parçada, SQLite'a vektörleri kaydedip hepsiyle tek tek mesafe hesaplamak da
  çalışırdı. Chroma'yı seçtik çünkü benzerlik araması + sınıf filtresi hazır geliyor, bilgi tabanı
  büyüdüğünde (binlerce parça) yavaşlamıyor ve ücretsiz, yerel çalışıyor (ayrı sunucu gerekmiyor).

---

## 12) Başkaları "PDF yükleyerek eğitmiş" — o ne?

**Kısa cevap:** Büyük ihtimalle **eğitim değil, RAG**. PDF'leri yükleyip LLM'in cevap verirken
onlardan faydalanmasını sağlamışlar. Biz de aynı şeyi yaptık, sadece kaynak olarak PDF yerine
kendi yazdığımız, kontrol ettiğimiz metin dosyalarını kullandık.

**"Eğitmek" ile "bilgi vermek" farkı (jüri bunu sorabilir!):**

| | Eğitmek (training / fine-tuning) | Bilgi vermek (RAG) |
|---|---|---|
| Ne değişir | Modelin **ağırlıkları** değişir | Model **değişmez**; soruyla birlikte ilgili metin verilir |
| Maliyet | GPU, saatler, çok veri | Ucuz, anında |
| Bilgi güncelleme | Yeniden eğitmek gerekir | Dosyayı değiştir, indeksi yeniden kur |
| Bizde | **CNN'i eğittik** (Colab, GPU) | **Claude'u eğitmedik**, RAG ile bilgi verdik |

"PDF yükleyip eğittim" çoğu zaman yanlış bir ifade; ChatGPT'ye / n8n'e / bir vektör veritabanına
PDF yüklemek RAG'dir.

**Neden biz PDF değil de kendi yazdığımız dosyaları kullandık?** PDF'ten metin çıkarmak
(tablolar, sayfa düzeni) gürültülü olur; ayrıca ilaç dozu gibi vermek istemediğimiz bilgiler
içerebilir. Kendi yazdığımız dosyalarda her cümleyi kontrol ettik, pestisit politikamıza
(marka/doz yok) uygun.

**Yani PDF kullandık mı?** Hayır. Bilgi kaynağı `agent/knowledge/` klasöründeki 38 metin (markdown)
dosyası; her sınıf için bir tane (ör. `Tomato___Late_blight.md`: etken, belirtiler, uygun koşullar,
karıştırılabilecek hastalıklar, önlemler, kaynak notu).

**Dürüst olunması gereken nokta:** Bu 38 dosya **yapay zekâ yardımıyla hazırlandı** ve bir **ziraat
mühendisine kontrol ettirilmedi**. Jüri sorarsa: "Bilgi dosyaları genel kaynaklara göre hazırlandı,
uzman doğrulaması yapılmadı; gerçek kullanımdan önce bir ziraat mühendisinin gözden geçirmesi
sonraki adım." Sistem bu yüzden her raporda "kesin teşhis değildir" uyarısı veriyor.

---

## 13) Agent'ı sen mi oluşturdun, direkt mi bağladın?

**Kısa cevap:** Claude'u kendimiz eğitmedik; API üzerinden **bağlandık**. Ama ne yapacağını,
neyle çalışacağını ve neyi yapmayacağını belirleyen her şeyi **biz tasarladık**: sistem promptu,
RAG bilgi tabanı, akış, güvenlik kuralları.

**Dürüst ve teknik doğru cevap:**
- n8n'de iki tür LLM düğümü var: **Basic LLM Chain** (sabit adımlar: girdi → prompt → cevap) ve
  **AI Agent** (LLM hangi aracı ne zaman kullanacağına **kendisi karar verir**).
- Biz bilerek **Basic LLM Chain** kullandık. Hangi adımın ne zaman çalışacağını n8n akışı
  (CNN → RAG → Claude → kayıt) belirliyor. Neden:
  1. **Öngörülebilir:** her fotoğrafta aynı adımlar çalışır; sağlık/tarım gibi riskli bir alanda
     LLM'in "RAG'e bakmaya gerek yok" diye karar vermesini istemeyiz.
  2. **Ucuz ve hızlı:** tek LLM çağrısı. Agent birkaç kez düşünüp araç çağırır.
  3. **Güvenli:** teşhisi CNN yapıyor, LLM sadece açıklıyor.
- Bizim yazdıklarımız: 8 kurallı sistem promptu (`agent/prompt_taslagi.md`): JSON çıktı biçimi,
  Türkçe adı ve güven değerini değiştirmeme, marka/doz vermeme, %70 altı uzmana yönlendirme,
  prompt enjeksiyonuna ("API anahtarını ver") karşı kural. Bir de ayrı sohbet dalının promptu.
- "LLM-Agent" bootcamp'in verdiği ad. Bizim sistemde "agent" davranışı **akışın bütünü**
  (fotoğraf mı, yazı mı, selam mı, düzeltme mi → hangi dala gideceğine karar veren n8n).

---

## 14) Streamlit için gereken dosyalar

**Çalıştırma:** `.venv\Scripts\python -m streamlit run ui/sunum.py` (Canlı Demo için önce FastAPI açık olmalı).

| Dosya | Görevi |
|---|---|
| `ui/sunum.py` | **Giriş noktası.** Sayfa ayarları, logo, menü grupları, modeli bir kez yükler |
| `ui/ortak.py` | Ortak parçalar: `SAYFALAR` listesi (sıra + Önceki/Sonraki), stil (CSS), model yükleme |
| `ui/sayfalar/01_ozet.py` … `10_sinirliliklar.py` | Her sayfa ayrı dosya (Özet, Veri Seti, EDA, CNN, Model Seçimi, Fine-Tuning, Test, Grad-CAM, RAG, Sınırlılıklar) |
| `ui/app.py` | **Canlı Demo** sayfası (fotoğraf yükle → FastAPI'ye gönder → rapor + PDF) |
| `.streamlit/config.toml` | Tema (lacivert renkler) |
| `ui/logo.png` | Logo |
| `model/model.keras` | CNN ve Grad-CAM sayfalarında canlı kullanılan model |
| `model/model_38sinif/` → `test_sonuclari.npz`, `veri_analizi.json`, `split_manifest.json`, `model_comparison.csv`, grafik PNG'leri | Test, EDA ve karşılaştırma sayfalarının verisi |
| `model/tubitak/*.csv`, `*.png` | 3 mimari ve önceki/gelişmiş karşılaştırması |
| `data/plantvillage/raw/color/` | Veri Seti sayfasında gerçek fotoğraflarda gezinmek için (git'e girmiyor) |
| `.env` | `ANTHROPIC_API_KEY` — yoksa Canlı Demo raporu Claude yerine yerel şablonla üretilir |
| `requirements.txt` | Gerekli Python paketleri (`streamlit`, `pandas`, `tensorflow-cpu`, …) |

**Streamlit nedir?** Python koduyla web arayüzü yapan bir kütüphane. HTML/JavaScript yazmadan
`st.title()`, `st.slider()`, `st.image()` gibi komutlarla sayfa oluşturuluyor. Kullanıcı bir şeye
dokununca Streamlit sayfanın kodunu baştan çalıştırır.

---

## 15) Macro F1, Macro Precision, Macro Recall, Macro AUC

Önce tek bir sınıf için ("geç yanıklık"):

- **Precision (kesinlik):** Model "geç yanıklık" dediklerinin **yüzde kaçı gerçekten** geç yanıklık?
  → "Alarm verdiğinde ne kadar haklı?" (Düşükse: yanlış alarm çok.)
- **Recall (duyarlılık):** Gerçekte geç yanıklık olanların **yüzde kaçını yakaladı?**
  → "Hastalıkları kaçırıyor mu?" (Düşükse: hasta yaprağı "sağlıklı" deyip geçiyor, tarımda tehlikeli.)
- **F1:** Precision ile recall'un dengeli ortalaması (harmonik ortalama). İkisinden biri düşükse F1 de düşer.

**Örnek:** Test setinde 100 gerçek geç yanıklık var. Model 95 fotoğrafa "geç yanıklık" dedi; bunların
90'ı doğru, 5'i aslında başka hastalık.
- Precision = 90 / 95 = **%94,7**
- Recall = 90 / 100 = **%90**
- F1 = 2 × 0,947 × 0,90 / (0,947 + 0,90) ≈ **%92,3**

**"Macro" ne demek?** Bu ölçüleri **38 sınıfın her biri için ayrı** hesaplayıp **basit ortalama**
almak. Her sınıf eşit ağırlıkta: toplam 152 fotoğraflı sağlıklı patates (en küçük sınıf) ile
5.507 fotoğraflı portakal HLB (en büyük sınıf) aynı öneme sahip.
- **Neden önemli?** Doğruluk (accuracy) çok fotoğraflı sınıfların başarısını öne çıkarır.
  Az fotoğraflı bir sınıfta kötü olsak bile doğruluk yüksek görünebilir. Macro ölçüler bunu gizlemez.
- Bizde: doğruluk **0,9902**, macro F1 **0,9859**. Macro F1 biraz daha düşük → az fotoğraflı
  bazı sınıflar ortalamanın hafifçe altında. Fark küçük, yani model sınıflar arasında dengeli.

**AUC (ROC eğrisinin altındaki alan):** Modelin "doğru sınıfa, yanlış sınıflardan daha yüksek
olasılık verme" becerisi, **eşikten bağımsız**. 1,0 = mükemmel sıralama, 0,5 = yazı-tura.
Macro AUC = her sınıf için "bu sınıf mı, değil mi" (bire karşı hepsi) AUC'si, sonra ortalama.
Bizde **0,9999**.

| Model | Doğruluk | Macro Precision | Macro Recall | Macro F1 | Macro AUC |
|---|---|---|---|---|---|
| **EfficientNetB0, 38 sınıf (üretimde)** | 0,9902 | 0,9882 | 0,9840 | 0,9859 | 0,9999 |
| EfficientNetB0 gelişmiş, 5 sınıf | 0,9762 | 0,9748 | 0,9693 | 0,9716 | 0,9993 |
| EfficientNetB0, 5 sınıf | 0,9468 | 0,9454 | 0,9370 | 0,9400 | 0,9958 |
| MobileNetV2, 5 sınıf | 0,9095 | 0,9022 | 0,8987 | 0,8985 | 0,9879 |
| MobileNetV3Small, 5 sınıf | 0,8794 | 0,8867 | 0,8479 | 0,8523 | 0,9866 |

**Jüri tuzağı:** "%99 doğruluk varsa neden tarlada mısır dedi?" → Bu sayılar PlantVillage **test
setinde**, yani eğitimle aynı tarzda (laboratuvar) fotoğraflarda. Tarla fotoğrafı farklı bir
dağılım (bkz. 16). Dürüst cevap bu.

---

## 16) Mısır'ı yanlış algıladı, "Bu domates" dedim

**Kısa cevap:** Açıklamasız tarla fotoğrafında model 38 sınıfın hepsi arasında seçti ve **%29
güvenle** "sağlıklı mısır" dedi. Güven %70'in altında olduğu için sistem "emin değilim → uzmana
yönlendir" olarak işaretledi, yani eşik doğru çalıştı. Sonradan yazılan "Domates" sohbet dalına
düşüyordu ve sohbet dalı fotoğrafı bilmiyordu. **Düzeltildi:** artık bitki adı yazılınca aynı
fotoğraf o bitkiye göre yeniden değerlendiriliyor.

**Neden yanlış bildi? (alan kayması / domain shift):** PlantVillage fotoğrafları laboratuvarda, düz
arka planda, tek yaprak. Tarla fotoğrafında arka plan, ışık ve açı farklı. Grad-CAM'de gördüğümüz
"bazen arka plana bakıyor" bulgusu bununla uyumlu.

**Tüm ayrıntı (hata nasıl bulundu, kod, testler, jüri cevapları):** `report/kod_notlarim.md` →
"26 Eylül 2026" bölümü.

---

## 17) Canlı Demo'da bitki neden önceden seçtirilmiyor? ("Analiz ayarları" kaldırıldı)

**Eski hâl:** Sol menüde "Analiz ayarları → Bitki" vardı ve **varsayılan olarak Domates seçiliydi**.
Mısır fotoğrafı yüklenince ana sonuç "Domates: sistemde tanımlı olmayan belirti" çıkıyor, hemen
altında ise "Yaygın Pas (Mısır) %100" görünüyordu: çelişkili bir ekran. Ayrıca jüri "sonucu önceden
yaptığınız bir seçimle yönlendiriyorsunuz" diye eleştirebilirdi.

**Yeni hâl (26 Eylül):** Model önce **filtresiz** tahmin ediyor. Sonucun hemen altında soruluyor:
"Model bu yaprağı **Mısır** yaprağı olarak değerlendirdi. Doğru mu?"
- **Evet** → sonuç onaylanır ("Bitki kullanıcı tarafından doğrulandı").
- **Hayır, bu bir: [Domates]** → aynı fotoğraf sadece domates sınıfları arasından yeniden
  değerlendirilir. Yaprak domates sınıflarına benzemiyorsa teşhis uydurulmaz, uzmana yönlendirilir.

**Jüriye:** "Model önce kendi kararını veriyor, kullanıcı sonra doğruluyor ya da düzeltiyor.
Telegram'da da aynı mantık var: fotoğraftan sonra bitki adını yazmak yeterli. Yani insan
kontrolü, modelin kararını önceden etkileyerek değil, sonradan doğrulayarak devreye giriyor
(**human-in-the-loop**, döngüdeki insan)."

**Teknik not:** Streamlit her butona basışta sayfanın kodunu baştan çalıştırır. Sonucun
kaybolmaması için analiz sonucu `st.session_state["analiz"]`'de (oturum hafızası) saklanıyor.
Bitki düzeltilince sadece o kayıt sıfırlanıp aynı fotoğraf yeniden gönderiliyor (`ui/app.py`).

---

## 18) "Grafikte 8. epoch'tan önce daha iyiyken neden 8'de durdurdunuz?"

**Kısa cevap:** 8. epoch'ta durdurmadık. Grafikteki kesikli çizgi durma noktası değil, **aşama
değişimi**. Eğitim 28 epoch sürdü ve son model, 8'den önceki en iyi noktadan **daha iyi**.

Öğrenme eğrisi (`model/model_38sinif/ogrenme_egrisi_EfficientNetB0_38sinif.png`, panoda Fine-Tuning sayfası):

| Epoch | Aşama | Doğrulama doğruluğu (yaklaşık) |
|---|---|---|
| 0–7 | 0: sadece kafa eğitiliyor (en fazla 8 epoch) | 0,935 → **0,969** (epoch 6'da tepe) |
| 8 | 1 başlıyor: gövdenin son %15'i açıldı | **0,959'a düşüş** (eğitim doğruluğu 0,90) |
| 8–13 | 1: son %15 açık | 0,959 → 0,980 |
| 14–19 | 2: son %30 açık | 0,980 → 0,985 |
| 20–27 | 3: son %40 açık | 0,985 → **~0,989** |

**8. epoch'taki düşüş neden oldu?** Gövdenin katmanları o ana kadar donuktu. 8. epoch'ta ilk kez
eğitime açıldılar ve model yeniden derlendi; optimizer (Adam) sıfırdan başladı. Açılan milyonlarca
ağırlığın aynı anda kıpırdaması modeli kısa süre sarstı. Buna literatürde "unfreezing shock"
(çözme şoku) denir. Model 2 epoch içinde toparlandı ve önceki tepeyi geçti.
Muhtemel katkıda bulunan bir etken: açılan kısımdaki **BatchNorm** katmanlarının ölçek/kaydırma
parametreleri de eğitime açıldı. Keras'ın önerisi, ince ayarda BatchNorm katmanlarını donuk tutmak
(sonraki deneyde denenebilir).

**Model 8'e kadar mı eğitildi, yoksa en iyi noktada mı kaldı?** Her aşamada
`EarlyStopping(restore_best_weights=True)` vardı: aşamanın sonunda model **en iyi doğrulama
sonucunu veren epoch'un ağırlıklarına geri dönüyor**. Aşama 0'da en iyisi epoch 6'ydı (0,969 >
epoch 7'deki 0,968), yani aşama 1 epoch 6'nın ağırlıklarıyla başladı.

**Neden eğitim doğruluğu çoğu yerde doğrulamadan düşük?** Ezber değil, tersi: eğitimde
**augmentation** (döndürme, yakınlaştırma) ve **Dropout** açık, yani model eğitimde daha zor bir
sınavdan geçiyor. Doğrulamada ikisi de kapalı. Ayrıca eğitim doğruluğu epoch boyunca değişen
ağırlıkların ortalaması.

**Dürüst eleştiri (jüri sorabilir):** Aşamaların hiçbiri erken durmadı; hepsi üst sınıra kadar
koştu (8+6+6+8 = 28). Eğri sonda hâlâ hafif yükseliyor, yani biraz daha eğitmek küçük bir kazanç
sağlayabilirdi. Colab süresi nedeniyle burada durduk.

---

## 19) Projenin değerlendirmesi — yazılımcı ve kullanıcı gözüyle (sunumda dürüstçe söyle)

Jüride en güçlü duruş: **güçlü yanları sayılarla, eksikleri sizden önce siz söyleyin**. Eksiğini
bilen ve sebebini açıklayan ekip, "her şey mükemmel" diyen ekipten daha çok puan alır.

### Yazılımcı gözüyle — güçlü yanlar
1. **Doğru deney düzeni:** tabakalı %70/%15/%15 bölme, sabit seed, bölmenin dosyaya kaydedilmesi;
   test evde yeniden üretildi (%98,98 ≈ Colab %99,02).
2. **Kontrollü karşılaştırma:** önce 3 mimari aynı koşulda, sonra kazanan mimaride sadece eğitim
   tarifi değişti (0,9468 → 0,9762). Hangi iyileşmenin nereden geldiği biliniyor.
3. **Belirsizliği yönetme:** %70 güven eşiği, bitki filtresi (olasılıklar yeniden ölçeklenmiyor →
   yapay kesinlik yok), kullanıcı doğrulaması (human-in-the-loop).
4. **LLM'i sınırlama:** teşhisi CNN yapıyor; Claude hastalık adını ve güveni değiştiremiyor,
   RAG'deki doğrulanmış bilgiyle konuşuyor, marka/doz vermiyor, prompt enjeksiyonuna kuralı var.
5. **Açıklanabilirlik ve dürüstlük:** Grad-CAM, karışıklık matrisi, yanlış bilinenler, sınırlılıklar sayfası.
6. **Uçtan uca çalışan ürün:** Telegram → n8n → CNN → RAG → Claude → Sheets kaydı → PDF.
7. **Zarif bozulma:** model yoksa demo modu, LLM yoksa şablon rapor, FastAPI kapalıysa sohbet çalışır.

### Yazılımcı gözüyle — eksikler (önem sırasıyla)
1. **Gerçek tarla fotoğrafıyla ölçüm yok (en büyük eksik).** %99 laboratuvar fotoğraflarında.
   Gerçek bir domates fotoğrafı %29 güvenle "mısır" çıktı; şeftali örneği %89 güvenle yanlıştı.
   Sonraki adım: 50–100 tarla fotoğrafı toplayıp ayrı ölçmek; gerçek ortam fotoğraflarından oluşan
   bir veri setiyle (ör. PlantDoc) eğitimi zenginleştirmek.
2. **Kapalı küme sorunu:** model "bilmiyorum" diyemez, 38 sınıftan birini seçmek zorunda. Eşik ve
   bitki filtresi riski azaltıyor ama çözmüyor. İleri çözüm: "diğer/bilinmeyen" sınıfı ya da
   dağılım dışı (out-of-distribution) tespiti.
3. **Veri setinin sınıf boşlukları:** sağlıklı kabak ve sağlıklı portakal yok; yaban mersini,
   ahududu ve soyada hastalık yok (bkz. 10).
4. **Otomatik test yok.** Testler elle ve oturum içinde yapıldı. 26 Eylül'de 4 gizli hata
   yakalandı (sohbet dalının fotoğrafı bilmemesi, Claude cevabındaki "thinking" bloğu yüzünden
   raporun sessizce şablona düşmesi, token sınırının JSON'u kesmesi, tanımsız sonuçta RAG'in
   alakasız bilgi getirmesi). Birim testleri olsaydı daha erken yakalanırdı.
5. **Hataların sessiz kalması:** yedek (şablon) rapor gerçek hatayı gizliyordu ve ekranda yanlış
   sebep ("anahtar tanımlı değil") yazıyordu. Düzeltildi, artık gerçek sebep görünüyor.
6. **Altyapı:** her şey kişisel bilgisayarda. Bilgisayar kapanınca bot durur; okul ağı (SSL
   denetimi) ngrok'u engelliyor. Gerçek kullanım için bir sunucu (VPS) gerekir.
7. **İki ayrı rapor yolu:** Telegram (n8n promptu) ve Streamlit (`agent/report.py`) aynı işi ayrı
   kodla yapıyor; biri güncellenip öbürü unutulabilir (daha önce oldu).
8. **Kişisel veri:** Google Sheets'e Telegram sohbet kimliği yazılıyor. Gerçek kullanımda KVKK
   aydınlatma metni ve açık rıza gerekir.
9. **Eğitim tarafı:** tek bir eğitim koşusu (farklı seed'lerle tekrar yok, sonuçların oynaklığı
   bilinmiyor), hiperparametre araması yok, aşamalar üst sınıra kadar koştu (bkz. 18).
10. **Hava durumu riski sezgisel:** nem/yağış eşikleri kalibre edilmedi; sadece bilgi amaçlı.
11. **Bilgi tabanı uzman onaylı değil:** 38 bilgi dosyası yapay zekâ yardımıyla hazırlandı, bir
    ziraat mühendisi gözden geçirmedi (bkz. 12).
12. **Görsel RAG pasif:** "benzer referans görseller" özelliğinin kodu var ama indeksi
    (`rag/image_embeddings.npz`) hiç kurulmadı; ekranda bu bölüm görünmüyor (bkz. 11).

### Kullanıcı (çiftçi) gözüyle
| Güzel olan | Zorlayan / eksik |
|---|---|
| Telegram: yeni uygulama kurmak yok, alışık olunan arayüz | Cevap ~15–20 saniye sürüyor (adım adım durum mesajı bekleme hissini azaltıyor) |
| Türkçe, sade dil; PDF rapor | 14 bitki ile sınırlı; Türkiye'de yaygın birçok ürün yok |
| Emin olmadığında "uzmana danışın" diyor | İlaç adı ve doz vermiyor (bilerek, yasal ve güvenlik nedeniyle), bu bazı çiftçileri hayal kırıklığına uğratabilir |
| Yanlış bitkide düzeltme yapılabiliyor ("Domates" yazmak) | İyi fotoğraf çekmeyi bilmek gerekiyor (tek yaprak, yakın, gölgede) |
| Selamlaşmada ne yapacağını anlatıyor | İnternet şart; kırsalda bağlantı zayıf olabilir |

### Sunumdan önce yapılacaklar (kontrol listesi)
- [x] n8n "tam v2" ile Telegram'dan açıklamasız fotoğraf + "Domates" testi (26 Eylül, çalıştı)
- [x] Değişikliklerin commit edilmesi
- [ ] Telefon hotspot'u ile tam prova (okul ağında çalışmaz)
- [ ] **Yedek demo videosu** (ağ ya da bilgisayar sorun çıkarırsa sunum kurtulur)
- [ ] Bu rehberdeki 2, 2b, 3, 8, 15, 18. soruların sesli tekrarı

---

## 20) Anthropic (Claude) anahtarı olmadan RAG yapamaz mıydım?

**Kısa cevap:** RAG'in **arama** kısmı için anahtar gerekmiyor, zaten bilgisayarda ücretsiz
çalışıyor. Anahtar sadece **raporu yazan dil modeli** (Claude) için gerekiyor. (Ve yine: RAG ile bir
şey "eğitilmez", bkz. 12.)

RAG'i iki parçaya ayırın:

| Parça | Ne yapar | Anahtar gerekir mi? |
|---|---|---|
| **Arama (retrieval)** | Hastalık adına göre en ilgili 2 bilgi parçasını bulur | **Hayır.** Embedding modeli ve Chroma bilgisayarda çalışıyor, ücretsiz |
| **Yazma (generation)** | Bulunan parçalara dayanarak çiftçiye sade bir rapor yazar | **Evet**, Claude kullanıldığı için |

**Anahtar olmasaydı seçenekler:**
1. **Şablon rapor (projede zaten var):** anahtar yoksa ya da Claude hata verirse `agent/report.py`
   sabit bir şablonla rapor üretiyor. Sistem çökmüyor ama rapor kişiselleşmiyor.
2. **Bulunan metni doğrudan göstermek:** RAG'in bulduğu parçalar hiç LLM'e verilmeden çiftçiye
   gösterilebilir. Anahtar gerekmez, ama metin sadeleştirilmemiş, uzun ve teknik olur.
3. **Yerel, açık kaynak bir dil modeli** (ör. Ollama ile bilgisayarda çalışan Llama/Qwen gibi
   modeller): ücretsiz ve anahtarsız. Bedeli: güçlü bir bilgisayar ister, yavaş çalışır, Türkçesi ve
   kurallara (marka/doz verme) uyumu Claude kadar güvenilir değil.

**Biz neden Claude'u seçtik?** Türkçesi güçlü, verilen kurallara (JSON biçimi, doz yok, adı ve güveni
değiştirme) iyi uyuyor, n8n'de hazır düğümü var. Maliyeti: rapor başına kuruşlar düzeyinde, deneme
kredisiyle karşılandı.

**Jüriye:** "RAG'in arama kısmı tamamen yerel ve ücretsiz. Sadece raporu sade dille yazdırmak için
Claude kullanıyoruz. Claude'a ulaşılamazsa sistem çökmüyor, şablon rapora düşüyor."

---

## Terimler sözlüğü (hızlı tekrar)

| Terim | Bir cümlede |
|---|---|
| Epoch | Eğitim setinin tamamının modele bir kez gösterilmesi |
| Batch | Modele aynı anda gösterilen fotoğraf grubu (bizde 32) |
| Öğrenme hızı (learning rate) | Ağırlıkların her adımda ne kadar değiştirileceği |
| Overfitting | Ezberleme: eğitimde çok iyi, yeni veride kötü |
| Dropout | Eğitimde nöronların bir kısmını (%20) rastgele kapatıp ezberi zorlaştırmak |
| Augmentation | Eğitim fotoğraflarını çevirip döndürüp yakınlaştırarak çeşitlendirmek |
| Softmax | Son katmanın çıktısını, toplamı %100 olan olasılıklara çeviren fonksiyon |
| Güven (confidence) | Softmax'ın en yüksek olasılığı. Bizde %70 altı → uzmana yönlendir |
| Embedding | Bir metni/görseli anlamını taşıyan bir sayı listesine çevirmek |
| API | Programların birbirine iş yaptırma kapısı (ör. Claude API) |
| Webhook | "Bir şey olunca şu adrese haber ver" kaydı (Telegram → n8n) |
| Endpoint (uç nokta) | Bir servisteki belirli adres (ör. `/predict`, `/rag-context`) |
