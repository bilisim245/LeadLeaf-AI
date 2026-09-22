# İLERLEME — LeadLeaf AI (Bitki Hastalığı Ön Değerlendirme Sistemi)

**Son güncelleme:** 2026-09-21

**⚠️ TAKVİM UYARISI (2026-09-18):** Teslime **7 gün kaldı** (≈2026-09-25) — orijinal "10 gün" bootcamp Gün-numaralandırması artık takvim günüyle 1:1 örtüşmüyor, sıkıştırılmış plan gerekiyor. Kalan Gün 1/2-3/5-10 işleri 7 takvim gününe şöyle dağıtıldı (bkz. altta "7 günlük sıkışık plan"): **en kritik/tek gerçek darboğaz hâlâ Colab eğitimi — henüz çalıştırılmadı, bugün (Gün 1) yapılması şart, aksi halde geri kalan her şey gecikir.**

### 7 günlük sıkışık plan
- **Gün 1 (2026-09-18, bugün):** Colab eğitimi çalıştır (Sen, kritik yol) + n8n Cloud hesabı/tünel/Telegram token hazırlığı (Sen, paralel)
- **Gün 2 (2026-09-19):** Model dosyalarını `model/` + `model/tubitak/`'a yerleştir, gerçek modelle `/predict`+`/predict_compare` testi (Ben); n8n workflow import + credential bağlama + ilk Telegram testi (Beraber)
- **Gün 3 (2026-09-20):** Prompt geliştirme + %70 güven eşiği IF node (Beraber); Sheets entegrasyonu (Ben taslak, Sen credential)
- **Gün 4 (2026-09-21):** PDF üretimi + uçtan uca uç durum testleri: yaprak olmayan görsel, düşük güven, bulanık foto (Beraber)
- **Gün 5 (2026-09-22):** Tampon gün (beklenmeyen hatalar) + rapor taslağına başla (Ben taslak, Sen sonuçlar)
- **Gün 6 (2026-09-23):** Rapor tamamla + sunum slaytları
- **Gün 7 (2026-09-24/25):** Demo videosu + son kontrol + GitHub push

**MİMARİ KARARI (2026-09-16) — TÜBİTAK araştırma düzenine geçiş: tek model yerine 3 model karşılaştırması.**
`notebooks/01_train_model_colab.py` baştan yazıldı: veri artık TEK SEFERDE, sınıf bazında stratified
%70/%15/%15 train/valid/test'e bölünüyor (`split_manifest.json`'a hangi dosyanın nereye düştüğü kaydediliyor —
tekrarlanabilirlik + sızıntı denetimi). Aynı split üzerinde MobileNetV2, MobileNetV3Small, EfficientNetB0
AYRI AYRI ama AYNI koşullarla (aynı augmentation, aynı epoch/early-stopping, aynı fine-tune oranı — son %25
katman) eğitiliyor. Her model KENDİ preprocess_input'unu kullanıyor — MobileNetV3Small ve EfficientNetB0'ın
preprocess_input'u Keras kaynağında pass-through (rescaling zaten modelin içinde), MobileNetV2'ninki gerçekten
[-1,1]'e ölçekliyor; bu ayrım "çift normalizasyon" hatasından KASITLI olarak kaçınıyor. Değerlendirme SADECE
bağımsız test setinde (val değil) yapılıyor: accuracy, macro precision/recall/F1, model boyutu (MB), görüntü
başına ortalama inference süresi (ms) — hepsi `model_comparison.csv`'ye yazılıyor, her model için ayrı confusion
matrix + öğrenme eğrisi PNG'si üretiliyor. Çıktı `leadleaf_tubitak_models.zip` (3× `.keras` + `class_names.json`
+ `split_manifest.json` + `model_comparison.csv` + grafikler + `demo_images/`).

`inference/app.py`'ye YENİ ve AYRI bir `/predict_compare` endpoint'i eklendi (üretim `/predict`'i etkilemiyor —
o hâlâ tek model + n8n/Telegram akışı için). `/predict_compare` `model/tubitak/` klasöründeki 3 modeli (varsa)
yükleyip aynı fotoğrafı üçüne birden veriyor; eksik model varsa SADECE o model tek başına DEMO MODU'na düşüyor
(servis çökmez, proje genelindeki DEMO MODU felsefesiyle tutarlı). Yanıt, model uzlaşması (tam/kısmi/yok —
kural tabanlı, ML tahmini DEĞİL) ve %70 güven eşiğini birleştiren açıklanabilir bir tavsiye içeriyor.

`ui/app.py`'ye üçüncü sekme eklendi: **"🔬 Model Karşılaştırma"** — tek fotoğrafı `/predict_compare`'e gönderip
3 modelin sonucunu tablo halinde, uzlaşma durumunu ve eşik uyarısını gösteriyor. `.env.example`'a
`TUBITAK_MODEL_DIR=./model/tubitak` eklendi. Eski tek-modelli `01_train_model_colab.py` akışı (MVP'nin üretim
tarafı, n8n/Telegram) DEĞİŞMEDİ — bu TÜBİTAK karşılaştırması onun ÜZERİNE eklenen ayrı bir araştırma katmanı.
**Aktif aşama:** Gün 4 tamamlandı (inference + yerel Streamlit demo çalışıyor) — Gün 5 (n8n Cloud) sırada
**Kimler:** Sen = hesap/çalıştırma/test · Ben = tüm kod + rapor taslağı · Beraber = entegrasyon

**ÖNEMLİ KAPSAM KARARI (2026-09-11):** 10 günlük teslim MVP'dir. Hava durumu, tarla defteri, bölgesel uyarı, konum haritası, 38 sınıf → TÜBİTAK/TEKNOFEST "Sonraki Aşama"sına bırakıldı (ayrıntı: README.md). İlaç dozu/bekleme süresi LLM'e yazdırılmaz — güvenlik riski.

**MİMARİ KARARI (2026-09-11):** Bootcamp "prompt geliştirme n8n'de" istiyor. LLM-Agent çağrısı ve Telegram botu Python'dan **n8n'in içine** taşındı (n8n Telegram Trigger/Send + HTTP Request → Claude API, prompt n8n node'unda). Python'da sadece FastAPI `/predict` (CNN) kalıyor. `agent/agent.py` ve `bot/telegram_bot.py` planları iptal edildi (henüz yazılmamışlardı); yerine `agent/prompt_taslagi.md` (n8n'e yapıştırılacak prompt) geldi.

**MİMARİ KARARI (2026-09-15):** n8n artık **local Docker değil, n8n Cloud** (kurulum hızı için). Detay: `n8n/README_N8N.md`. Ayrıca Kaggle eğitimi bitmeden uçtan uca görselleştirme yapabilmek için `inference/app.py`'ye **DEMO MODU** eklendi (model dosyası yoksa rastgele ama tutarlı sonuç döner, model gelince otomatik gerçek moda geçer) ve `ui/app.py` (Gradio) + `agent/report.py` (sadece bu yerel demo için, prompt_taslagi.md ile aynı sistem promptu) yazıldı.

**KARAR (2026-09-15):** Model eğitimi için **Kaggle yerine Google Colab** (sıfır yerel kurulum, tarayıcıda ücretsiz GPU). `notebooks/01_train_model_colab.py` yazıldı — Kaggle hesabı SADECE veri setini indirmek için gerekiyor (kaggle.json ile, API üzerinden), eğitimin kendisi Colab'ın GPU'sunda çalışıyor. `01_train_model_kaggle.py` alternatif olarak duruyor (Kaggle'da çalıştırmak isteyen olursa).

**KAPSAM GÜNCELLEMESİ (2026-09-15) — "derinlik" geri bildirimi:** Referans alınan bir müşteri analitiği dashboard'u (trend grafiği + senaryo analizi + sıralı aksiyon önerisi) ile karşılaştırıldığında, tek-fotoğraf/tek-rapor akışının yüzeysel kaldığı değerlendirildi. Buna karşılık:
1. **`bot/db.py` (tarla defteri) ve `agent/weather.py` (hava durumu) "bonus" aşamasından MVP'ye çekildi** — ikisi de zaten yazılıp test edilmişti, sadece bağlı değildi. Artık `ui/app.py` bunları kullanıyor.
2. **`ui/app.py` "Tarla 360" dashboard'una dönüştürüldü:** geçmiş trend grafiği (bu tarlanın önceki gözlemleri), hava durumu bazlı mantar riski, bölgesel kümelenme notu ("Serik'te son 7 günde N çiftçi daha aynı hastalığı bildirdi") ve **kural tabanlı senaryo analizi** ("Mevcut durum" vs "Kültürel önlem" vs "Tekrar kontrol" vs "Uzmana danış", her biri için modellenen risk ve fark). ⚠️ Senaryo tablosu AÇIKÇA "ML tahmini değil, kural tabanlı simülasyon" olarak etiketlendi — sahte kesinlik/overclaiming riskinden kaçınmak için (bkz. Etik notu).
3. **`notebooks/00_veri_kesfi.py` yazıldı** — veri setini varsayımla değil sayılarla incelemek için: sınıf dağılımı/dengesizliği, görsel boyutu istatistikleri, ve **train/valid arasında perceptual-hash ile sızıntı (data leakage) kontrolü** (bu "Augmented" veri setine literatürde yöneltilen bilinen bir eleştiri — aynı orijinal fotoğrafın augment'lerinin hem train hem valid'e sızması doğrulama doğruluğunu olduğundan iyimser gösterebilir).

Bu, n8n Telegram akışını GECİKTİRMEZ (paralel ilerliyor) — sadece yerel demonun ve rapor kalitesinin derinliğini artırıyor.

**KARAR (2026-09-16) — Arayüz: Gradio'dan Streamlit'e geçildi.** Geri bildirim: Gradio'da kart görünümü için kullanılan özel HTML/CSS istenen akışı vermedi ("çok kötüydü"). `ui/app.py` Streamlit ile YENİDEN YAZILDI — aynı işlevsellik (trend, hava durumu, senaryo tablosu, benzer görsel galerisi, RAG notu) artık Streamlit'in native bileşenleriyle (st.metric, st.success/warning/error, st.line_chart, st.dataframe) CSS'siz, sidebar + üstten-alta doğal akışla sunuluyor. `requirements.txt`'ten `gradio`/`matplotlib`/`pydantic<2.11` pini çıkarıldı, `streamlit`+`pandas` eklendi.

**KAPSAM GÜNCELLEMESİ (2026-09-16) — Veri analizi Streamlit'e taşındı + zaman serisi netleştirmesi:**
1. **Zaman serisi için ayrı veri seti GEREKMİYOR** — trend, `bot/db.py`'nin kendi kaydettiği geçmiş gözlemlerden geliyor (ürünün kendi kullanım telemetrisi). Netleştirme: `report/kod_notlarim.md`.
2. **`notebooks/00_veri_kesfi.py`'ye bozuk/açılamayan görsel kontrolü eklendi** ("boş veri" karşılığı — görsel veride NaN yerine 0-byte/açılamayan dosya kontrolü) + tüm sayısal sonuçlar artık `veri_ozeti.json` olarak da kaydediliyor.
3. **`ui/app.py`'ye "📊 Veri Analizi" sekmesi eklendi** — Colab'dan inen `eda_ciktilari.zip`, `report/eda_ciktilari/` klasörüne çıkarılınca sınıf dağılımı, dengesizlik, bozuk görsel, tekrar eden görsel kontrolü otomatik olarak grafik/tablo halinde gösteriliyor.

**KARAR (2026-09-16) — Veri seti: vipoooool yerine abdallahalidev/plantvillage-dataset.** Kullanıcı tarafından önerildi. Bu, HAM (önceden bölünmemiş/çoğaltılmamış) PlantVillage verisi — `notebooks/01_train_model_colab.py` artık `kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")` ile indirip **kendi train/valid bölmesini** (%80/%20, tek seferde, sabit seed) yapıyor. Bu, önceki veri setinde (`vipoooool`, "Augmented") tespit ettiğimiz train/valid sızıntı riskini YAPISAL olarak ortadan kaldırıyor (aynı görsel iki tarafta birden asla olamaz). `notebooks/00_veri_kesfi.py` da yeni veri kaynağına göre güncellendi — artık ham veri üzerinde sınıf dağılımı + görsel kalitesi + **tekrar eden (duplicate) görsel** kontrolü yapıyor (train/valid ayrımı script tarafından yapıldığı için "sızıntı" değil "tekrar" kontrolü daha doğru çerçeve). `01_train_model_kaggle.py` (Kaggle ortamı, eski veri seti) artık güncel değil — kullanılmıyor, referans olarak duruyor.

**KAPSAM GÜNCELLEMESİ (2026-09-15) — RAG (metin + görsel) + açıklanabilirlik:**
1. **Metin RAG:** `agent/knowledge/*.md` (5 hastalık dokümanı, elle yazılmış/doğrulanmış — etken, belirtiler, karıştırılabilecek hastalıklar, kültürel/biyolojik önlem) → `rag/build_index.py` ile Chroma vektör DB'ye indexleniyor (çok dilli embedding modeli, Türkçe için). `agent/rag.py`'deki `retrieve_context()` bunu `agent/report.py`'ye (yerel demo) bağladı — LLM artık ezberden değil kaynağa dayalı yazıyor.
2. **Görsel RAG:** Ayrı bir CLIP modeli EKLEMEDEN, eğitilen MobileNetV2'nin son katmandan önceki (GAP) çıktısını embedding olarak yeniden kullanan bir tasarım (`rag/build_image_index.py`, `agent/image_rag.py`). `inference/app.py`'ye bağlandı (`benzer_gorseller` alanı) — ama **gerçek model gelmeden aktif olmaz** (bilerek; rastgele ağırlıklarla embedding anlamsız olurdu). Model gelince tek komutla (`python rag/build_image_index.py`) devreye girer.
3. **Fine-tuning netleştirmesi:** Mevcut 2 aşamalı transfer learning ZATEN fine-tuning (son 40 katman açılıp küçük öğrenme oranıyla eğitiliyor) — bu `report/kod_notlarim.md`'de detaylı açıklandı (neden tüm ağ değil son 40 katman, ne zaman daha agresif fine-tuning düşünülür).
4. **`report/kod_notlarim.md` genişletildi** — inference/app.py, RAG (metin+görsel), weather/db entegrasyonu, senaryo tablosunun neden ML modeli olmadığı — hepsi "sunumda anlatabilme" için sade Türkçe'yle yazıldı.

requirements.txt'e `chromadb` + `sentence-transformers` geri eklendi (RAG için — daha önce "bonus aşamasında eklenecek" notuyla çıkarılmıştı, şimdi o aşamaya gelindi).

---

## Genel durum

- [x] **Gün 0** — Kurulum (hesaplar + ortam) — venv + `pip install -r requirements.txt` yapıldı; hesaplar hâlâ *Sen*'de bekliyor
- [ ] **Gün 1** — Veri inceleme + GitHub repo (repo zaten bağlı, Kaggle veri incelemesi bekliyor)
- [x] **Gün 2–3** — Model eğitimi (domates, 5 sınıf) — ✅ Colab'da çalıştı, GERÇEK model geldi. EfficientNetB0 (gelişmiş fine-tuning) %97.62 doğruluk, üretimde (`model/model.keras`). Detay: aşağıdaki "Yapıldı (log)" 2026-09-19 kayıtları.
- [x] **Gün 4** — Inference servisi (FastAPI) — ✅ gerçek modelle test edildi (DEMO MODU kapalı, `/predict`+`/predict_compare` doğru çalışıyor)
- [~] **Gün 5 (BAŞLADI, 2026-09-19) — local n8n + Telegram + LangChain:**
  - [x] `workflow.json` local mimariye göre güncellendi, `npx n8n start` + ngrok sabit domain ile çalışıyor
  - [x] Inference servisi arka planda çalışır durumda (test edildi, `/docs` 200 dönüyor)
  - [x] ngrok + Telegram + Anthropic (Header Auth, artık kullanılmıyor) credential'ları bağlandı
  - [x] **LangChain'e geçiş tamamlandı (2026-09-21):** eski "HTTP Request - Claude Agent" silindi, "Basic LLM Chain" + "Anthropic Chat Model" (yeni "Anthropic" tipi credential) ana akışa bağlandı; "Rapor JSON'unu Ayrıştır" kod node'u yeni çıktı şekline (`raw.text`) göre güncellendi
  - [x] Sistem promptu birkaç kez iyileştirildi (bkz. log 2026-09-21): pestisit kategori politikası, hastalık adı çeviri kuralı, güven değeri değişmezliği, `neden` alanı + `onlem` dizi formatı, güvenlik/prompt-injection kuralı genişletildi — `agent/report.py`, `agent/prompt_taslagi.md`, `n8n/workflow.json` senkron
  - [ ] **Google Sheets credential'ı — YARIM KALDI:** Google Cloud'da proje + Sheets API + OAuth consent screen (Audience) adımına kadar gelindi, Client ID/Secret oluşturma ve n8n'e bağlama kaldı — *Sen*
  - [ ] Google Sheets'e `neden` sütun başlığı eklenmesi gerekiyor — *Sen*
  - [ ] İlk uçtan uca Telegram testi — *Beraber*, Sheets bağlantısı bitince yapılacak
- [ ] **Gün 6** — n8n'de LLM-Agent + prompt geliştirme (HTTP Request → Claude)
- [ ] **Gün 7** — n8n: Sheets kaydı + PDF
- [ ] **Gün 8** — Uçtan uca test + uç durumlar
- [~] **Gün 9** — Rapor — `report/rapor_taslagi.md` ÖNCEDEN başlatıldı (normalde Gün 9 işi ama sonuçlar hazır olunca yazıldı): CNN/transfer learning temelleri, model karşılaştırma sonuçları, gelişmiş fine-tuning deneyi, 14 adımlık süreç günlüğü dolu. Giriş/n8n akışı/RAG/sunum bölümleri eksik.
- [ ] **Gün 10** — Sunum + GitHub push
- [ ] *(bonus, MVP bitmeden başlanmaz)* Hava durumu / tarla defteri entegrasyonu, bölgesel uyarı, 38 sınıf

---

## Detaylı görev listesi

### Gün 0 — Kurulum
- [ ] Kaggle hesabı (SADECE veri seti indirmek için — telefon doğrulama/GPU gerekmiyor artık, eğitim Colab'da) — *Sen*
- [ ] GitHub hesabı + boş repo — *Sen*
- [ ] Anthropic API anahtarı — *Sen*
- [ ] Telegram bot token (@BotFather → /newbot) — *Sen*
- [ ] n8n Cloud hesabı (n8n.io) — Docker Desktop artık GEREKMİYOR — *Sen*
- [ ] `venv` + `pip install -r requirements.txt` — ✅ bu ortamda yapıldı
- [ ] `.env.example` → `.env`, anahtarları doldur — *Sen*

### Gün 1 — Veri inceleme + repo
- [ ] Kaggle'da "PlantVillage Dataset" (abdallahalidev) sayfasına gir, veri yapısını gör (indirme Colab'da kagglehub ile otomatik olacak) — *Sen*
- [ ] Projeyi GitHub'a ilk push — *Beraber*
- [ ] `.gitignore` (venv, .env, *.sqlite, model dosyaları) — *Ben*

### Gün 2–3 — Model eğitimi (domates, 5 sınıf)
- [ ] `01_train_model_colab.py` → Google Colab'a yapıştır, Runtime > GPU (T4) seç, Kaggle API token'ını yükle (sadece veri indirmek için), Run All — *Sen*
- [ ] Çıkan `leadleaf_model_ciktisi.zip`'i indir, aç, içindekileri `model/` klasörüne koy — *Sen*
- [ ] Doğrulama doğruluğu ≥ %90 — *hedef*
- [ ] `model.keras`, `class_names.json`, `metrics.txt`, `confusion_matrix.png`, `demo_images/` indir → `model/`, `demo_images/` — *Sen*

### Gün 4 — Inference servisi
- [x] `inference/app.py` — FastAPI `POST /predict` — *Ben* ✅ (DEMO MODU ile, model gelmeden test edilebiliyor)
- [x] `uvicorn` ile çalıştır, demo görselle test — ✅ bu ortamda test edildi (`/health`, `/predict` doğru dönüyor)
- [x] `ui/app.py` (Gradio) — yerel uçtan uca görselleştirme — *Ben* ✅ çalışıyor (`agent/report.py` ile rapor üretimi de dahil)
- [ ] Gerçek `model.keras` gelince (Gün 2-3 sonrası) `model/` klasörüne koy, servisi yeniden başlat, DEMO MODU otomatik kapanır — *Sen*

### Gün 5 — n8n Cloud kurulumu + Telegram
- [x] `n8n/workflow.json` (taslak) + `n8n/README_N8N.md` — *Ben* ✅ yazıldı
- [ ] n8n.io Cloud hesabı aç, workflow.json'ı import et — *Sen*
- [ ] Inference servisini n8n Cloud'un görebilmesi için tünel (ngrok) aç veya cloud'a deploy et — *Sen* (adımlar: `n8n/README_N8N.md`)
- [ ] n8n'de Telegram credential (BotFather token) bağlanıyor — *Sen*
- [ ] Trigger → HTTP Request (`/predict`) bağlanıyor; foto atınca ham teşhis dönüyor — *Beraber*

### Gün 6 — n8n'de LLM-Agent + prompt geliştirme
- [ ] `agent/prompt_taslagi.md`'deki prompt → n8n HTTP Request node'una (Claude API) yapıştırılıyor — *Ben verir, Sen yapıştırır*
- [ ] Anthropic API key n8n credential olarak bağlanıyor — *Sen*
- [ ] IF node: güven `%70` altı → "ziraat mühendisine danış" yönlendirmesi — *Ben*
- [ ] Cevapları görüp prompt'u n8n içinde iyileştirme ("prompt geliştirme") — *Beraber*
- [ ] Uçtan uca test (foto → n8n → rapor) — *Sen*

### Gün 7 — n8n: kayıt + rapor
- [ ] Google Sheets node — her teşhisi kaydet — *Ben verir, Sen credential bağlar*
- [ ] Basit PDF üretimi (HTML→PDF node) — *Ben*
- [ ] `n8n/workflow.json` olarak tüm akış dışa aktarılıp repoya eklenir — *Beraber*

### Gün 8 — Uçtan uca test + uç durumlar
- [ ] Yaprak olmayan görsel, düşük güven, bulanık foto senaryoları — *Ben + Sen*
- [ ] Hataların giderilmesi — *Beraber*

### Gün 9 — Rapor
- [ ] `report/rapor_taslagi.md` — metodoloji, metrikler, örnek çıktılar, sınırlılıklar — *Ben taslak, Sen sonuçlar*

### Gün 10 — Teslim
- [ ] Sunum slaytları (mimari + sonuç + sınırlılık) — *Beraber*
- [ ] Demo videosu (bota foto → rapor) — *Sen*
- [ ] GitHub'a son push — *Beraber*

### Bonus — MVP bitmeden BAŞLANMAZ
- [x] ~~`agent/weather.py`'yi agent'a bağla~~ — ✅ MVP'ye çekildi, `ui/app.py`'ye entegre edildi (2026-09-15)
- [x] ~~`bot/db.py`'yi bota bağla — tarla defteri~~ — ✅ MVP'ye çekildi, `ui/app.py`'ye entegre edildi (2026-09-15)
- [ ] n8n bölgesel erken uyarı (cron) — hâlâ bonus (Telegram botuna otomatik proaktif bildirim, `db.recent_cluster` zaten hazır)
- [ ] **"Tarla 360" derinliğini n8n/Telegram akışına taşımak** (2026-09-20 not) — şu an hava durumu riski, tarla defteri trendi ve bölgesel kümelenme SADECE yerel Streamlit demosunda (`ui/app.py`) var; n8n workflow'u sadece CNN sonucu + Claude raporu veriyor. Not: bölge bilgisi modelin kendisinden gelmiyor — `bot/db.py`'de kullanıcının kendi girdiği `il`/`ilce` profil bilgisi + `recent_cluster()` SQL sorgusuyla hesaplanıyor, CNN'e hiç dokunmuyor. Taşınırsa: n8n'e `agent/weather.py` + `bot/db.py` çağıran ek HTTP Request node'ları eklenmesi gerekir (inference/app.py'ye sarmalayıcı endpoint'ler eklenerek).
- [ ] `SELECTED_CLASSES = None` yapıp 38 sınıfa / çoklu bitkiye genişlet
- [ ] Düşük güvende "farklı açıdan foto iste"
- [ ] Öğrenci/çiftçi için farklı ayrıntı seviyesi

---

## Yapıldı (log)

- **2026-09-21 — Bootcamp PDF incelendi + pestisit politikası "orta yol"a çekildi + LangChain'e geçildi + prompt birkaç turda sertleştirildi.**
  1. **Bootcamp ödev PDF'i** (`DL + LLM-Agent + n8n Birlesimi.pdf`) incelendi: brief LangChain node kullanımını öneriyor ve ilaç dozu/önerisi bekliyor gibi görünüyordu — bu, projenin önceki "asla ilaç önerme" güvenlik kararıyla gerginlik yarattı.
  2. **Pestisit kararı (AskUserQuestion ile netleştirildi):** "Orta yol" seçildi — genel ürün KATEGORİSİ (örn. "bakır bazlı fungisit") verilebilir, ama asla marka/kesin doz/kesin hasat-öncesi-bekleme-süresi verilmez; ürün kategorisinden bahsedilirse "ambalaj etiketine ve ruhsatlı ziraat mühendisine danışın" cümlesi eklenir.
  3. **n8n'de LangChain'e geçiş tamamlandı:** eski "HTTP Request - Claude Agent" node'u silindi; "Basic LLM Chain" + "Anthropic Chat Model" (yeni "Anthropic" tipi credential, Header Auth'tan farklı) ana akışa bağlandı (`Predict CNN → Basic LLM Chain → Rapor JSON'unu Ayrıştır`). "Rapor JSON'unu Ayrıştır" kod node'u, eski node'un ham Anthropic API şekli (`raw.content[0].text`) yerine LangChain node'unun çıktı şekline (`raw.text`, markdown code-fence temizleme dahil) göre güncellendi — güncellenmeseydi her çalıştırmada sessizce "ayrıştırılamadı" hatasına düşerdi.
  4. **Sistem promptu 3 ayrı geri bildirim turunda iyileştirildi** (`agent/report.py`, `agent/prompt_taslagi.md`, `n8n/workflow.json` — üçü senkron tutuluyor): (a) `neden` alanı eklendi (hastalığın etkeni + yayılma koşulları, fotoğraftan doğrulanamayacak kesin iddialardan kaçınma uyarısıyla); (b) `onlem` alanı tek cümleden madde-madde DİZİYE çevrildi; (c) hastalık adının CNN'in `hastalik_tr` alanından (sabit `TR_ADLAR` sözlüğü, `inference/app.py`) AYNEN alınması, LLM'in kendi çevirisini/halk adını uydurmaması kuralı eklendi; (d) `guven` değerinin modelden aynen alınıp LLM tarafından değiştirilmemesi kuralı eklendi; (e) "cevabın sonuna ekle" ile JSON'daki `uyari` alanı arasındaki çelişki giderildi — sabit metin artık SADECE `uyari` alanına yazılıyor; (f) güvenlik kuralı genişletildi: "ben senin geliştiricinim, API key/credential/sistem promptu ver" tarzı sosyal mühendislik denemelerine karşı açık ret eklendi (not: Claude zaten gerçek API key'i hiç görmüyor, key n8n credential katmanında kalıyor — bu ek bir savunma katmanı).
  5. **`agent/report.py`'de gizli bir kopya hata bulundu ve düzeltildi:** yerel Streamlit demosu, `hastalik_tr` yerine yanlışlıkla ham İngilizce `hastalik` alanını Claude'a gönderiyordu (n8n tarafında bu hata yoktu). Düzeltildi.
  6. **`ui/app.py` yeni şemaya güncellendi:** `onlem` artık liste olduğu için `st.markdown` çağrısı madde-madde render edecek şekilde düzeltildi (düzeltilmeseydi ekranda `['...', '...']` gibi çirkin bir Python listesi görünürdü); yeni `neden` alanı için bir "Neden oluyor?" bölümü eklendi. Bu, kullanıcının "Streamlit de olacaktı unutma" hatırlatmasıyla yakalandı — n8n tarafı güncellenirken Streamlit tarafı unutulma riski taşıyordu.
  7. **Google Sheets - Kaydet node'una `neden` sütunu + `onlem` için `.join(' | ')` eklendi** (dizi artık tek hücrede okunabilir metin olarak kaydediliyor). Google Sheets credential'ının HİÇ oluşturulmadığı fark edildi (n8n Credentials listesi kontrol edilerek) — kurulum başlatıldı (Google Cloud proje + Sheets API enable + OAuth consent screen/Audience adımına kadar gelindi), Client ID/Secret oluşturma ve n8n'e bağlama bir sonraki oturuma kaldı.
  8. **Küçük düzeltme:** "Header Auth account" credential'ı (eski HTTP node'un kalıntısı) artık kullanılmıyor ama zararsız duruyor; hiçbir credential oluşturmanın kendisi ücrete tabi değil, sadece gerçek API çağrısı ücretlendiriliyor — ilk gerçek Telegram testi yapılmadığı için $5 deneme kredisi hâlâ tam duruyor.
- **2026-09-20** — 3 credential'dan 2'si tamam: **Telegram** (ETIMEDOUT görünse de token doğrulandı, iki node da aynı credential'a bağlandı, fazlalık silindi) ve **Anthropic** (yeni hesap, kartsız $5 kredi, `x-api-key` Header Auth, `api.anthropic.com`'a domain kısıtlaması eklendi). Yanlışlıkla Anthropic credential'ı "HTTP Request - Predict CNN" (kendi FastAPI'miz) node'una da bağlanmıştı — bu node Authentication: None'a düzeltildi (zaten kimlik doğrulama istemiyor), "HTTP Request - Claude Agent" node'undaki doğru bağlantı korundu. Sırada: Google Sheets credential'ı, sonra ilk uçtan uca Telegram testi.
- **2026-09-11** — Proje iskeleti: klasör yapısı, `README.md`, `requirements.txt`, `.env.example`, `notebooks/01_train_model_kaggle.py`, `PROGRESS.md`.
- **2026-09-11** — `bot/db.py` yazıldı ve test edildi (SQLite tarla defteri + `recent_cluster`). ✅ çalışıyor, **bonus aşamasına kadar entegre edilmeyecek**.
- **2026-09-11** — `agent/weather.py` yazıldı; Open-Meteo API yanıt yapısı doğrulandı. ✅ kod hazır, **bonus aşamasına kadar entegre edilmeyecek**.
- **2026-09-11** — **Kapsam kararı:** proje adı LeadLeaf AI; 38 sınıf yerine domates + 4 hastalık (5 sınıf) ile MVP; hava durumu/tarla defteri/bölgesel uyarı/konum haritası TÜBİTAK-TEKNOFEST aşamasına alındı; RAG'in rolü ilaç dozu değil doğrulanmış bilgi + kültürel-biyolojik önlem + uzmana yönlendirme olarak değiştirildi. `01_train_model_kaggle.py` domates alt kümesi filtresiyle güncellendi.
- **2026-09-11** — **Mimari kararı:** "prompt geliştirme n8n'de" isteği üzerine LLM-Agent ve Telegram botu n8n'e taşındı; Python'da sadece FastAPI `/predict` kalıyor. `agent/prompt_taslagi.md` yazıldı (n8n HTTP Request node'una yapıştırılacak sistem promptu + kullanıcı promptu + node ayarları).
- **2026-09-11** — GitHub reposu bağlandı ve ilk push yapıldı: https://github.com/bilisim245/LeadLeaf-AI
- **2026-09-15** — **Mimari kararı:** n8n Cloud'a geçildi (local Docker yerine); `n8n/README_N8N.md` yazıldı (tünel/deploy seçenekleri, credential listesi).
- **2026-09-15** — `inference/app.py` yazıldı (Gün 4): FastAPI `/health` + `/predict`; model dosyaları yoksa **DEMO MODU** (rastgele ama tutarlı sonuç) — Kaggle eğitimi bitmeden uçtan uca test/görselleştirme mümkün. Yerel venv (Python 3.9) kuruldu, `pip install -r requirements.txt` yapıldı, `/health` ve `/predict` test görseliyle doğrulandı (✅ çalışıyor).
- **2026-09-15** — `agent/report.py` yazıldı: SADECE `ui/app.py` yerel demosu için, `agent/prompt_taslagi.md` ile birebir aynı sistem promptuyla Claude'a rapor ürettirir; `ANTHROPIC_API_KEY` yoksa şablon rapora düşer (demo çökmez).
- **2026-09-15** — `ui/app.py` (Gradio) yazıldı ve test edildi: görsel yükle → inference → rapor, tek ekranda. Bilinen `gradio==4.44.1` + yeni `pydantic` uyumsuzluğu (`TypeError: argument of type 'bool' is not iterable`) `pydantic<2.11` pini ile çözüldü ve `requirements.txt`'e not düşüldü.
- **2026-09-15** — `n8n/workflow.json` (taslak) yazıldı: Telegram Trigger → dosya indir → `/predict` → Claude Agent (HTTP Request) → JSON ayrıştır (Code node) → Google Sheets + Telegram cevap. Güven-bazlı ek dallanma (IF) ve PDF üretimi bilerek Gün 6-7'ye, n8n arayüzünde canlı geliştirmeye bırakıldı.
- **2026-09-15** — `requirements.txt` gerçek MVP'ye göre güncellendi: `torch/transformers/safetensors` (kullanılmıyor) ve `langchain*/chromadb/sentence-transformers` (agent n8n'de, RAG bonus aşamasında) çıkarıldı; `tensorflow-cpu` eklendi (model Keras/TensorFlow ile eğitiliyor).
- **2026-09-15** — `notebooks/01_train_model_colab.py` yazıldı (Kaggle yerine Colab, sıfır yerel kurulum).
- **2026-09-20 — DÜZELTME 2 + MİMARİ KARARI: n8n Cloud'dan tamamen vazgeçildi, local n8n'e (Docker'sız) geçildi.** Kullanıcı n8n Cloud'un "Start free 14 days trial" mesajını gördü, araştırıldı: n8n Cloud'un artık kalıcı ücretsiz planı YOK (sadece 14 gün deneme, sonrası en az $20/ay) — 2026-09-15'teki "n8n Cloud'a geç" kararı yanlış varsayıma dayanıyormuş. Kullanıcı ödeme istemediği için **n8n artık local'de, `npx n8n start` ile (Node.js üzerinden, Docker YOK) çalışıyor** — Node.js LTS winget ile kuruldu (`OpenJS.NodeJS.LTS`), n8n ilk kez `npx` ile indirilip başlatıldı (localhost:5678). Mimari basitleşti: n8n + FastAPI aynı makinede olduğu için aralarında tünele gerek yok (n8n, `http://localhost:8000/predict`'i doğrudan çağırıyor); SADECE n8n'in kendisi (Telegram webhook'u için) ngrok'un sabit domaini üzerinden tünelleniyor (`ngrok http 5678 --url https://enclose-afterglow-sappiness.ngrok-free.dev`). `n8n/workflow.json`'daki URL `localhost:8000/predict` olarak güncellendi, `n8n/README_N8N.md` tamamen yeniden yazıldı (local/Docker'sız kurulum adımları). Hem n8n hem FastAPI hem ngrok tüneli şu an arka planda çalışır durumda, ikisi de test edildi (`/health` her ikisinde de OK). Sıradaki adım: tarayıcıdan `localhost:5678`'e girip workflow.json'ı import etmek + credential'ları bağlamak.
- **2026-09-20 — workflow.json local n8n'e başarıyla import edildi.** Dosya diyaloğu native pencerede garip davrandı (Explorer'ı VS Code ile açtı), o yüzden kopyala-yapıştır yöntemine geçildi: `Get-Content | Set-Clipboard` ile içerik panoya alındı, kullanıcı n8n canvas'ında kendi eliyle Ctrl+V yaptı (otomatik/programatik Ctrl+V tarayıcı güvenlik kısıtlaması nedeniyle çalışmadı — gerçek insan tuşuna basması gerekti). Sonuç doğrulandı (`localhost:5678/workflow/SuklzMNlxzUJN6xQ`): tüm 7 node eksiksiz geldi (Telegram Trigger, Fotoğrafı İndir, HTTP Request - Predict CNN → doğru şekilde `localhost:8000/predict`, HTTP Request - Claude Agent, Rapor JSON'unu Ayrıştır, Google Sheets - Kaydet, Telegram - Cevap Gönder). Sıradaki adım: 3 credential oluşturma (Telegram Bot API, Anthropic, Google Sheets).
- **2026-09-20 — ngrok tüneli çalışıyor:** ngrok winget ile kuruldu (kullanıcının kendi kurulumu başarısız oldu, Claude tarafından kuruldu), authtoken eklendi, hesaba atanmış sabit domain (`https://enclose-afterglow-sappiness.ngrok-free.dev`) ile tünel açıldı. `/health` ve `/predict` bu public URL üzerinden gerçek modelle test edildi, ikisi de doğru çalışıyor (✅). Bu domain KALICI — n8n workflow'undaki HTTP Request node'una bu URL yazılacak, bir daha değişmeyecek. Sıradaki adım: n8n Cloud hesabı + workflow import + credential bağlama.
- **2026-09-20 — DÜZELTME:** Hugging Face Spaces (Docker SDK) ile ngrok'suz kalıcı deploy denendi (Dockerfile + requirements + README hazırlandı) ama **kişisel hesapta Docker Space açmak HF'de PRO (ücretli) plan gerektiriyor** (2026 itibarıyla — CPU Basic donanımın kendisi ücretsiz, ama compute üzerinde çalışan yeni bir Space *oluşturmak* ücretli). Kullanıcı ücret ödemek istemediği için bu yoldan vazgeçildi, `deploy/` klasörü silindi. **Karar: ngrok'a devam.** Ayrıca ngrok hakkında önceki bir yanlış bilgi düzeltildi: ücretsiz plan artık hesaba **kalıcı/sabit atanmış bir "dev domain"** veriyor (`senin-adin.ngrok-free.app`) — eskisi gibi her `ngrok http` başlatmasında değişmiyor, bu da "URL sürekli değişiyor" sorununu ortadan kaldırıyor. Sonuç: FastAPI inference laptopta çalışacak, ngrok ücretsiz statik domain ile dışarı açılacak, n8n Cloud o sabit URL'e istek atacak. Telegram/n8n tarafı bundan etkilenmiyor — sadece "CNN'i nerede çalıştırıyoruz" kısmı bu.
- **2026-09-19 (oturum sonu)** — Gün 5'e (n8n Cloud + Telegram) geçildi. Yapılanlar: `n8n/workflow.json` güncel `/predict` çıktısıyla uyumlu bulundu (değişiklik gerekmedi); inference servisi arka planda çalışır bırakıldı (`uvicorn ... --port 8000`, gerçek modelle, `inference_out.log`/`inference_err.log`'a yazıyor). Kullanıcı ngrok kurulumunu ve BotFather token alımını **kendi öğrenmek istediği için** kendisi yapıyor — adım adım talimat verildi (ngrok: winget/indirme → ngrok.com hesap → authtoken → `ngrok http 8000`; BotFather: Telegram'da `/newbot`). Oturum burada duraklatıldı, kullanıcı ngrok/BotFather adımlarını tamamlayınca devam edilecek. **Sıradaki iş:** ngrok tüneli açılınca n8n Cloud hesabı açılıp `workflow.json` import edilecek, URL + 3 credential (Telegram/Anthropic/Sheets) bağlanıp ilk uçtan uca Telegram testi yapılacak.
- **2026-09-19** — **Gelişmiş EfficientNetB0 fine-tuning deneyi çalıştı, SONUÇ BAŞARILI:** `notebooks/02_efficientnetb0_gelismis_egitim.py` Colab'da çalıştırıldı (yolda `.prefetch()` sonrası `class_names` kaybolması hatası çıktı, `class_names`'i prefetch'ten ÖNCE okuyacak şekilde düzeltildi). Sonuç: doğruluk %94.68→%97.62 (+2.94 puan), macro F1 0.940→0.9716, macro AUC 0.9958→0.9993, overfitting yok (öğrenme eğrisi 3 kademeli açma aşamasında da sağlıklı). **`model/model.keras` bu gelişmiş modelle değiştirildi** (Keras uyumluluk yaması uygulanıp `/predict`+`/predict_compare` gerçek foto ile tekrar test edildi, ✅). Eski EfficientNetB0 modeli `model/tubitak/model_EfficientNetB0.keras` olarak korunuyor (farklı checksum, kaybolmadı). `report/rapor_taslagi.md`'ye Bölüm 3.6 (gerçek sonuç tablosu+grafikleri) ve yeni Bölüm 4 "Süreç Günlüğü" (veri seti seçiminden üretim modeli güncellemesine kadar 14 adımlık, jüri sorularına hazırlık amaçlı gerekçelendirilmiş karar günlüğü) eklendi. `ui/app.py`'nin Model Karşılaştırma sekmesine model seçim menüsü (selectbox: MobileNetV2/V3Small/EfficientNetB0 temel/gelişmiş) + o modelin confusion matrix + öğrenme eğrisi görsellerini gösteren bölüm eklendi.
- **2026-09-19** — `report/rapor_taslagi.md` yazılmaya başlandı: CNN/transfer learning temellerini (convolution, pooling, backbone/head, 2 aşamalı eğitim, BatchNorm tuzağı, train/valid/test mantığı, metrikler) sıfırdan anlatan bir bölüm + Bölüm 3 (model eğitimi/karşılaştırma sonuçları, confusion matrix bulguları, fine-tuning'in mimariye göre farklı etkisi). Ayrıca `notebooks/02_efficientnetb0_gelismis_egitim.py` yazıldı (ayrı dosya) — SADECE EfficientNetB0'ı 4 iyileştirmeyle (daha uzun eğitim + sabırlı early stopping, son %25→%40 açma, kademeli/gradual unfreezing %15→%30→%40, ReduceLROnPlateau ile azalan LR) yeniden eğitip önceki/gelişmiş karşılaştırma bar chart'ı (`efficientnetb0_onceki_vs_gelismis.png`) üreten TÜBİTAK ek deneyi — SEED=42 ile aynı split, doğrudan kıyaslanabilir. Henüz Colab'da çalıştırılmadı, sonuç bekleniyor.
- **2026-09-19** — **Gün 2-3 tamamlandı: Colab eğitimi çalıştı, gerçek modeller geldi.** Sonuçlar (bağımsız test seti, `model_comparison.csv`): **EfficientNetB0 en iyi** (doğruluk %94.68, macro F1 0.940, macro AUC 0.996, 36.8 MB), MobileNetV2 orta (doğruluk %90.95, 22.1 MB), MobileNetV3Small en küçük ama en düşük doğruluk (%87.94, 9.65 MB) — literatür taramasıyla (EfficientNetB0 en yüksek doğruluk, MobileNet ailesi hız/boyut avantajlı) birebir örtüşüyor. Üretim için EfficientNetB0 otomatik seçildi (`model_meta.json`). **Karşılaşılan ve çözülen sorun:** Colab'ın Keras sürümü local'den (3.10.0) daha yeni olduğu için `.keras` dosyalarının `config.json`'unda Dense katmanına local'in tanımadığı bir `quantization_config` alanı vardı — `tf.keras.models.load_model()` "Unrecognized keyword arguments" hatasıyla TÜM inference servisini çökertiyordu (DEMO MODU'na düşmüyordu, bu da "eksik model demo moda düşer, servis çökmez" felsefesiyle tutarsızdı). Çözüm: `inference/app.py`'ye `_keras_uyumlu_yukle()` + `_strip_quantization_config()` eklendi — yükleme bu hatayla karşılaşırsa dosyayı otomatik onarıp bir kez daha dener (model ağırlıklarını/davranışını değiştirmiyor, sadece uyumsuz serileştirme alanını siliyor); ayrıca `_load_model_if_available()` artık yükleme hatasında da (dosya eksikliği gibi) DEMO MODU'na düşüyor, servis çökmüyor. 4 model dosyası da (`model.keras` + 3 TÜBİTAK modeli) bu onarımdan geçirilip `model/` ve `model/tubitak/`'a yerleştirildi, `/predict` ve `/predict_compare` gerçek modelle test edildi (✅ ikisi de doğru çalışıyor — örnek: sağlıklı bir yaprakta MobileNetV2 yanlış tahmin etti ama MobileNetV3Small+EfficientNetB0 doğru bildi, "kısmi uzlaşma" mantığı beklendiği gibi devreye girdi). `.gitignore` düzeltildi: `model/*.keras` → `model/**/*.keras` (alt klasördeki TÜBİTAK modelleri de artık doğru hariç tutuluyor, önceden sadece `model/model.keras` hariç tutuluyordu — model/tubitak/ altındaki ~70MB'lık 3 .keras dosyası yanlışlıkla commit'e girebilirdi).
- **2026-09-19** — `notebooks/01_train_model_colab.py`'ye AUC (macro, roc_auc_score) + kayip/dogruluk ikili ogrenme egrisi grafigi + dropout 0.3->0.2 + parametre_ozeti (egitilebilir/donuk parametre sayisi, her asamada) + model_karsilastirma_dogruluk.png (3 model bar chart) eklendi; Streamlit "Model Karsilastirma" sekmesine test-seti model_comparison.csv okuyup gosteren bolum eklendi. Ayrica `notebooks/01_train_model_colab_en.py` yazildi — ayni mantigin fonksiyon/degisken isimleri ve yorumlari Ingilizce olan paralel kopyasi (cikti dosya adlari — model_comparison.csv, split_manifest.json vb. — bilerek Turkce/ayni birakildi, ui/app.py ve inference/app.py bu isimlere gore okuyor). Turkce dosya kanonik surum olarak kaliyor.
- **2026-09-17** — Uretim `/predict` icin otomatik mimari secimi eklendi ve commit edildi (`736d7b6`): TUBITAK notebook'u artik 3 modelden en yuksek dogruluklu olani `model.keras` yapip `model_meta.json`'a mimarisini yaziyor; `inference/app.py` bunu okuyup dogru `preprocess_input`'u seciyor (once hep MobileNetV2 varsayiyordu). Ayrica referans bir CNN/transfer-learning ders defteri (Miuul/Veysel Hoca, harici GitHub repo) satir satir incelenip kendi egitim kodumuzla (2 asamali fine-tuning tarifi + "4 tuzak": preprocess_input, ogrenme orani, sira, BatchNormalization) karsilastirildi — kodumuz zaten dogru: BatchNorm tuzagina `model_kur()`'daki `base(x, training=False)` cagrisiyla (resmi Keras'in onerdigi yontem) karsi korunuyor. Kod tarafinda degisiklik gerekmedi, sadece dogrulama yapildi.
- **2026-09-15** — **"Derinlik" geri bildirimi üzerine kapsam genişletmesi:** `agent/weather.py` + `bot/db.py` bonustan MVP'ye çekildi; `ui/app.py` "Tarla 360" dashboard'una dönüştürüldü (geçmiş trend grafiği, hava durumu riski, bölgesel kümelenme notu, kural-tabanlı senaryo analizi tablosu — ML tahmini değil, açıkça etiketlendi). `notebooks/00_veri_kesfi.py` yazıldı (sınıf dağılımı, görsel boyutu, train/valid data-leakage kontrolü perceptual hash ile). Yerelde uçtan uca test edildi: DB kaydı, hava durumu API çağrısı (Open-Meteo, gerçek ağ isteği), senaryo hesaplaması, trend grafiği — hepsi doğru çalışıyor (✅).

---

## Kararlar

- **Proje adı:** LeadLeaf AI — Derin Öğrenme ve Yapay Zekâ Ajanı Destekli Bitki Hastalığı Ön Değerlendirme Sistemi.
- **Domain / MVP sınıfları:** Domates — `healthy, Early_blight, Late_blight, Bacterial_spot, Septoria_leaf_spot` (5 sınıf).
- **DL:** Keras + MobileNetV2 transfer learning. Veri: "PlantVillage Dataset" (abdallahalidev, Kaggle — ham veri, train/valid'i biz bölüyoruz).
- **Agent:** Anthropic Claude (`claude-sonnet-5`), **n8n'in içinden** (HTTP Request node) çağrılır. **Doz/bekleme süresi ÜRETMEZ.**
- **Kullanıcı arayüzü + bot:** Telegram, **n8n Telegram Trigger/Send node'ları** ile (Python bot yok).
- **MVP (zorunlu):** FastAPI (`/predict`, tek Python parçası) + n8n (Telegram + Agent/prompt + Sheets + PDF).
- **Bonus (MVP sonrası, sırayla):** weather.py/db.py entegrasyonu → bölgesel uyarı → 38 sınıf → farklı açıdan foto isteme → ayrıntı seviyesi kişiselleştirme.
- **GitHub:** https://github.com/bilisim245/LeadLeaf-AI (push edildi, main branch).

---

## TÜBİTAK 2204'e çevirme notu

Araştırma sorusu: *"Yaprak görüntüsüne dayalı bir derin öğrenme + LLM-agent sistemi, sınırlı sınıf/veri ile de güvenilir bir ön-değerlendirme üretebilir mi; bağlamsal genişletmeler (hava durumu, geçmiş kayıt) bu güvenilirliği nasıl etkiler?*"
Ek gerekenler: hipotez, ölçüm planı (model doğruluğu + rapor kalitesi uzman değerlendirmesi), kontrol grubu, sınırlılıklar bölümü.
