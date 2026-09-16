# İLERLEME — LeadLeaf AI (Bitki Hastalığı Ön Değerlendirme Sistemi)

**Son güncelleme:** 2026-09-16

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
- [ ] **Gün 2–3** — Model eğitimi (domates, 5 sınıf) — **Colab GPU gerektirir (tarayıcıda, kurulum yok), bu ortamda yapılamaz**
- [x] **Gün 4** — Inference servisi (FastAPI) — tek Python parçası — ✅ yazıldı + test edildi (DEMO MODU'nda `/health` ve `/predict` çalışıyor)
- [ ] **Gün 5** — n8n Cloud kurulumu + Telegram Trigger/Send bağlantısı (workflow.json taslağı hazır, `n8n/README_N8N.md`)
- [ ] **Gün 6** — n8n'de LLM-Agent + prompt geliştirme (HTTP Request → Claude)
- [ ] **Gün 7** — n8n: Sheets kaydı + PDF
- [ ] **Gün 8** — Uçtan uca test + uç durumlar
- [ ] **Gün 9** — Rapor
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
- [ ] `SELECTED_CLASSES = None` yapıp 38 sınıfa / çoklu bitkiye genişlet
- [ ] Düşük güvende "farklı açıdan foto iste"
- [ ] Öğrenci/çiftçi için farklı ayrıntı seviyesi

---

## Yapıldı (log)

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
