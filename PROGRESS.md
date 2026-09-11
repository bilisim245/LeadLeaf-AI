# İLERLEME — LeadLeaf AI (Bitki Hastalığı Ön Değerlendirme Sistemi)

**Son güncelleme:** 2026-09-11
**Aktif aşama:** Gün 0 — Kurulum
**Kimler:** Sen = hesap/çalıştırma/test · Ben = tüm kod + rapor taslağı · Beraber = entegrasyon

**ÖNEMLİ KAPSAM KARARI (2026-09-11):** 10 günlük teslim MVP'dir. Hava durumu, tarla defteri, bölgesel uyarı, konum haritası, 38 sınıf → TÜBİTAK/TEKNOFEST "Sonraki Aşama"sına bırakıldı (ayrıntı: README.md). İlaç dozu/bekleme süresi LLM'e yazdırılmaz — güvenlik riski.

**MİMARİ KARARI (2026-09-11):** Bootcamp "prompt geliştirme n8n'de" istiyor. LLM-Agent çağrısı ve Telegram botu Python'dan **n8n'in içine** taşındı (n8n Telegram Trigger/Send + HTTP Request → Claude API, prompt n8n node'unda). Python'da sadece FastAPI `/predict` (CNN) kalıyor. `agent/agent.py` ve `bot/telegram_bot.py` planları iptal edildi (henüz yazılmamışlardı); yerine `agent/prompt_taslagi.md` (n8n'e yapıştırılacak prompt) geldi.

---

## Genel durum

- [ ] **Gün 0** — Kurulum (hesaplar + ortam)
- [ ] **Gün 1** — Veri inceleme + GitHub repo
- [ ] **Gün 2–3** — Model eğitimi (domates, 5 sınıf)
- [ ] **Gün 4** — Inference servisi (FastAPI) — tek Python parçası
- [ ] **Gün 5** — n8n kurulumu + Telegram Trigger/Send bağlantısı
- [ ] **Gün 6** — n8n'de LLM-Agent + prompt geliştirme (HTTP Request → Claude)
- [ ] **Gün 7** — n8n: Sheets kaydı + PDF
- [ ] **Gün 8** — Uçtan uca test + uç durumlar
- [ ] **Gün 9** — Rapor
- [ ] **Gün 10** — Sunum + GitHub push
- [ ] *(bonus, MVP bitmeden başlanmaz)* Hava durumu / tarla defteri entegrasyonu, bölgesel uyarı, 38 sınıf

---

## Detaylı görev listesi

### Gün 0 — Kurulum
- [ ] Kaggle hesabı + telefon doğrulama — *Sen*
- [ ] GitHub hesabı + boş repo — *Sen*
- [ ] Anthropic API anahtarı — *Sen*
- [ ] Telegram bot token (@BotFather → /newbot) — *Sen*
- [ ] Docker Desktop kurulu — *Sen*
- [ ] `venv` + `pip install -r requirements.txt` — *Sen*
- [ ] `.env.example` → `.env`, anahtarları doldur — *Sen*

### Gün 1 — Veri inceleme + repo
- [ ] Kaggle'da "New Plant Diseases Dataset" (vipoooool) ekle, domates 5 sınıfını gör — *Sen*
- [ ] Projeyi GitHub'a ilk push — *Beraber*
- [ ] `.gitignore` (venv, .env, *.sqlite, model dosyaları) — *Ben*

### Gün 2–3 — Model eğitimi (domates, 5 sınıf)
- [ ] `01_train_model_kaggle.py` → Kaggle'a yapıştır, GPU aç, `SELECTED_CLASSES` (domates 5 sınıf) ile çalıştır — *Sen*
- [ ] Doğrulama doğruluğu ≥ %90 — *hedef*
- [ ] `model.keras`, `class_names.json`, `metrics.txt`, `confusion_matrix.png`, `demo_images/` indir → `model/`, `demo_images/` — *Sen*

### Gün 4 — Inference servisi
- [ ] `inference/app.py` — FastAPI `POST /predict` — *Ben*
- [ ] `uvicorn` ile çalıştır, demo görselle test — *Sen*

### Gün 5 — n8n kurulumu + Telegram
- [ ] `n8n/docker-compose.yml` — *Ben*
- [ ] `docker compose up`, n8n arayüzü açılıyor (localhost:5678) — *Sen*
- [ ] n8n'de Telegram credential (BotFather token) bağlanıyor, Telegram Trigger + Send node'ları kuruluyor — *Ben verir, Sen bağlar*
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
- [ ] `agent/weather.py`'yi agent'a bağla (hazır, test edildi)
- [ ] `bot/db.py`'yi bota bağla — tarla defteri (hazır, test edildi)
- [ ] n8n bölgesel erken uyarı (cron)
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

---

## Kararlar

- **Proje adı:** LeadLeaf AI — Derin Öğrenme ve Yapay Zekâ Ajanı Destekli Bitki Hastalığı Ön Değerlendirme Sistemi.
- **Domain / MVP sınıfları:** Domates — `healthy, Early_blight, Late_blight, Bacterial_spot, Septoria_leaf_spot` (5 sınıf).
- **DL:** Keras + MobileNetV2 transfer learning. Veri: "New Plant Diseases Dataset" (vipoooool).
- **Agent:** Anthropic Claude (`claude-sonnet-5`), **n8n'in içinden** (HTTP Request node) çağrılır. **Doz/bekleme süresi ÜRETMEZ.**
- **Kullanıcı arayüzü + bot:** Telegram, **n8n Telegram Trigger/Send node'ları** ile (Python bot yok).
- **MVP (zorunlu):** FastAPI (`/predict`, tek Python parçası) + n8n (Telegram + Agent/prompt + Sheets + PDF).
- **Bonus (MVP sonrası, sırayla):** weather.py/db.py entegrasyonu → bölgesel uyarı → 38 sınıf → farklı açıdan foto isteme → ayrıntı seviyesi kişiselleştirme.
- **GitHub:** https://github.com/bilisim245/LeadLeaf-AI (push edildi, main branch).

---

## TÜBİTAK 2204'e çevirme notu

Araştırma sorusu: *"Yaprak görüntüsüne dayalı bir derin öğrenme + LLM-agent sistemi, sınırlı sınıf/veri ile de güvenilir bir ön-değerlendirme üretebilir mi; bağlamsal genişletmeler (hava durumu, geçmiş kayıt) bu güvenilirliği nasıl etkiler?*"
Ek gerekenler: hipotez, ölçüm planı (model doğruluğu + rapor kalitesi uzman değerlendirmesi), kontrol grubu, sınırlılıklar bölümü.
