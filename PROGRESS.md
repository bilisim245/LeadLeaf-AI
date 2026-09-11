# İLERLEME — LeadLeaf AI (Bitki Hastalığı Ön Değerlendirme Sistemi)

**Son güncelleme:** 2026-09-11
**Aktif aşama:** Gün 0 — Kurulum
**Kimler:** Sen = hesap/çalıştırma/test · Ben = tüm kod + rapor taslağı · Beraber = entegrasyon

**ÖNEMLİ KAPSAM KARARI (2026-09-11):** 10 günlük teslim MVP'dir. Hava durumu, tarla defteri, bölgesel uyarı, konum haritası, 38 sınıf → TÜBİTAK/TEKNOFEST "Sonraki Aşama"sına bırakıldı (ayrıntı: README.md). İlaç dozu/bekleme süresi LLM'e yazdırılmaz — güvenlik riski.

---

## Genel durum

- [ ] **Gün 0** — Kurulum (hesaplar + ortam)
- [ ] **Gün 1** — Veri inceleme + GitHub repo
- [ ] **Gün 2–3** — Model eğitimi (domates, 5 sınıf)
- [ ] **Gün 4** — Inference servisi (FastAPI)
- [ ] **Gün 5** — Telegram botu iskeleti
- [ ] **Gün 6** — Agent (güven kontrolü + kültürel/biyolojik öneri)
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

### Gün 5 — Telegram botu iskeleti
- [ ] `bot/telegram_bot.py` — foto al → `/predict` → ham teşhis mesajı — *Ben*
- [ ] Token ile bot ayağa kalkıyor, telefondan foto atınca cevap geliyor — *Sen*

### Gün 6 — LLM-Agent
- [ ] `agent/agent.py` — teşhis → JSON rapor: hastalık, güven, açıklama, **kültürel/biyolojik öneri (doz değil)**, uyarı; `%70` altı → "ziraat mühendisine danış" — *Ben*
- [ ] Bot, agent raporunu okunur mesaja çevirip döner — *Ben*
- [ ] Uçtan uca test (foto → rapor) — *Sen*

### Gün 7 — n8n
- [ ] `n8n/docker-compose.yml` — *Ben*
- [ ] `n8n/workflow_rapor.json` — Webhook → Google Sheets kaydı → basit PDF — *Ben verir, Sen credential bağlar*
- [ ] Bot, sonucu n8n webhook'una gönderiyor — *Beraber*

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

---

## Kararlar

- **Proje adı:** LeadLeaf AI — Derin Öğrenme ve Yapay Zekâ Ajanı Destekli Bitki Hastalığı Ön Değerlendirme Sistemi.
- **Domain / MVP sınıfları:** Domates — `healthy, Early_blight, Late_blight, Bacterial_spot, Septoria_leaf_spot` (5 sınıf).
- **DL:** Keras + MobileNetV2 transfer learning. Veri: "New Plant Diseases Dataset" (vipoooool).
- **Agent:** Anthropic Claude (`claude-sonnet-5`). **Doz/bekleme süresi ÜRETMEZ.**
- **Kullanıcı arayüzü:** Telegram botu.
- **MVP (zorunlu):** model + inference + bot + agent + n8n (Sheets+PDF).
- **Bonus (MVP sonrası, sırayla):** weather.py/db.py entegrasyonu → bölgesel uyarı → 38 sınıf → farklı açıdan foto isteme → ayrıntı seviyesi kişiselleştirme.

---

## TÜBİTAK 2204'e çevirme notu

Araştırma sorusu: *"Yaprak görüntüsüne dayalı bir derin öğrenme + LLM-agent sistemi, sınırlı sınıf/veri ile de güvenilir bir ön-değerlendirme üretebilir mi; bağlamsal genişletmeler (hava durumu, geçmiş kayıt) bu güvenilirliği nasıl etkiler?*"
Ek gerekenler: hipotez, ölçüm planı (model doğruluğu + rapor kalitesi uzman değerlendirmesi), kontrol grubu, sınırlılıklar bölümü.
