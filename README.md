# LeadLeaf AI — Derin Öğrenme ve Yapay Zekâ Ajanı Destekli Bitki Hastalığı Ön Değerlendirme Sistemi

**Bootcamp projesi.** Kaynak doküman: "DL + LLM-Agent + n8n Birleşimi" → *Görüntüden Rapor* → *A) Tarım — Bitki Hastalığı Tespiti*.

**Tek cümle:** Çiftçi Telegram botuna domates yaprağı fotoğrafı gönderir → derin öğrenme modeli hastalığı sınıflandırır (+ % güven) → LLM-Agent anlaşılır bir ön-değerlendirme yazar, kültürel/biyolojik önlem önerir, düşük güvende "ziraat mühendisine danış" der → n8n PDF üretir, Google Sheets'e kaydeder, Telegram'dan cevap döner.

> **İleride TÜBİTAK 2204 / TEKNOFEST'e çevirmek için:** aynı kod tabanı kalır; "Sonraki Aşama" bölümündeki özgünlük bileşenleri (hava durumu, bölgesel uyarı, tarla defteri, 38 sınıfa genişleme) eklenir + bilimsel soru/hipotez/ölçüm planı yazılır.

---

## 10 günlük ZORUNLU kapsam (MVP) — bu bitmeden hiçbir bonusa dokunulmaz

```
Telegram foto → MobileNetV2 (5 sınıf: domates saglikli + 4 yaygin hastalik)
   → hastalik + %guven
   → LLM-Agent: guven kontrolu + anlasilir on-degerlendirme
              + kulturel/biyolojik onlem (ILAC DOZU DEGIL)
              + "tani degil, on-degerlendirme" uyarisi
              + %70 alti guven -> "ziraat muhendisine danis"
   → n8n: Google Sheets kaydi + basit PDF
   → Telegram cevabi
```

Neden 5 sınıf (38 değil): aynı mimariyle risk çok daha düşük, model hızlı ve yüksek doğrulukla eğitilir. Sonra tek satır değişiklikle (`SELECTED_CLASSES = None`) 38 sınıfa büyür.

**Neden ilaç dozu değil, kültürel/biyolojik önlem:** ruhsatlı ürün/doz bilgisi zamanla değişir; LLM'in (RAG olsa bile) hatalı/güncel-olmayan doz üretmesi gerçek zarar riski taşır. Bunun yerine: hastalık hakkında doğrulanmış bilgi + kültürel-biyolojik önlem + kaynak (varsa) + **ziraat mühendisine yönlendirme**.

---

## Sonraki Aşama — TÜBİTAK / TEKNOFEST özgünlük bileşenleri (MVP'ye dahil DEĞİL)

Bunlar **zorunlu teslimden çıkarıldı** çünkü 10 günde ya entegrasyon riski yüksek ya da gerçek veriyle test edilemez (ör. bölgesel kümelenmeyi 1-2 test kullanıcısıyla gösteremezsin):

| Bileşen | Durum | Not |
|---|---|---|
| Hava durumu (Open-Meteo) | `agent/weather.py` yazılı + test edildi, **entegre değil** | Bonus döneminde bota bağlanır |
| Tarla defteri / zaman içi takip | `bot/db.py` yazılı + test edildi, **entegre değil** | Bonus döneminde bota bağlanır |
| Bölgesel hastalık uyarısı | Yapılmadı | Gerçek kullanıcı tabanı olmadan demo edilemez; TÜBİTAK aşamasında |
| Konum tabanlı hastalık haritası | Yapılmadı | TÜBİTAK aşamasında |
| Gelişmiş RAG (çok kaynaklı) | Yapılmadı | TÜBİTAK aşamasında |
| 38 sınıf / çoklu bitki | Notebook'ta hazır (`SELECTED_CLASSES = None`) | MVP çalışınca açılır |
| Düşük güvende farklı açıdan foto isteme | Yapılmadı | Kolay ek, TÜBİTAK aşamasında |
| Öğrenci/çiftçi için farklı ayrıntı seviyesi | Yapılmadı | TÜBİTAK aşamasında |

---

## Katmanlar (MVP)

| Katman | Ne yapar | Araç |
|---|---|---|
| **DL modeli** | Yaprak fotoğrafı → 5 sınıftan biri + % güven | Keras/TensorFlow, MobileNetV2 transfer learning |
| **Inference servisi** | Modeli HTTP ile sunar (`/predict`) | FastAPI |
| **Telegram botu** | Kullanıcı arayüzü: foto al, cevap ver | python-telegram-bot |
| **LLM-Agent** | Güven kontrolü, kültürel/biyolojik öneri, JSON rapor, eskalasyon | Anthropic Claude |
| **n8n** | Webhook → Google Sheets kaydı → PDF | n8n (Docker) |

---

## Klasör yapısı

```
bitki-hastalik-tespiti/
├── README.md
├── PROGRESS.md                      ← günlük ilerleme takibi
├── requirements.txt
├── .env.example
├── notebooks/
│   └── 01_train_model_kaggle.py     Kaggle GPU'da model eğitimi (domates 5 sınıf)
├── model/                           model.keras + class_names.json (Kaggle'dan iner)
├── inference/
│   └── app.py                       FastAPI: görsel → {hastalik, guven, ilk3}   [Gün 4]
├── agent/
│   ├── agent.py                     teşhis → JSON rapor (kültürel/biyolojik öneri)  [Gün 5-6]
│   ├── weather.py                   BEKLEMEDE — bonus aşamasında entegre edilecek
│   └── knowledge/                   RAG kaynak dokümanları (doğrulanmış bilgi)   [Gün 8, bonus]
├── rag/
│   └── build_index.py               Chroma index  [Gün 8, bonus]
├── bot/
│   ├── telegram_bot.py              ana bot: foto al, agent çağır, cevap ver     [Gün 5-6]
│   └── db.py                        BEKLEMEDE — bonus aşamasında entegre edilecek
├── n8n/
│   ├── docker-compose.yml
│   └── workflow_rapor.json          Webhook → Sheets → PDF                       [Gün 7]
├── ui/
│   └── app.py                       Gradio — jüriye hızlı gösterim / bot yedeği  [opsiyonel]
└── report/
    └── rapor_taslagi.md             bootcamp raporu + sunum notları              [Gün 9]
```

---

## Gün 0 — Kurulum (hepsi ücretsiz)

1. **Kaggle:** kaggle.com → üye ol → Settings'ten telefon doğrula (GPU için).
2. **GitHub:** github.com → `LeadLeaf-AI` (veya `bitki-hastalik-tespiti`) adında boş repo.
3. **Anthropic:** console.anthropic.com → API Keys → anahtar oluştur, kaydet.
4. **Telegram botu:** Telegram'da **@BotFather** → `/newbot` → token'ı kaydet.
5. **Docker Desktop** kur (n8n günü için).
6. Python 3.10–3.12 kurulu olsun.
7. Yerel ortam:
   ```powershell
   cd C:\Users\90539\bitki-hastalik-tespiti
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

---

## 10 günlük plan (MVP)

| Gün | Hedef | Çıktı |
|----|-------|-------|
| 1 | Kurulum + veri inceleme + repo | Hesaplar hazır, GitHub repo açık, domates 5 sınıfı görüldü |
| 2–3 | **Model eğitimi** (`01_train_model_kaggle.py`, `SELECTED_CLASSES` = domates 5 sınıf) | Doğrulama doğruluğu ≥ %90, confusion matrix; `model.keras` + `class_names.json` + `demo_images/` → `model/` |
| 4 | Inference servisi | `POST /predict` görsel → `{hastalik, guven, ilk3}` |
| 5 | Telegram botu iskeleti | Bota foto at → `/predict` çağrılır → ham teşhis mesajı |
| 6 | LLM-Agent | Güven kontrolü + kültürel/biyolojik öneri + JSON rapor; bot düzgün mesaj döner |
| 7 | n8n | `docker compose up` → Webhook → Google Sheets kaydı → basit PDF |
| 8 | Uçtan uca test + uç durumlar | Yaprak olmayan görsel, düşük güven, bilinmeyen sınıf senaryoları |
| 9 | Rapor | `report/rapor_taslagi.md`: metodoloji, metrikler, örnek çıktılar, sınırlılıklar |
| 10 | Sunum + GitHub | Slaytlar + demo videosu + repoya son push |

**Bonus (MVP bitmeden başlanmaz):** `weather.py`/`db.py` entegrasyonu, bölgesel uyarı, konum haritası, 38 sınıf.

---

## Etik / sınırlılık notu (rapora da girecek)

- Sistem **kesin teşhis koymaz**, ön değerlendirme yapar; her çıktıda uyarı bulunur.
- Düşük güvende ("%70 altı") otomatik "uzmana danış" mesajı.
- **İlaç dozu/bekleme süresi önerilmez** — sadece kültürel/biyolojik önlem + doğrulanmış bilgi + uzmana yönlendirme.
- Model laboratuvar görselleriyle eğitildi; gerçek tarla fotoğraflarında başarım düşebilir — sınırlılık olarak belirtilecek.
- MVP sadece domates + 5 sınıf; genellenebilirlik sınırlıdır.

---

## Sık sorunlar

- **Kaggle GPU yok:** Notebook → sağ panel → Accelerator = GPU; telefon doğrulaması gerekli.
- **Model yerelde yüklenmiyor:** Kaggle'daki TensorFlow sürümünü not al, yerelde aynısını kur.
- **Bot cevap vermiyor:** BotFather token'ı `.env`'de doğru mu; `telegram_bot.py` çalışıyor mu.
- **n8n Sheets/Gmail yetki hatası:** credential'lar n8n arayüzünden bağlanır (~5 dk).
