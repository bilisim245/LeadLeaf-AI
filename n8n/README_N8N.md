# n8n Cloud Kurulumu (local docker yerine)

**Mimari karar güncellemesi (2026-09-15):** n8n artık local Docker yerine **n8n Cloud**
üzerinde çalışacak. Neden: kurulum daha hızlı, Telegram/Sheets credential bağlama aynı,
Docker Desktop bağımlılığı ortadan kalkıyor. Bedeli: n8n Cloud, bilgisayarındaki
`localhost:8000` inference servisine DOĞRUDAN ulaşamaz — bir **tünel** ya da **public URL**
gerekiyor (aşağıda).

## 1) n8n Cloud hesabı

1. https://n8n.io/cloud/ → ücretsiz deneme/hesap oluştur.
2. Yeni workflow → sağ üstten **Import from File** → bu klasördeki `workflow.json`'ı yükle.

## 2) Inference servisini n8n Cloud'un görebileceği hale getir

İki seçenekten biri:

### Seçenek A — Tünel (MVP için en hızlı, ücretsiz)

```powershell
# ngrok kur: https://ngrok.com/download (ücretsiz hesap yeterli)
ngrok http 8000
```

Çıktıdaki `https://xxxx.ngrok-free.app` adresini kopyala → n8n'deki
**"HTTP Request - Predict CNN"** node'unun URL alanına yapıştır (sonuna `/predict` ekle).

⚠️ Ücretsiz ngrok URL'i her yeniden başlatmada değişir — değiştikçe n8n node'unu güncelle.
Demo/sunum günü tünelin **açık** olması gerekir (bilgisayar kapanırsa bot çalışmaz).

### Seçenek B — Inference'ı küçük bir cloud'a deploy et (daha kalıcı, teslim öncesi önerilir)

Render.com / Railway.app gibi ücretsiz katmanlı bir servise `inference/` klasörünü
deploy et (Dockerfile veya doğrudan `uvicorn inference.app:app` start command'ı ile).
Böylece sabit bir public URL olur, bilgisayarın kapalı olsa bile bot çalışır.

## 3) Credential'lar (n8n Cloud > Credentials)

| Credential | Tür | Not |
|---|---|---|
| Telegram Bot (LeadLeaf) | Telegram API | BotFather token'ı |
| Anthropic API Key | HTTP Header Auth | Header adı: `x-api-key`, değer: senin Anthropic API key'in |
| Google Sheets (LeadLeaf) | Google Sheets OAuth2 | Google hesabınla yetkilendir |

Import edilen workflow'daki her node'da credential alanı `REPLACE_ME` — n8n arayüzünden
gerçek credential'ını seçmen yeterli (id otomatik güncellenir).

## 4) Test

Telegram botuna bir domates yaprağı fotoğrafı gönder → n8n execution log'unda adım adım
akışı izle → Sheets'e satır düştüğünü ve Telegram'a cevap geldiğini doğrula.

## 5) RAG'i n8n'e taşımak (opsiyonel, ileri seviye — sunumda "yol haritası" olarak anlatılabilir)

Yerel demoda (`ui/app.py`) RAG zaten çalışıyor (`agent/rag.py` + Chroma). n8n Cloud'da
AYNI seviyeye çıkmak için iki yol var:

- **Basit:** `HTTP Request - Claude Agent` node'undan ÖNCE bir HTTP Request ile
  `agent/rag.py`'nin mantığını saran küçük bir endpoint'e (`inference/app.py`'ye
  eklenecek `/rag-context?hastalik=...` gibi) istek atıp dönen metni prompt'a ekle.
- **n8n-native:** n8n'in kendi **Vector Store node'ları** (Chroma/Pinecone/Qdrant) ve
  **Embeddings node'u** ile `agent/knowledge/*.md` dosyalarını doğrudan n8n içinde
  indexleyip sorgula (LangChain tabanlı n8n AI node'ları bunu destekliyor).

Bu MVP'nin zorunlu kapsamında DEĞİL — şu an prompt (`agent/prompt_taslagi.md`) sabit
metinle çalışıyor. Sunumda "RAG'i şu an yerelde gösteriyoruz, n8n'e taşımak yol
haritasında" demek yeterli ve dürüst bir çerçeve.

## 6) Prompt geliştirme ("prompt geliştirme n8n'de")

`HTTP Request - Claude Agent` node'unun `jsonBody` alanındaki `system` metnini n8n
arayüzünden doğrudan düzenleyip test edebilirsin (execution'ı tekrar çalıştır, sonucu
gör). Sonunda beğendiğin hâli hem burada hem `agent/prompt_taslagi.md` ve
`agent/report.py`'deki `SYSTEM_PROMPT` içinde güncel tut (üçü de aynı metni taşımalı).
