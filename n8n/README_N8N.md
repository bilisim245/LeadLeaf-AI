# n8n Kurulumu — Local (Node.js ile, Docker'sız, Cloud'suz)

**Mimari karar güncellemesi (2026-09-20) — n8n Cloud'dan vazgeçildi:** 2026-09-15'te
"local Docker yerine n8n Cloud" kararı alınmıştı, o sırada n8n Cloud'un kalıcı ücretsiz bir
planı olduğu varsayılmıştı. **Bu varsayım yanlış çıktı** — n8n Cloud artık sadece **14 günlük
ücretsiz deneme** sunuyor, sonrasında en ucuz plan bile $20/ay. Kullanıcı ücret ödemek
istemediği için bu karardan vazgeçildi.

**Yeni karar: n8n'i kendi bilgisayarında, Docker'sız çalıştır.** n8n aslında bir Node.js
uygulaması (npm paketi) — Docker'a hiç gerek yok, sadece Node.js yeterli:

```powershell
npx n8n start
```

`npx`, Node.js ile birlikte gelen bir araç: "bu paketi (n8n) indir (daha önce indirilmediyse)
ve çalıştır" demek. Ayrı bir "installer" indirip kurmuyorsun — ilk çalıştırmada npm'in resmi
paket deposundan indirilip anında başlıyor (ilk seferinde ~1 dakika sürebilir, sonraki
çalıştırmalarda paket diskte olduğu için çok daha hızlı açılır). Node.js'in kendisi
https://nodejs.org 'dan (ya da `winget install OpenJS.NodeJS.LTS`) kurulur.

n8n varsayılan olarak `http://localhost:5678` adresinde açılır — tarayıcıdan girip
kullanıyorsun, hesap/kayıt gerekmiyor (self-hosted community edition'da ilk girişte
kendi owner hesabını sen oluşturuyorsun, dışarıya kayıt yok).

## Neden Docker değil?

Docker, n8n'i izole bir "konteyner" içinde çalıştırmak için ayrı bir sanallaştırma programı
(Docker Desktop) kurmanı ister — Windows'ta bazen WSL2 ayarı gerektirir. n8n zaten bir Node.js
programı olduğu için, Node.js kuruluysa doğrudan çalıştırılabilir — ekstra bir katman
gerekmiyor. Bkz. `PROGRESS.md`'deki 2026-09-20 kaydı.

## Mimari — tek tünel yeterli

Local n8n + local FastAPI inference **aynı bilgisayarda** çalıştığı için aralarında tünele
gerek YOK — n8n, `HTTP Request - Predict CNN` node'unda doğrudan `http://localhost:8000/predict`
adresini çağırabilir. **Tek gereken tünel:** Telegram'ın n8n'in webhook'una ulaşabilmesi için.

```
Telegram → (ngrok tüneli) → local n8n (localhost:5678) → local FastAPI (localhost:8000) → Claude API → Telegram cevabı
```

## 1) n8n'i başlat

```powershell
$env:WEBHOOK_URL = "https://SENIN-SABIT-NGROK-DOMAININ/"
npx n8n start
```

`WEBHOOK_URL` önemli: n8n, Telegram Trigger node'unu aktif ettiğinde Telegram'a "webhook'umu
şu adrese gönder" diye bu URL'i bildiriyor — bu yüzden n8n'i başlatmadan ÖNCE bu ortam
değişkenini ayarlamak gerekiyor.

## 2) ngrok tünelini n8n'in portuna (5678) yönlendir

```powershell
ngrok http --url=SENIN-SABIT-NGROK-DOMAININ 5678
```

(Not: inference servisi için AYRI bir tünele gerek yok — n8n ona `localhost:8000` üzerinden
zaten ulaşıyor.)

## 3) Tarayıcıdan n8n'e gir, workflow'u import et

**Adım adım (2026-09-20'de gerçekten denenip doğrulanan yol):**

1. `http://localhost:5678` adresine git → ilk girişte kendi owner hesabını oluştur
   (email/şifre — bu sadece kendi bilgisayarında saklanıyor, n8n.io'ya kayıt DEĞİL).
2. Sol üstteki **"+"** ikonuna (ya da "Personal" altındaki "+ Add workflow"a) tıkla →
   **"New workflow"** → boş bir canvas açılır.
3. Canvas'ın sağ üst köşesindeki **"..."** (üç nokta) menüsüne tıkla → **Import** →
   **From file**.
4. ⚠️ **Bilinen sorun:** Bu adımda açılması gereken native "Aç" penceresi bazı
   sistemlerde beklendiği gibi davranmayabilir (ör. dosyayı seçmek yerine varsayılan
   uygulamayla — Kod editörü gibi — açabilir). Bu olursa **alternatif yöntem**: dosyanın
   TAMAMINI kopyala (VS Code'da aç → Ctrl+A → Ctrl+C, ya da PowerShell'de
   `Get-Content -Raw n8n\workflow.json | Set-Clipboard`), sonra n8n canvas'ının BOŞ bir
   yerine tıkla ve klavyeden **Ctrl+V** yapıştır — n8n panodaki JSON'u otomatik tanıyıp
   tüm node'ları canvas'a çizer. (Not: bu yapıştırmayı bir OTOMASYON aracının sentetik
   tuş basışıyla yapmaya çalışmak çalışmayabilir — tarayıcının pano izni gerçek bir
   kullanıcı tuşuna basmasını isteyebiliyor; gerçek elle Ctrl+V basmak güvenilir yol.)
5. Import doğru gittiyse, sekme başlığı ve canvas'taki node isimleri
   `workflow.json`'daki isimlerle (Telegram Trigger, Fotoğrafı İndir, HTTP Request -
   Predict CNN, ...) birebir eşleşir — toplam 6 node + 1 sticky note (talimat notu).

## 4) Credential'lar (n8n arayüzü > Credentials)

| Credential | Tür | Not |
|---|---|---|
| Telegram Bot (LeadLeaf) | Telegram API | BotFather token'ı |
| Anthropic API Key | HTTP Header Auth | Header adı: `x-api-key`, değer: senin Anthropic API key'in |
| Google Sheets (LeadLeaf) | Google Sheets OAuth2 | Google hesabınla yetkilendir |

**Nasıl bağlanır (her node için aynı akış):**

1. Canvas'ta o node'a (örn. "Telegram Trigger") **çift tıkla** — sağda ayar paneli açılır.
2. "Credential to connect with" (ya da "Authentication") alanındaki açılır menüden
   **"Create new credential"** seç.
3. Açılan formu doldur (Telegram için sadece bot token'ı; Anthropic için Header Auth →
   isim `x-api-key`, değer kendi API key'in; Google Sheets için "Sign in with Google"
   butonuna basıp tarayıcıda hesabını seçmen yeterli).
4. **Save** — bundan sonra o credential, aynı türdeki (Telegram/Anthropic/Sheets)
   TÜM node'larda açılır menüden seçilebilir hâle gelir; yani 3 credential'ı bir kere
   oluşturman, workflow'daki tüm ilgili node'lar için yeterli (her node'a tek tek
   yeniden girmene gerek yok, sadece açılır menüden mevcut credential'ı seçersin).

Import edilen workflow'daki her node'da credential alanı `REPLACE_ME` — yukarıdaki
adımla gerçek credential'ını seçmen yeterli (id otomatik güncellenir).

## 5) Test

Telegram botuna bir domates yaprağı fotoğrafı gönder → n8n'in **Executions** sekmesinde
adım adım akışı izle → Sheets'e satır düştüğünü ve Telegram'a cevap geldiğini doğrula.

⚠️ Bilgisayarın kapanırsa (veya `npx n8n start` / `ngrok` terminalleri kapanırsa) bot
çalışmayı durdurur — demo/sunum sırasında ikisinin de açık olduğundan emin ol.

## 6) RAG'i n8n'e taşımak (opsiyonel, ileri seviye — sunumda "yol haritası" olarak anlatılabilir)

Yerel demoda (`ui/app.py`) RAG zaten çalışıyor (`agent/rag.py` + Chroma). n8n'de AYNI
seviyeye çıkmak için iki yol var:

- **Basit:** `HTTP Request - Claude Agent` node'undan ÖNCE bir HTTP Request ile
  `agent/rag.py`'nin mantığını saran küçük bir endpoint'e (`inference/app.py`'ye
  eklenecek `/rag-context?hastalik=...` gibi) istek atıp dönen metni prompt'a ekle.
- **n8n-native:** n8n'in kendi **Vector Store node'ları** (Chroma/Pinecone/Qdrant) ve
  **Embeddings node'u** ile `agent/knowledge/*.md` dosyalarını doğrudan n8n içinde
  indexleyip sorgula (LangChain tabanlı n8n AI node'ları bunu destekliyor).

Bu MVP'nin zorunlu kapsamında DEĞİL — şu an prompt (`agent/prompt_taslagi.md`) sabit
metinle çalışıyor. Sunumda "RAG'i şu an yerelde gösteriyoruz, n8n'e taşımak yol
haritasında" demek yeterli ve dürüst bir çerçeve.

## 7) Prompt geliştirme ("prompt geliştirme n8n'de")

`HTTP Request - Claude Agent` node'unun `jsonBody` alanındaki `system` metnini n8n
arayüzünden doğrudan düzenleyip test edebilirsin (execution'ı tekrar çalıştır, sonucu
gör). Sonunda beğendiğin hâli hem burada hem `agent/prompt_taslagi.md` ve
`agent/report.py`'deki `SYSTEM_PROMPT` içinde güncel tut (üçü de aynı metni taşımalı).
