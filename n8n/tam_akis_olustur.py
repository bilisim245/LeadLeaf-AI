"""
Canlı n8n akışının (SuklzMNlxzUJN6xQ) bugünkü hâlini okuyup profesyonel_gorunum.md'deki bütün
değişiklikleri uygulanmış TEK bir akış dosyası üretir: n8n/leadleaf_tam_akis.json
(n8n'de: Workflows → Import from File). n8n veritabanını yalnızca OKUR, hiçbir şey yazmaz.

Çalıştırma:  .venv\\Scripts\\python n8n\\tam_akis_olustur.py
"""
import copy
import json
import os
import sqlite3
import uuid

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.expanduser("~/.n8n/database.sqlite")
CIKTI = os.path.join(KOK, "n8n", "leadleaf_tam_akis.json")
TG = {"telegramApi": {"id": "wPqVk0IPIK85TlyK", "name": "Telegram account 2"}}
TT = "$('Telegram Trigger').item.json"
PC = "$('HTTP Request - Predict CNN').item.json"

db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
nodes_json, conn_json, settings_json = db.execute(
    "select nodes, connections, settings from workflow_entity where id='SuklzMNlxzUJN6xQ'").fetchone()
eski = {n["name"]: n for n in json.loads(nodes_json)}
nodes = {ad: copy.deepcopy(n) for ad, n in eski.items()}


def yeni(name, type_, ver, pos, params, cred=None):
    n = {"parameters": params, "id": str(uuid.uuid4()), "name": name, "type": type_, "typeVersion": ver,
         "position": pos}
    if cred:
        n["credentials"] = cred
    nodes[name] = n


def if_(ifade):
    return {"conditions": {"options": {"caseSensitive": True, "leftValue": "", "typeValidation": "strict", "version": 2},
                           "conditions": [{"id": str(uuid.uuid4()), "leftValue": ifade, "rightValue": "",
                                           "operator": {"type": "boolean", "operation": "true", "singleValue": True}}],
                           "combinator": "and"}, "options": {}}


def konum(ad, x, y):
    nodes[ad]["position"] = [x, y]


# --- 1) Mevcut düğümlerde değişiklikler -------------------------------------------------------
nodes["Telegram Trigger"]["parameters"]["updates"] = ["message", "callback_query"]
with open(os.path.join(KOK, "n8n", "rapor_ayristir_kod.js"), encoding="utf-8") as f:
    nodes["Rapor JSON'unu Ayrıştır"]["parameters"]["jsCode"] = f.read()
cevap = nodes["Telegram - Cevap Gönder"]["parameters"]
cevap["text"] = "={{ $json.telegram_mesaji }}"
cevap["additionalFields"] = {"appendAttribution": False, "parse_mode": "HTML"}
nodes["Telegram - Sohbet Cevabı"]["parameters"].setdefault("additionalFields", {})["appendAttribution"] = False
govde = nodes["HTTP Request - Predict CNN"]["parameters"]["bodyParameters"]["parameters"]
if not any(p.get("name") == "bitki" for p in govde):
    govde.append({"parameterType": "formData", "name": "bitki",
                  "value": "={{ " + TT + ".message.caption || '' }}"})

# --- 2) "İnceleniyor" + ön tespit -------------------------------------------------------------
yeni("Telegram - İnceleniyor", "n8n-nodes-base.telegram", 1.2, [0, 0], {
    "chatId": "={{ $json.message.chat.id }}",
    "text": "📥 Fotoğrafınız alındı.\n🔍 Yapay zekâ modeli yaprağı inceliyor…",
    "additionalFields": {"appendAttribution": False}}, TG)
yeni("Telegram - Ön Tespit", "n8n-nodes-base.telegram", 1.2, [0, 0], {
    "operation": "editMessageText", "messageType": "message",
    "chatId": "={{ " + TT + ".message.chat.id }}",
    "messageId": "={{ $('Telegram - İnceleniyor').first().json.result.message_id }}",
    "text": ("={{ (() => { const ipuclari = ["
             "'En doğru sonuç için yaprağı gün ışığında ama doğrudan güneş altında değil, gölgede çekin.',"
             "'Fotoğrafta tek bir yaprak olsun ve ekranın büyük kısmını kaplasın.',"
             "'Lekeli yüzeyi net görünecek şekilde, yaprağa tepeden ve yakından çekin.',"
             "'Belirtinin en belirgin olduğu yaprağı seçin; tamamen kurumuş ya da çürümüş yapraklar yanıltıcı olabilir.',"
             "'Fotoğraf açıklamasına bitkinin adını yazarsanız (ör. şeftali) teşhis daha isabetli olur.'];"
             " const ipucu = ipuclari[Math.floor(Math.random() * ipuclari.length)];"
             " const not_ = $json.uzmana_yonlendir ? ' (model emin değil)' : '';"
             " return '✅ Fotoğraf alındı\\n✅ Model inceledi: ' + $json.hastalik_tr + ' (%'"
             " + String($json.guven).replace('.', ',') + ')' + not_ + '\\n'"
             " + '⏳ Bilgi kaynakları taranıyor, rapor hazırlanıyor (yaklaşık 15 saniye)…\\n\\n💡 İpucu: ' + ipucu; })() }}"),
    "additionalFields": {}}, TG)

yeni("Telegram - Durum: Hazır", "n8n-nodes-base.telegram", 1.2, [0, 0], {
    "operation": "editMessageText", "messageType": "message",
    "chatId": "={{ " + TT + ".message.chat.id }}",
    "messageId": "={{ $('Telegram - İnceleniyor').first().json.result.message_id }}",
    "text": ("={{ '✅ Fotoğraf alındı\\n✅ Model inceledi: ' + " + PC + ".hastalik_tr + ' (%' + "
             "String(" + PC + ".guven).replace('.', ',') + ')\\n✅ Rapor hazırlandı 👇' }}"),
    "additionalFields": {}}, TG)

# --- 3) Selamlaşmada sabit tanıtım ------------------------------------------------------------
yeni("Selamlaşma mı?", "n8n-nodes-base.if", 2, [0, 0], if_(
    "={{ (() => { const t = ($json.message.text || '').toLocaleLowerCase('tr').trim(); "
    "return t.startsWith('/start') || t.startsWith('/yardim') || "
    "/^(merhaba|mrb|selam|slm|selamlar|sa|hey|günaydın|iyi günler|iyi akşamlar|yardım|yardim)[\\s!.,?]*(nasılsın|nasilsin|naber)?[\\s!.,?]*$/.test(t); })() }}"))
yeni("Telegram - Tanıtım", "n8n-nodes-base.telegram", 1.2, [0, 0], {
    "chatId": "={{ $json.message.chat.id }}",
    "text": ("🌿 Merhaba, ben LeadLeaf. Yaprak fotoğrafından bitki hastalığı ön değerlendirmesi yapıyorum.\n\n"
             "<b>Nasıl kullanılır?</b>\nHastalıklı görünen tek bir yaprağın fotoğrafını çekip bana gönderin. "
             "Açıklamaya bitkinin adını yazarsanız (ör. \"şeftali\") daha doğru sonuç veririm.\n\n"
             "<b>Tanıdığım bitkiler:</b> domates, patates, biber, elma, üzüm, mısır, şeftali, kiraz, kabak, çilek, "
             "portakal, ahududu, soya, yaban mersini.\n\n"
             "Sonuçlarım kesin teşhis değildir. İlaç ve doz önermem, bunun için ziraat mühendisine danışın."),
    "additionalFields": {"appendAttribution": False, "parse_mode": "HTML"}}, TG)

# --- 4) PDF butonu ----------------------------------------------------------------------------
with open(os.path.join(KOK, "n8n", "pdf_dugumleri.json"), encoding="utf-8") as f:
    pdf = json.load(f)
for n in pdf["nodes"]:
    nodes[n["name"]] = n
# --- 5) Uzmana yönlendirme + takip ------------------------------------------------------------
with open(os.path.join(KOK, "n8n", "uzman_ve_takip_dugumleri.json"), encoding="utf-8") as f:
    uz = json.load(f)
for n in uz["nodes"]:
    nodes[n["name"]] = n
# Sunum için takip hatırlatması 1 dakika sonra gelsin (gerçek kullanımda 3 gün: amount=3, unit="days")
nodes["Bekle (takip)"]["parameters"].update({"amount": 1, "unit": "minutes"})

# --- 6) Bağlantılar ---------------------------------------------------------------------------
def c(*hedefler):
    return [{"node": h, "type": "main", "index": 0} for h in hedefler]


conn = json.loads(conn_json)
conn.update(pdf["connections"])
conn.update(uz["connections"])
conn["Telegram Trigger"] = {"main": [c("Buton mu?")]}
conn["Buton mu?"] = {"main": [c("Telegram - Butonu Onayla"), c("Fotoğraf var mı?")]}
conn["Fotoğraf var mı?"] = {"main": [c("Telegram - İnceleniyor", "Fotoğrafı İndir"), c("Selamlaşma mı?")]}
conn["Selamlaşma mı?"] = {"main": [c("Telegram - Tanıtım"), c("Basic LLM Chain - Sohbet")]}
conn["HTTP Request - Predict CNN"] = {"main": [c("Telegram - Ön Tespit", "HTTP Request - RAG Context")]}
conn["Rapor JSON'unu Ayrıştır"] = {"main": [c("Telegram - Durum: Hazır", "Telegram - Cevap Gönder", "Google Sheets - Kaydet",
                                              "HTTP Request - Raporu Kaydet", "Güven < %70 mi?",
                                              "Hastalık var mı?")]}

# --- 7) Yerleşim: n8n kardeş dalları yukarıdan aşağı çalıştırır; "Bekle" en altta olmalı ---------
yer = {
    "Telegram Trigger": (-600, 0), "Buton mu?": (-380, 0), "Fotoğraf var mı?": (-160, -200),
    "Telegram - İnceleniyor": (80, -420), "Fotoğrafı İndir": (80, -200),
    "HTTP Request - Predict CNN": (300, -200), "Telegram - Ön Tespit": (540, -420),
    "HTTP Request - RAG Context": (540, -200), "Basic LLM Chain": (780, -200),
    "Anthropic Chat Model": (780, 0), "Rapor JSON'unu Ayrıştır": (1100, -200),
    "Telegram - Durum: Hazır": (1360, -680), "Telegram - Cevap Gönder": (1360, -520), "Google Sheets - Kaydet": (1360, -360),
    "HTTP Request - Raporu Kaydet": (1360, -200), "Telegram - PDF Sorusu": (1600, -200),
    "Güven < %70 mi?": (1360, -40), "Telegram - Uzmana Bildir": (1600, -60),
    "Hastalık var mı?": (1360, 140), "Bekle (takip)": (1600, 120), "Telegram - Takip Hatırlatması": (1840, 120),
    "Selamlaşma mı?": (80, 160), "Telegram - Tanıtım": (320, 100), "Basic LLM Chain - Sohbet": (320, 260),
    "Anthropic Chat Model - Sohbet": (320, 440), "Telegram - Sohbet Cevabı": (620, 260),
    "Telegram - Butonu Onayla": (-160, 620), "PDF istendi mi?": (80, 620),
    "Telegram - PDF Hazırlanıyor": (320, 560), "HTTP Request - PDF Oluştur": (560, 560),
    "Telegram - PDF Gönder": (800, 560), "Telegram - PDF İstenmedi": (320, 720),
}
for ad, (x, y) in yer.items():
    konum(ad, x, y)

bilinmeyen = [h["node"] for v in conn.values() for dal in v.get("main", []) for h in dal if h["node"] not in nodes]
assert not bilinmeyen, bilinmeyen
akis = {"name": "LeadLeaf AI — Bitki Hastalığı Bot (tam)", "nodes": list(nodes.values()), "connections": conn,
        "settings": json.loads(settings_json or "{}"), "active": False}
with open(CIKTI, "w", encoding="utf-8") as f:
    json.dump(akis, f, ensure_ascii=False, indent=2)
print(f"Kaydedildi: {CIKTI} · {len(nodes)} düğüm")
