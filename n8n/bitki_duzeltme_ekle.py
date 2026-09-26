"""
Canlı n8n akışının ("LeadLeaf AI — Bitki Hastalığı Bot (tam)", wcbehfCDFv8Qb3Wj) bugünkü hâlini
okuyup "bitki düzeltmesi" özelliğini ekler ve içe aktarılabilir dosya üretir:
n8n/leadleaf_tam_akis.json (n8n'de: Workflows → Import from File). n8n veritabanını yalnızca OKUR.

Sorun (2026-09-26 gerçek test): domates fotoğrafı açıklamasız gönderildi → model %29 güvenle
"mısır — sağlıklı" dedi. Çiftçi ardından "Domates" yazınca mesaj sohbet dalına düştü; sohbet dalı
önceki fotoğrafı bilmediği için alakasız genel bir cevap verdi.

Çözüm: sohbet dalına girmeden önce FastAPI'ye sorulur (/son-foto): "bu yazı bir bitki adı mı ve bu
sohbette son 30 dakikada fotoğraf var mı?" Evetse, son fotoğrafın Telegram file_id'si fotoğraf
dalına geri beslenir ve AYNI fotoğraf o bitkinin sınıflarıyla yeniden değerlendirilir.

Çalıştırma:  .venv\\Scripts\\python n8n\\bitki_duzeltme_ekle.py
"""
import json
import os
import sqlite3
import uuid

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.expanduser("~/.n8n/database.sqlite")
CIKTI = os.path.join(KOK, "n8n", "leadleaf_tam_akis.json")
TT = "$('Telegram Trigger').item.json"

db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
nodes_json, conn_json, settings_json = db.execute(
    "select nodes, connections, settings from workflow_entity where id='wcbehfCDFv8Qb3Wj'").fetchone()
nodes = {n["name"]: n for n in json.loads(nodes_json)}
conn = json.loads(conn_json)


def yeni(name, type_, ver, pos, params, **ek):
    nodes[name] = {"parameters": params, "id": str(uuid.uuid4()), "name": name, "type": type_,
                   "typeVersion": ver, "position": pos, **ek}


def c(*hedefler):
    return [{"node": h, "type": "main", "index": 0} for h in hedefler]


# --- 1) Predict CNN: bitki yazıdan da gelebilir; chat_id + file_id saklansın diye gönderilir ------
govde = nodes["HTTP Request - Predict CNN"]["parameters"]["bodyParameters"]["parameters"]
govde[:] = [p for p in govde if p.get("name") not in ("bitki", "chat_id", "file_id")]
govde += [
    # Fotoğrafta açıklama (caption) olur; yeniden değerlendirmede ise bitki adı yazı (text) olarak gelir
    {"name": "bitki", "value": "={{ " + TT + ".message.caption || " + TT + ".message.text || '' }}"},
    {"name": "chat_id", "value": "={{ " + TT + ".message.chat.id }}"},
    {"name": "file_id", "value": "={{ $json.result.file_id }}"},
]

# --- 2) Rapor mesajı: emin değilse "bitkinin adını yazın" ipucu ------------------------------------
with open(os.path.join(KOK, "n8n", "rapor_ayristir_kod.js"), encoding="utf-8") as f:
    nodes["Rapor JSON'unu Ayrıştır"]["parameters"]["jsCode"] = f.read()

# --- 3) "İnceleniyor" mesajı yeniden değerlendirmede farklı olsun -----------------------------------
nodes["Telegram - İnceleniyor"]["parameters"]["text"] = (
    "={{ $json.yeniden_bitki ? '🔁 Son gönderdiğiniz fotoğraf ' + $json.yeniden_bitki + "
    "' olarak yeniden inceleniyor…' : '📥 Fotoğrafınız alındı.\\n🔍 Yapay zekâ modeli yaprağı inceliyor…' }}")

# --- 4) Yeni düğümler: Son Fotoğraf → Bitki düzeltmesi mi? → Yeniden Değerlendirme Hazırla ---------
yeni("HTTP Request - Son Fotoğraf", "n8n-nodes-base.httpRequest", 4.2, [-880, 816], {
    "url": "=http://127.0.0.1:8000/son-foto/{{ $json.message.chat.id }}",
    "sendQuery": True,
    "queryParameters": {"parameters": [{"name": "metin", "value": "={{ $json.message.text || '' }}"}]},
    "options": {"timeout": 5000}},
    onError="continueRegularOutput")  # FastAPI kapalıysa sohbet yine çalışsın
yeni("Bitki düzeltmesi mi?", "n8n-nodes-base.if", 2, [-640, 816], {
    "conditions": {"options": {"caseSensitive": True, "leftValue": "", "typeValidation": "strict", "version": 2},
                   "conditions": [{"id": str(uuid.uuid4()), "leftValue": "={{ $json.yeniden === true }}",
                                   "rightValue": "",
                                   "operator": {"type": "boolean", "operation": "true", "singleValue": True}}],
                   "combinator": "and"}, "options": {}})
yeni("Yeniden Değerlendirme Hazırla", "n8n-nodes-base.code", 2, [-400, 656], {
    "jsCode": (
        "// Telegram'dan gelen yazı mesajını, son fotoğrafın file_id'sini taşıyan bir fotoğraf mesajına\n"
        "// çevirir; böylece fotoğraf dalındaki düğümler (Fotoğrafı İndir, Predict CNN, ...) hiç\n"
        "// değişmeden aynı fotoğrafı yeniden işler. Bitki adı Predict CNN'e mesaj metninden gider.\n"
        "const mesaj = $('Telegram Trigger').first().json.message;\n"
        "const son = $input.first().json;\n"
        "return [{ json: { message: { ...mesaj, photo: [{ file_id: son.file_id }] }, yeniden_bitki: son.bitki } }];\n")})

# --- 5) Sohbet: girdisi artık Telegram Trigger değil → metni doğrudan Trigger'dan al; düzeltme ipucu --
sohbet = nodes["Basic LLM Chain - Sohbet"]["parameters"]
sohbet["text"] = "={{ " + TT + ".message.text }}"
sistem = sohbet["messages"]["messageValues"][0]
ek = ("\n\nKullanıcı önceki bir fotoğrafın sonucuna itiraz ediyorsa (ör. \"bu mısır değil\"), ondan SADECE "
      "bitkinin adını yazmasını iste (ör. \"domates\"); böylece aynı fotoğraf o bitkiye göre yeniden "
      "değerlendirilir. Fotoğrafı göremediğini, bu yüzden yeniden göndermesine gerek olmadığını söyle.")
if "itiraz ediyorsa" not in sistem["message"]:
    sistem["message"] = sistem["message"].replace("\n\nKısa ve sade", ek + "\n\nKısa ve sade")
    assert "itiraz ediyorsa" in sistem["message"], "sohbet promptunda beklenen yer bulunamadı"

# --- 6) Bağlantılar ----------------------------------------------------------------------------------
conn["Selamlaşma mı?"] = {"main": [c("Telegram - Tanıtım"), c("HTTP Request - Son Fotoğraf")]}
conn["HTTP Request - Son Fotoğraf"] = {"main": [c("Bitki düzeltmesi mi?")]}
conn["Bitki düzeltmesi mi?"] = {"main": [c("Yeniden Değerlendirme Hazırla"), c("Basic LLM Chain - Sohbet")]}
conn["Yeniden Değerlendirme Hazırla"] = {"main": [c("Telegram - İnceleniyor", "Fotoğrafı İndir")]}

# --- 7) Yerleşim: sohbet düğümlerini yeni düğümlere yer açmak için sağa kaydır ------------------------
for ad, (x, y) in {"Basic LLM Chain - Sohbet": (-400, 880), "Anthropic Chat Model - Sohbet": (-400, 1000),
                   "Telegram - Sohbet Cevabı": (-96, 880)}.items():
    nodes[ad]["position"] = [x, y]

bilinmeyen = [h["node"] for v in conn.values() for dal in v.get("main", []) for h in dal if h["node"] not in nodes]
assert not bilinmeyen, bilinmeyen
akis = {"name": "LeadLeaf AI — Bitki Hastalığı Bot (tam v2)", "nodes": list(nodes.values()), "connections": conn,
        "settings": json.loads(settings_json or "{}"), "active": False}
with open(CIKTI, "w", encoding="utf-8") as f:
    json.dump(akis, f, ensure_ascii=False, indent=2)
print(f"Kaydedildi: {CIKTI} · {len(nodes)} düğüm")
