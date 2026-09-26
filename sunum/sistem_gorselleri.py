"""
Sunum kitapçığının "Sistem nasıl çalışıyor?" bölümü için görseller (sunum/sistem_gorselleri/*.png):
  mimari.png         — bileşenler ve veri depoları: kim kime bağlanıyor, veri nerede duruyor
  rag_akisi.png      — RAG'in iki aşaması: hazırlık (bir kez) ve çalışırken (her fotoğrafta)
  vektor_haritasi.png— Chroma'daki 197 parçanın 384 boyutlu vektörlerinin 2 boyuta indirilmiş haritası
  n8n_akisi.png      — canlı n8n akışı (n8n/leadleaf_tam_akis.json), düğüm numaralarıyla

Veriler gerçek dosyalardan okunur (Chroma: rag/chroma_db, n8n: n8n/leadleaf_tam_akis.json).
Çalıştırma:  .venv\\Scripts\\python sunum\\sistem_gorselleri.py
"""
from __future__ import annotations

import json
import os
import textwrap

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("USE_TF", "0")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

KLASOR = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(KLASOR)
CIKTI = os.path.join(KLASOR, "sistem_gorselleri")
os.makedirs(CIKTI, exist_ok=True)

# Referans paletin ilk 3 kategorik rengi (dağılım grafiğinde de renk körlüğü testini geçen üçlü)
MAVI, TURUNCU, SU = "#2a78d6", "#eb6834", "#1baf7a"
GRI_NOKTA, METIN, IKINCIL, YUZEY = "#b9bcc4", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "text.color": METIN,
                     "axes.edgecolor": "#c8cbd2", "figure.facecolor": YUZEY, "axes.facecolor": YUZEY})


def kutu(ax, x, y, metin, renk, g=16, y_=7, boyut=8.5, kalin_ilk=True):
    """Merkezi (x, y) olan, açık dolgulu, renkli kenarlı kutu; metin koyu mürekkeple."""
    ax.add_patch(FancyBboxPatch((x - g / 2, y - y_ / 2), g, y_, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=renk + "22", ec=renk, lw=1.4))
    satirlar = metin.split("\n")
    ax.text(x, y + (len(satirlar) - 1) * 1.05, satirlar[0], ha="center", va="center", fontsize=boyut,
            fontweight="bold" if kalin_ilk else "normal")
    for i, s in enumerate(satirlar[1:], 1):
        ax.text(x, y + (len(satirlar) - 1) * 1.05 - i * 2.1, s, ha="center", va="center", fontsize=boyut - 1.3,
                color=IKINCIL)


def ok(ax, a, b, etiket="", renk="#6b7080", egri=0.0, etiket_kay=(0, 0)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=renk,
                                 connectionstyle=f"arc3,rad={egri}", shrinkA=2, shrinkB=2))
    if etiket:
        ax.text((a[0] + b[0]) / 2 + etiket_kay[0], (a[1] + b[1]) / 2 + etiket_kay[1], etiket, fontsize=7,
                color=IKINCIL, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc=YUZEY, ec="none"))


def lejant(ax, ogeler, x=1, y=2):
    for i, (renk, ad) in enumerate(ogeler):
        ax.add_patch(FancyBboxPatch((x + i * 24, y - 1), 3, 2, boxstyle="round,pad=0.1", fc=renk + "22", ec=renk))
        ax.text(x + i * 24 + 4.2, y, ad, fontsize=8, va="center")


# ---------------------------------------------------------------------------------------------
# 1) Mimari: bileşenler + veri depoları
def mimari():
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_xlim(0, 100), ax.set_ylim(0, 72), ax.axis("off")
    ax.text(0, 70.5, "LeadLeaf AI — bileşenler ve veri depoları", fontsize=13, fontweight="bold", va="top")
    ax.text(0, 67.3, "Oklar: kim kime istek gönderiyor. Turuncu kutular verinin durduğu yerler (veritabanı / dosya).",
            fontsize=8.5, color=IKINCIL, va="top")
    S, V, D = MAVI, TURUNCU, SU
    kutu(ax, 9, 57, "Çiftçi\nTelegram uygulaması", D, g=15)
    kutu(ax, 29, 57, "Telegram sunucuları\n(bot mesajları)", D, g=17)
    kutu(ax, 49, 57, "ngrok tüneli\nsabit internet adresi", D, g=17)
    kutu(ax, 70, 57, "n8n akışı\nbilgisayar · port 5678", S, g=17)
    kutu(ax, 91, 57, "Claude API\n(Anthropic)", D, g=15)
    kutu(ax, 44, 43, "n8n veritabanı (SQLite)\n~/.n8n/database.sqlite", V, g=22)
    kutu(ax, 91, 43, "Google Sheets\nanaliz kayıtları", V, g=15)
    kutu(ax, 60, 29, "FastAPI servisi (inference/app.py)\nport 8000 · /predict · /rag-context · /son-foto · /rapor-pdf",
         S, g=46)
    kutu(ax, 14, 29, "Streamlit panosu\nport 8501 · sunum + Canlı Demo", S, g=22)
    kutu(ax, 30, 12, "CNN modeli\nmodel/model.keras", V, g=17)
    kutu(ax, 50, 12, "Chroma (vektör DB)\nrag/chroma_db · 197 parça", V, g=19)
    kutu(ax, 70, 12, "Son fotoğraflar\ndata/son_fotolar.json", V, g=17)
    kutu(ax, 90, 12, "Raporlar (PDF için)\ndata/raporlar/*.json", V, g=17)
    kutu(ax, 10, 12, "Tarla defteri (SQLite)\nbot/tarla_defteri.sqlite", V, g=18)
    ok(ax, (16.5, 57), (20.5, 57), "")
    ok(ax, (37.5, 57), (40.5, 57), "webhook")
    ok(ax, (57.5, 57), (61.5, 57), "")
    ok(ax, (78.5, 57), (83.5, 57), "rapor iste")
    ok(ax, (63, 53.5), (50, 46.7), "akışlar, anahtarlar,\nçalıştırma geçmişi", etiket_kay=(-9, 1))
    ok(ax, (78, 54), (86, 46.7), "satır ekle", etiket_kay=(3, 1))
    ok(ax, (70, 53.3), (70, 32.7), "HTTP istekleri", etiket_kay=(0, 3))
    ok(ax, (25, 29), (37, 29), "HTTP")
    for x in (30, 50, 70, 90):
        ok(ax, (min(max(x, 40), 80), 25.3), (x, 15.7))
    ok(ax, (14, 25.3), (11, 15.7))
    ax.text(14, 35.5, "Canlı Demo raporu için Claude'a da bağlanır (agent/report.py)", fontsize=7, color=IKINCIL,
            ha="center")
    lejant(ax, [(S, "Bizim servisimiz"), (V, "Veri deposu"), (D, "Dış hizmet / kullanıcı")])
    fig.savefig(os.path.join(CIKTI, "mimari.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 2) RAG akışı
def rag_akisi():
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 100), ax.set_ylim(0, 54), ax.axis("off")
    ax.text(0, 53, "RAG iki aşamada çalışır", fontsize=13, fontweight="bold", va="top")
    ax.text(0, 46.5, "1) HAZIRLIK — bir kez (python rag/build_index.py)", fontsize=10, fontweight="bold", color=MAVI)
    adimlar1 = ["38 bilgi dosyası\nagent/knowledge/*.md", "Başlıklara göre böl\n'## Belirtiler' vb. → 197 parça",
                "Embedding modeli\nher parça → 384 sayı", "Chroma'ya kaydet\nmetin + sınıf + vektör"]
    for i, t in enumerate(adimlar1):
        kutu(ax, 12 + i * 25, 38, t, MAVI, g=21, y_=8)
        if i:
            ok(ax, (12 + (i - 1) * 25 + 10.8, 38), (12 + i * 25 - 10.8, 38))
    ax.text(0, 25.5, "2) ÇALIŞIRKEN — her fotoğrafta, otomatik (FastAPI /rag-context)", fontsize=10,
            fontweight="bold", color=TURUNCU)
    adimlar2 = ["CNN sonucu\n'Tomato___Late_blight'", "Aynı modelle\nsorguyu 384 sayıya çevir",
                "Chroma'da ara\nsadece o sınıf · en yakın 2", "Claude'a ver\nrapor bu metne dayanır"]
    for i, t in enumerate(adimlar2):
        kutu(ax, 12 + i * 25, 17, t, TURUNCU, g=21, y_=8)
        if i:
            ok(ax, (12 + (i - 1) * 25 + 10.8, 17), (12 + i * 25 - 10.8, 17))
    ok(ax, (87, 33.5), (62, 21.5), "aynı veritabanı", renk=IKINCIL, egri=0.15)
    ax.text(0, 5, "Hiçbir model EĞİTİLMEZ: parçalar sadece sayıya çevrilip saklanır. Bilgi değişirse hazırlık adımı "
                  "yeniden çalıştırılır (birkaç saniye).", fontsize=8.5, color=IKINCIL)
    fig.savefig(os.path.join(CIKTI, "rag_akisi.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 3) Vektör haritası: 197 parça, 384 boyut → 2 boyut (PCA)
BITKI_TR = {"Tomato": "Domates", "Potato": "Patates", "Pepper,_bell": "Biber", "Apple": "Elma", "Peach": "Şeftali",
            "Cherry_(including_sour)": "Kiraz", "Grape": "Üzüm", "Corn_(maize)": "Mısır", "Strawberry": "Çilek",
            "Orange": "Portakal", "Raspberry": "Ahududu", "Soybean": "Soya", "Squash": "Kabak",
            "Blueberry": "Yaban mersini"}


def vektor_haritasi():
    import chromadb
    from sentence_transformers import SentenceTransformer

    col = chromadb.PersistentClient(path=os.path.join(KOK, "rag", "chroma_db")).get_collection(
        "hastalik_bilgi_tabani")
    r = col.get(include=["embeddings", "metadatas", "documents"])
    E = np.array(r["embeddings"])
    siniflar = [m["sinif"] for m in r["metadatas"]]
    sorgu_sinif = "Tomato___Late_blight"
    q = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2").encode([sorgu_sinif])[0]
    bulunan = col.query(query_embeddings=[q.tolist()], n_results=2, where={"sinif": sorgu_sinif})["ids"][0]

    ort = E.mean(axis=0)
    _, _, vt = np.linalg.svd(E - ort, full_matrices=False)
    P = (E - ort) @ vt[:2].T
    qp = (q - ort) @ vt[:2].T

    fig, ax = plt.subplots(figsize=(10, 6.6))
    kendi = np.array([s == sorgu_sinif for s in siniflar])
    ax.scatter(P[~kendi, 0], P[~kendi, 1], s=26, c=GRI_NOKTA, ec=YUZEY, lw=1.2, label="Diğer 37 hastalığın parçaları")
    ax.scatter(P[kendi, 0], P[kendi, 1], s=44, c=MAVI, ec=YUZEY, lw=1.2,
               label="Geç yanıklık dosyasının parçaları (sınıf filtresinin izin verdikleri)")
    idx = [r["ids"].index(i) for i in bulunan]
    ax.scatter(P[idx, 0], P[idx, 1], s=150, facecolors="none", ec=TURUNCU, lw=2.2,
               label="Sorguya en yakın 2 parça (Claude'a giden)")
    ax.scatter([qp[0]], [qp[1]], s=150, marker="*", c=TURUNCU, ec=YUZEY, lw=1, label="Sorgu: 'Tomato___Late_blight'")
    for i in idx:
        ax.plot([qp[0], P[i, 0]], [qp[1], P[i, 1]], color=TURUNCU, lw=1, ls="--")
        baslik = r["documents"][i].splitlines()[0].lstrip("# ").strip()[:32]
        ax.annotate(baslik, (P[i, 0], P[i, 1]), xytext=(8, -12), textcoords="offset points", fontsize=7.5,
                    color=METIN)
    # Parçalar bitkiye göre değil KONU BAŞLIĞINA göre kümelenir -> kümelerin adı başlık türüdür
    def tur(doc):
        ilk = doc.splitlines()[0].strip()
        return "Hastalık adı + etken" if ilk.startswith("# ") else ilk.lstrip("# ").split("(")[0].split("—")[0].strip()

    turler = [tur(d) for d in r["documents"]]
    for ad in ["Hastalık adı + etken", "Belirtiler", "Kaynak notu", "Uygun koşullar", "Kültürel/biyolojik önlemler",
               "Ne zaman endişelenmeli"]:
        m = np.array([t.startswith(ad[:12]) for t in turler])
        if m.sum() >= 5:
            c = P[m].mean(axis=0)
            ax.text(c[0], c[1] + 0.13, f"{ad} ({m.sum()})", fontsize=8.5, fontweight="bold", color=IKINCIL,
                    ha="center", bbox=dict(boxstyle="round,pad=0.2", fc=YUZEY, ec="none", alpha=0.85))
    ax.set_title("Chroma'daki 197 parçanın haritası (384 sayı → 2 boyuta indirildi, PCA)", loc="left",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("1. ana bileşen"), ax.set_ylabel("2. ana bileşen")
    ax.grid(color="#e6e8ec", lw=0.6), ax.set_axisbelow(True)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.text(0, -0.13, "Her nokta bir metin parçası. Parçalar bitkiye göre değil, konu başlığına göre kümeleniyor "
                      "(bütün 'Belirtiler' parçaları bir arada). Bu yüzden\naramada sınıf filtresi şart: yalnız mavi "
                      "noktalar arasından en yakın 2'si seçilir. Harita 384 boyutun 2'sini gösterir; asıl arama 384 "
                      "boyutun tamamında.", transform=ax.transAxes, fontsize=8, color=IKINCIL, va="top")
    fig.savefig(os.path.join(CIKTI, "vektor_haritasi.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 4) n8n akışı (canlı akış dosyasından), numaralı
N8N_SIRA = [  # kitapçıktaki tabloyla aynı sıra/numara
    "Telegram Trigger", "Buton mu?", "Fotoğraf var mı?",
    "Telegram - İnceleniyor", "Fotoğrafı İndir", "HTTP Request - Predict CNN", "Telegram - Ön Tespit",
    "HTTP Request - RAG Context", "Basic LLM Chain", "Anthropic Chat Model", "Rapor JSON'unu Ayrıştır",
    "Telegram - Durum: Hazır", "Telegram - Cevap Gönder", "Google Sheets - Kaydet", "HTTP Request - Raporu Kaydet",
    "Telegram - PDF Sorusu", "Güven < %70 mi?", "Telegram - Uzmana Bildir", "Hastalık var mı?", "Bekle (takip)",
    "Telegram - Takip Hatırlatması",
    "Selamlaşma mı?", "Telegram - Tanıtım", "HTTP Request - Son Fotoğraf", "Bitki düzeltmesi mi?",
    "Yeniden Değerlendirme Hazırla", "Basic LLM Chain - Sohbet", "Anthropic Chat Model - Sohbet",
    "Telegram - Sohbet Cevabı",
    "Telegram - Butonu Onayla", "PDF istendi mi?", "Telegram - PDF Hazırlanıyor", "HTTP Request - PDF Oluştur",
    "Telegram - PDF Gönder", "Telegram - PDF İstenmedi",
]
TUR_RENK = [  # (tür, renk, ad) — paletin sabit sırası
    ("telegram", MAVI, "Telegram"), ("if", TURUNCU, "Karar (IF)"), ("httpRequest", SU, "HTTP isteği (FastAPI)"),
    ("chainLlm", "#eda100", "Yapay zekâ (Claude)"), ("lmChatAnthropic", "#eda100", None),
    ("code", "#e87ba4", "Kod"), ("googleSheets", "#008300", "Google Sheets"), ("wait", "#4a3aa7", "Bekle"),
]


def n8n_akisi():
    with open(os.path.join(KOK, "n8n", "leadleaf_tam_akis.json"), encoding="utf-8") as f:
        w = json.load(f)
    dugum = {n["name"]: n for n in w["nodes"]}
    assert set(dugum) == set(N8N_SIRA), set(dugum) ^ set(N8N_SIRA)
    renk_of = {t: r for t, r, _ in TUR_RENK}

    def renk(n):
        t = n["type"].split(".")[-1]
        return renk_of.get("telegram" if t.startswith("telegram") else t, "#888888")

    fig, ax = plt.subplots(figsize=(13, 8.2))
    G, Y, OLCEK = 200, 96, 1.4  # n8n'deki dikey aralık kutulara dar; çizimde 1,4 kat açılır
    for n in w["nodes"]:
        n["position"] = [n["position"][0], n["position"][1] * OLCEK]
    for a, v in w["connections"].items():
        for tur, dallar in v.items():
            for di, dal in enumerate(dallar):
                for h in dal:
                    x1, y1 = dugum[a]["position"]
                    x2, y2 = dugum[h["node"]]["position"]
                    alt = tur != "main"
                    renk_ok = "#9aa0ab" if alt else ("#6b7080" if len(dallar) < 2 else (YESIL_OK if di == 0 else KIRMIZI_OK))
                    if alt:  # model alt düğümü: altından üstteki zincire
                        a_, b_ = (x1, -(y1 - Y / 2)), (x2, -(y2 + Y / 2))
                    else:
                        a_, b_ = (x1 + G / 2, -y1), (x2 - G / 2, -y2)
                    ax.add_patch(FancyArrowPatch(a_, b_, arrowstyle="-|>", mutation_scale=9, lw=1.1, color=renk_ok,
                                                 ls="--" if alt else "-", connectionstyle="arc3,rad=0.0",
                                                 shrinkA=1, shrinkB=1))
    for i, ad in enumerate(N8N_SIRA, 1):
        n = dugum[ad]
        x, y = n["position"]
        r = renk(n)
        ax.add_patch(FancyBboxPatch((x - G / 2, -y - Y / 2), G, Y, boxstyle="round,pad=4,rounding_size=14",
                                    fc=r + "22", ec=r, lw=1.3))
        ax.text(x - G / 2 + 12, -y + 27, str(i), fontsize=9.5, fontweight="bold", va="center", color=METIN)
        kisa = ad.replace("Telegram - ", "").replace("HTTP Request - ", "").replace("Anthropic Chat Model", "Claude modeli")
        ax.text(x, -y - 12, "\n".join(textwrap.wrap(kisa, 15)[:2]), fontsize=5.7, ha="center", va="center")
    xs = [n["position"][0] for n in w["nodes"]]
    ys = [-n["position"][1] for n in w["nodes"]]
    ax.set_xlim(min(xs) - 150, max(xs) + 150), ax.set_ylim(min(ys) - 300, max(ys) + 150)
    ax.set_aspect("equal"), ax.axis("off")
    ax.set_title("n8n akışı — LeadLeaf AI Bitki Hastalığı Bot (tam v2), numaralar tablodaki sırayla aynı",
                 loc="left", fontsize=12, fontweight="bold")
    lx, ly = min(xs) - 120, min(ys) - 150
    ogeler = [(r, a) for _, r, a in TUR_RENK if a] + [(YESIL_OK, "IF: evet (üst çıkış)"), (KIRMIZI_OK, "IF: hayır (alt çıkış)")]
    for k, (r, a) in enumerate(ogeler):
        xx = lx + (k % 5) * 520
        yy = ly - (k // 5) * 70
        if a.startswith("IF"):
            ax.plot([xx, xx + 60], [yy, yy], color=r, lw=1.6)
        else:
            ax.add_patch(FancyBboxPatch((xx, yy - 18), 60, 36, boxstyle="round,pad=2", fc=r + "22", ec=r))
        ax.text(xx + 80, yy, a, fontsize=7.5, va="center")
    fig.savefig(os.path.join(CIKTI, "n8n_akisi.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


YESIL_OK, KIRMIZI_OK = "#008300", "#e34948"

if __name__ == "__main__":
    mimari()
    rag_akisi()
    vektor_haritasi()
    n8n_akisi()
    print("Kaydedildi:", CIKTI, sorted(os.listdir(CIKTI)))
