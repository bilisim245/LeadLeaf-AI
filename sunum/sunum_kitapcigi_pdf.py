"""
Sunum kitapçığı: tek yerden çalışmak için panonun (ui/sunum.py) sayfa sırasıyla HER ŞEY.
Her pano sayfası için: senaryo (ne zaman, tıkla, göster, söyle), ekrandaki öğeler, grafikler (ekran
görüntüsüyle), ekranda gösterilen kod ve satır satır açıklaması, derinlemesine anlatım, kararlar ve
jüri soruları.

Metinler KOPYALANMAZ, kaynak dosyalardan okunur (bir yerde düzeltilen her şey burada da güncel olur):
  sunum/streamlit_senaryosu_pdf.py, pano_rehberi_pdf.py, grafik_rehberi_pdf.py, juri_sorulari_pdf.py,
  karar_defteri_pdf.py  -> listeler `ast` ile ÇALIŞTIRILMADAN okunur (o betikler içe aktarılınca PDF üretir)
  report/calisma_rehberi.md, report/kod_notlarim.md -> markdown olarak okunur
  Koddaki parçalar -> projedeki gerçek dosyalardan (ui/ortak.py kod_parcasi ile aynı mantık)

Çalıştırma:  .venv\\Scripts\\python sunum\\sunum_kitapcigi_pdf.py
Çıktı:       sunum/sunum_kitapcigi.pdf
"""
from __future__ import annotations

import ast
import glob
import os
import re

import numpy as np
from fpdf import FPDF
from PIL import Image

KLASOR = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(KLASOR)
EKRAN = os.path.join(KLASOR, "ekran_goruntuleri")
CIKTI = os.path.join(KLASOR, "sunum_kitapcigi.pdf")
LACIVERT, GRI, METIN, YESIL, TURUNCU = (15, 35, 71), (90, 100, 120), (30, 30, 30), (47, 125, 74), (170, 95, 20)
ACIK_FON, KOD_FON = (238, 242, 248), (244, 245, 247)


def sabitler(yol: str) -> dict:
    """Bir Python dosyasındaki düz sabitleri (liste, demet, metin) çalıştırmadan okur."""
    with open(os.path.join(KOK, yol), encoding="utf-8") as f:
        agac = ast.parse(f.read())
    sonuc = {}
    for d in agac.body:
        if isinstance(d, ast.Assign) and isinstance(d.targets[0], ast.Name):
            try:
                sonuc[d.targets[0].id] = ast.literal_eval(d.value)
            except ValueError:
                pass
    return sonuc


def kod_parcasi(dosya: str, baslangic: str, bitis: str | None = None) -> str:
    """ui/ortak.py'deki kod_parcasi ile aynı: panoda gösterilen kodun aynısı."""
    with open(os.path.join(KOK, dosya), encoding="utf-8") as f:
        satirlar = f.read().splitlines()
    i = next(k for k, s in enumerate(satirlar) if s.strip().startswith(baslangic))
    j = len(satirlar)
    if bitis:
        j = next((k for k in range(i + 1, len(satirlar)) if satirlar[k].strip().startswith(bitis)), j)
    return "\n".join(satirlar[i:j]).rstrip()


def md_bolumleri(yol: str, seviye: str = "## ") -> dict:
    """Markdown dosyasını başlıklara böler. Anahtar: '## 2b) ...' -> '2b'; diğer başlıklar -> başlığın kendisi."""
    with open(os.path.join(KOK, yol), encoding="utf-8") as f:
        metin = f.read()
    bolumler, anahtar, baslik, satirlar = {}, None, None, []
    for s in metin.splitlines():
        if s.startswith(seviye):
            if anahtar:
                bolumler[anahtar] = (baslik, "\n".join(satirlar).strip())
            baslik = s[len(seviye):].strip()
            m = re.match(r"(\w+)\)", baslik)
            anahtar, satirlar = (m.group(1) if m else baslik), []
        elif anahtar:
            satirlar.append(s)
    if anahtar:
        bolumler[anahtar] = (baslik, "\n".join(satirlar).strip())
    return bolumler


SENARYO = sabitler("sunum/streamlit_senaryosu_pdf.py")
PANO = sabitler("sunum/pano_rehberi_pdf.py")
GRAFIK = sabitler("sunum/grafik_rehberi_pdf.py")
JURI = sabitler("sunum/juri_sorulari_pdf.py")
KARAR = sabitler("sunum/karar_defteri_pdf.py")
SAYFALAR = sabitler("ui/ortak.py")["SAYFALAR"]  # (dosya, başlık, ikon) — panonun sunum sırası
REHBER = md_bolumleri("report/calisma_rehberi.md")

# Sayfa anahtarı: "sayfalar/04_cnn.py" -> "04", "app.py" -> "app"
def anahtar(dosya: str) -> str:
    ad = os.path.basename(dosya)
    return ad[:2] if ad[:2].isdigit() else "app"


# Senaryo adımındaki sayfa adı -> pano sayfası
SENARYO_SAYFA = {"Proje Özeti": "01", "Veri Seti": "02", "Keşifsel Veri Analizi": "03",
                 "CNN ve Transfer Learning": "04", "Model Karşılaştırma": "05", "Fine-Tuning": "06",
                 "Test Sonuçları": "07", "Model Nereye Bakıyor?": "08", "RAG ve Rapor": "09",
                 "Canlı Demo": "app", "Telegram (telefon)": "app", "Sınırlılıklar ve Sonraki Adım": "10"}
# Jüri soruları ve karar defteri bölümleri (sıra numarası) -> pano sayfası
JURI_SAYFA = {0: "01", 1: "03", 2: "05", 3: "07", 4: "09", 5: "app", 6: "10"}
KARAR_SAYFA = {0: "02", 1: "04", 2: "06", 3: "07", 4: "09", 5: "app"}
# Çalışma rehberindeki sorular -> pano sayfası (19 = değerlendirme, kitapçığın sonunda)
REHBER_SAYFA = {"01": ["14"], "02": ["10"], "03": ["2"], "04": ["3", "4", "5"], "05": ["8"],
                "06": ["2b", "18"], "07": ["15", "6"], "08": [], "09": ["1", "7", "12", "13", "20"],
                "app": ["17", "9", "11"], "10": ["16"]}

# Ekranda gösterilen kodlar: (başlık, dosya, başlangıç, bitiş, [(kod parçası, sade açıklama)])
KODLAR = {
    "04": [("Modelin kurulduğu kod (panoda 'Projedeki model' sekmesi)",
            "notebooks/03_efficientnetb0_38_sinif.py", "inputs = keras.Input", "def parametre_ozeti", [
        ("inputs = keras.Input((IMG, IMG, 3))",
         "Modelin girişi: 224×224 piksellik, 3 renk kanallı (kırmızı, yeşil, mavi) bir fotoğraf."),
        ("x = yeni_augment_katmani()(inputs)",
         "Veri artırma (augmentation): eğitim sırasında her fotoğraf rastgele yatay çevrilir, ±28,8 dereceye "
         "kadar döndürülür (RandomRotation(0.08) = tam turun ±%8'i), %10'a kadar yakınlaştırılır/uzaklaştırılır. "
         "Model aynı fotoğrafı her seferinde biraz farklı görür, ezberlemesi zorlaşır. Tahmin yaparken bu katman çalışmaz."),
        ("x = efficientnet.preprocess_input(x)",
         "Piksel değerleri EfficientNet'in ImageNet'te alıştığı biçime getirilir. Hazır modele, eğitildiği "
         "biçimde veri vermek gerekir."),
        ('base = keras.applications.EfficientNetB0(..., include_top=False, weights="imagenet")',
         "Transfer learning: ImageNet'te eğitilmiş hazır model (gövde) yüklenir. include_top=False: ImageNet'in "
         "1000 sınıflık kafası alınmaz, yerine bizim 38 sınıflık kafamız takılacak."),
        ("base.trainable = False",
         "Gövde donduruluyor: ilk aşamada ağırlıkları değişmez, sadece yeni kafa öğrenir (kafa rastgele "
         "başladığı için önce onun bir şeyler öğrenmesi gerekir)."),
        ("x = base(x, training=False)",
         "Gövde 'tahmin modunda' çalıştırılır. Bunun asıl etkisi BatchNorm katmanlarında: ImageNet'ten gelen "
         "ortalama/varyans istatistikleri korunur, ince ayarda da bozulmaz."),
        ("x = layers.GlobalAveragePooling2D()(x)",
         "Gövdenin çıktısı 7×7×1280'lik bir özellik haritasıdır (1280 filtre, her biri 7×7). Her filtrenin 49 "
         "değerinin ortalaması alınır → 1280 sayılık tek bir vektör: 'bu yaprakta hangi özellikler ne kadar var'."),
        ("x = layers.Dropout(0.2)(x)",
         "Eğitimde bu 1280 sayının rastgele %20'si sıfırlanır. Model tek bir özelliğe aşırı güvenmez, ezber azalır. "
         "Tahminde kapalıdır."),
        ('outputs = layers.Dense(len(class_names), activation="softmax")(x)',
         "Kafa: 38 çıkışlı katman. Softmax, çıktıları toplamı %100 olan olasılıklara çevirir. En büyük olasılık "
         "= tahmin, değeri = güven (%70 altı → uzmana yönlendir)."),
        ("model = keras.Model(inputs, outputs)", "Giriş ile çıkış arasındaki bütün katmanlar tek bir model olarak birleştirilir."),
    ])],
    "06": [("Kademeli açma döngüsü (panoda sayfanın altındaki kod)",
            "notebooks/03_efficientnetb0_38_sinif.py", "for i, (oran, ust_sinir_epoch)", "# %% 8)", [
        ("for i, (oran, ust_sinir_epoch) in enumerate(zip(UNFREEZE_ASAMALARI, ASAMA_EPOCH_SAYILARI), start=1):",
         "3 aşamalık döngü. UNFREEZE_ASAMALARI = [0.15, 0.30, 0.40] (gövdenin açılacak kısmı), "
         "ASAMA_EPOCH_SAYILARI = [6, 6, 8] (her aşamada en fazla kaç epoch)."),
        ("base.trainable = True\nn_freeze = int(len(base.layers) * (1 - oran))\nfor layer in base.layers[:n_freeze]: layer.trainable = False",
         "Önce gövdenin tamamı açılır, sonra ilk (1 − oran) kısmı yeniden dondurulur. Sonuç: sadece SON %15, "
         "sonra %30, sonra %40 eğitilir. İlk katmanlar (kenar, renk bulan evrensel filtreler) hiç değişmez."),
        ("print(...)", "Kaç katmanın açık olduğunu ekrana yazar (Colab çıktısında görülür)."),
        ('model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", ...)',
         "Donuk/açık değişikliğinin geçerli olması için model yeniden derlenir. Öğrenme hızı 1e-5: kafa "
         "eğitimindekinin (1e-3) yüzde biri → ince ayar. Kayıp fonksiyonu: sınıf numarasıyla çok sınıflı hata. "
         "(Yeniden derleme Adam'ı sıfırdan başlatır; 8. epoch'taki düşüşün bir nedeni budur.)"),
        ("parametre_ozeti(model, ...)", "Eğitilebilir ve donuk parametre sayısını yazar (Fine-Tuning sayfasındaki tablo)."),
        ('ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=REDUCE_LR_PATIENCE, min_lr=1e-7)',
         "Doğrulama kaybı 2 epoch (REDUCE_LR_PATIENCE = 2) düzelmezse öğrenme hızı yarıya iner; 1e-7'nin altına inmez. "
         "Takılınca daha küçük adımlarla ilerlemek için."),
        ('EarlyStopping(monitor="val_accuracy", mode="max", patience=FINE_TUNE_PATIENCE, restore_best_weights=True)',
         "Doğrulama doğruluğu 5 epoch (FINE_TUNE_PATIENCE = 5) artmazsa aşama durur ve model EN İYİ epoch'un "
         "ağırlıklarına geri döner. 38 sınıflı eğitimde hiçbir aşama erken durmadı."),
        ("hedef_epoch = sonraki_baslangic_epoch + ust_sinir_epoch\nh = model.fit(..., initial_epoch=..., epochs=hedef_epoch, ...)",
         "Epoch numaraları aşamalar boyunca kesintisiz devam eder (aşama 1: 8–13, aşama 2: 14–19, aşama 3: 20–27). "
         "Öğrenme eğrisinin tek parça olmasının nedeni bu."),
        ("gecmisler.append(h)\nsonraki_baslangic_epoch = h.epoch[-1] + 1",
         "Her aşamanın doğruluk/kayıp geçmişi saklanır (öğrenme eğrisi bunlardan çizilir); sonraki aşama kaldığı "
         "epoch'tan başlar."),
    ])],
    "09": [("Bilgi tabanında arama fonksiyonu (panoda 'Kod: arama fonksiyonu')",
            "agent/rag.py", "def retrieve_context", 'if __name__ == "__main__"', [
        ('def retrieve_context(hastalik: str, ek_sorgu: str = "", k: int = 2) -> str:',
         "Girdi: CNN'in bulduğu sınıf adı (ör. Tomato___Late_blight). Çıktı: o hastalık hakkında en ilgili "
         "k = 2 metin parçası. Bu metin Claude'un önüne konur."),
        ('if hastalik.endswith("___Tanimsiz") or hastalik == "Desteklenmeyen_bitki": return ""',
         "26 Eylül'de eklendi: 'tanımsız belirti' bir hastalık değil. Arama yapılırsa alakasız parçalar "
         "(ör. 'sağlıklı soya') geliyor ve Claude 'yaprak sağlıklı olabilir' diye yanıltıyordu."),
        ("if not _yukle(): return \"\"",
         "Chroma veritabanı ve embedding modeli bir kez yüklenir. Yüklenemezse boş metin döner: sistem çökmez, "
         "rapor RAG'siz yazılır (zarif bozulma)."),
        ('sorgu = f"{hastalik} {ek_sorgu}".strip()\nembedding = _model.encode([sorgu]).tolist()',
         "Sorgu metni, çok dilli embedding modeliyle (paraphrase-multilingual-MiniLM-L12-v2) 384 sayılık bir "
         "vektöre çevrilir. Anlamca yakın metinlerin vektörleri de birbirine yakındır."),
        ('sonuc = _collection.query(query_embeddings=embedding, n_results=k, where={"sinif": hastalik})',
         "Chroma'da arama: SADECE o hastalığın dosyasından gelen parçalar arasında (where filtresi), sorgu "
         "vektörüne en yakın 2 parça. Filtre, başka bir hastalığın bilgisinin karışmasını engeller."),
        ("if not parcalar: ... _collection.query(query_embeddings=embedding, n_results=k)",
         "Filtreyle bir şey bulunamazsa (ör. sınıf adı bilgi tabanında yoksa) filtresiz, tüm bilgi tabanında aranır."),
        ('return "\\n\\n---\\n\\n".join(parcalar)', "Bulunan parçalar aralarına ayraç konarak tek metin hâlinde döndürülür."),
    ])],
    "app": [("'Bitki doğru mu?' sorusu (ui/app.py, 26 Eylül)",
             "ui/app.py", "# Bitki kullanıcıya SONUÇTAN SONRA sorulur", "if tanimsiz:", [
        ('tahmin_bitki = BITKI_TR.get(cnn["hastalik"].split("___")[0])',
         "Sınıf adının '___' öncesi bitkidir: Corn_(maize)___Common_rust_ → Corn_(maize) → 'Mısır'."),
        ('if analiz["bitki"]: st.caption(...)',
         "Kullanıcı bitkiyi düzelttiyse soru tekrar sorulmaz; 'bu bitkiyle yeniden değerlendirildi' notu gösterilir."),
        ('elif analiz["onay"]: st.caption(...)', "'Evet' denmişse 'bitki kullanıcı tarafından doğrulandı' notu."),
        ("with st.container(border=True): ... st.button(f\"Evet, {tahmin_bitki}\") ... analiz[\"onay\"] = True; st.rerun()",
         "Soru kutusu. Streamlit'te her butona basış sayfanın kodunu baştan çalıştırır (st.rerun). Analiz sonucu "
         "st.session_state['analiz'] içinde (oturum hafızası) saklandığı için kaybolmaz, model tekrar çalışmaz."),
        ('dogru_bitki = e2.selectbox("Hayır, bu bir:", [b for b in BITKILER if b != tahmin_bitki], index=None)',
         "Modelin tahmin ettiği bitki dışındaki 13 bitki listelenir; başlangıçta hiçbiri seçili değildir."),
        ('st.session_state["analiz"] = {"foto": ..., "dosya": ..., "bitki": dogru_bitki, "onay": True}; st.rerun()',
         "Kayıt, sonuç olmadan (cnn yok) yeniden kurulur. Sayfa yeniden çalışınca aynı fotoğraf bu kez "
         "bitki = 'Domates' gibi bir bilgiyle /predict'e gönderilir; model sadece o bitkinin sınıfları arasından seçer."),
    ])],
}

# --- Yazı tipi dışı karakterler (emoji, kutu çizgileri) -------------------------------------------
from fontTools.ttLib import TTFont  # noqa: E402

_CMAP = TTFont(r"C:\Windows\Fonts\arial.ttf").getBestCmap()
_YEDEK = {"─": "-", "│": "|", "├": "|", "└": "`", "┌": "+", "►": ">", "▶": ">", "▼": "v", "✅": "[+]",
          "⚠": "(!)", "❌": "[x]", "💬": "", "🔁": "", "✓": "+", "✗": "x"}


def temiz(metin: str) -> str:
    return "".join(ch if ord(ch) in _CMAP or ch in "\n\t" else _YEDEK.get(ch, "") for ch in str(metin))


def sade(metin: str) -> str:
    """Markdown işaretlerini (**, `) kaldırır."""
    return temiz(metin.replace("**", "").replace("*", "").replace("`", ""))


class Belge(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Arial", "", 8)
        self.set_text_color(*GRI)
        self.cell(0, 6, f"LeadLeaf AI · {self.altbilgi} · {self.page_no()}", align="C")


def yeni_belge(altbilgi: str) -> Belge:
    """Yeni bir PDF belgesi. Aşağıdaki yardımcılar (yaz, markdown, ...) modül düzeyindeki `pdf`'i kullanır;
    ikinci belge (soru-cevap) için `pdf` bununla yeniden atanır."""
    b = Belge()
    b.altbilgi = altbilgi
    b.add_font("Arial", "", r"C:\Windows\Fonts\arial.ttf")
    b.add_font("Arial", "B", r"C:\Windows\Fonts\arialbd.ttf")
    b.set_margins(16, 14, 16)
    b.set_auto_page_break(True, margin=16)
    return b


pdf = yeni_belge("Sunum kitapçığı")
SOL = 16


def yaz(metin, boyut=10, kalin=False, renk=METIN, ara=5.2, fon=None):
    pdf.set_font("Arial", "B" if kalin else "", boyut)
    pdf.set_text_color(*renk)
    if fon:
        pdf.set_fill_color(*fon)
    pdf.multi_cell(0, ara, temiz(metin), fill=bool(fon), new_x="LMARGIN", new_y="NEXT")


def kalinli(metin, boyut=10, renk=METIN, ara=5.2, girinti=0.0):
    """'**kalın**' işaretli bir satırı kalın/normal parçalar hâlinde, satır sonunda kaydırarak yazar."""
    pdf.set_left_margin(SOL + girinti)
    pdf.set_x(SOL + girinti)
    pdf.set_text_color(*renk)
    for i, parca in enumerate(temiz(metin.replace("`", "")).split("**")):
        if parca:
            pdf.set_font("Arial", "B" if i % 2 else "", boyut)
            pdf.write(ara, parca)
    pdf.ln(ara)
    pdf.set_left_margin(SOL)


def ara_baslik(metin, renk=LACIVERT, boyut=12.5):
    if pdf.get_y() > 250:
        pdf.add_page()
    pdf.ln(2)
    yaz(metin, boyut, True, renk, 7)
    pdf.set_draw_color(*renk)
    pdf.line(SOL, pdf.get_y(), 210 - SOL, pdf.get_y())
    pdf.ln(2)


def kod_kutusu(kod, boyut=8.2):
    pdf.set_font("Arial", "", boyut)
    pdf.set_text_color(40, 45, 60)
    pdf.set_fill_color(*KOD_FON)
    # Baştaki boşluklar (girinti) kaybolmasın diye bölünmez boşluğa çevrilir
    satirlar = [re.sub(r"^ +", lambda m: "\u00a0" * len(m.group()), s) for s in temiz(kod).splitlines()]
    pdf.multi_cell(0, 4.2, "\n".join(satirlar) or " ", fill=True, new_x="LMARGIN", new_y="NEXT")


def tablo(satirlar):
    satirlar = [[sade(h.replace("\x00", "|")) for h in r] for r in satirlar]
    n = max(len(r) for r in satirlar)
    satirlar = [r + [""] * (n - len(r)) for r in satirlar]
    pdf.set_font("Arial", "", 8.6)
    pdf.set_text_color(*METIN)
    pdf.set_draw_color(200, 205, 215)
    # Sütun genişliği içeriğe göre: en uzun hücrenin uzunluğu (kısa sütunlar dar, metin sütunları geniş)
    genislik = [min(max(max(len(r[k]) for r in satirlar), 10), 60) for k in range(n)]
    with pdf.table(text_align="LEFT", line_height=4.4, padding=1.2, col_widths=genislik,
                   headings_style=_BASLIK_STILI, first_row_as_headings=True) as t:
        for r in satirlar:
            satir = t.row()
            for h in r:
                satir.cell(h)
    pdf.ln(2)


from fpdf.fonts import FontFace  # noqa: E402

_BASLIK_STILI = FontFace(emphasis="BOLD", color=LACIVERT, fill_color=ACIK_FON)


_OZEL = re.compile(r"\s*([-*] |\d+\. |> |#|\||```)")


def _paragraflar(satirlar: list[str]) -> list[str]:
    """Kaynak dosyada elle kırılmış satırları paragrafa birleştirir (kod blokları ve tablolar hariç).
    Özel işaretle başlamayan satır, önceki metin/madde/alıntı satırının devamı sayılır."""
    cikti, kodda = [], False
    for s in satirlar:
        if s.startswith("```"):
            kodda = not kodda
            cikti.append(s)
            continue
        onceki = cikti[-1] if cikti else ""
        devam = (not kodda and s.strip() and not _OZEL.match(s) and onceki.strip()
                 and not onceki.startswith(("```", "|", "#")) and not onceki.strip() == "---")
        if devam:
            cikti[-1] = onceki.rstrip() + " " + s.strip()
        else:
            cikti.append(s)
    return cikti


def markdown(metin: str):
    """Çalışma rehberi / kod notlarındaki markdown'u sade biçimde basar (başlık, madde, tablo, kod, alıntı)."""
    satirlar = _paragraflar(metin.replace("\\|", "\x00").splitlines())
    i = 0
    while i < len(satirlar):
        s = satirlar[i]
        if s.startswith("```"):
            j = i + 1
            while j < len(satirlar) and not satirlar[j].startswith("```"):
                j += 1
            kod_kutusu("\n".join(satirlar[i + 1:j]).replace("\x00", "|"))
            pdf.ln(1.5)
            i = j + 1
            continue
        if s.startswith("|"):
            tablo_satir = []
            while i < len(satirlar) and satirlar[i].startswith("|"):
                hucreler = [h.strip() for h in satirlar[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", h) for h in hucreler):
                    tablo_satir.append(hucreler)
                i += 1
            tablo(tablo_satir)
            continue
        i += 1
        gorsel = re.match(r"!\[(.*?)\]\((.*?)\)", s.strip())
        if gorsel:  # ![açıklama](yol) -> tam genişlikte görsel + altında açıklama
            yol = os.path.join(KOK, gorsel.group(2))
            if os.path.exists(yol):
                ekran_goruntusu(yol, genislik=178)
                yaz(gorsel.group(1), 8.5, renk=GRI, ara=4.5)
                pdf.ln(1.5)
        elif not s.strip() or s.strip() == "---":
            pdf.ln(1.5)
        elif s.startswith("# "):
            ara_baslik(sade(s[2:]), boyut=14)
        elif s.startswith("## ") or s.startswith("### "):
            pdf.ln(1)
            yaz(sade(s.lstrip("#").strip()), 11, True, LACIVERT, 6)
        elif s.lstrip().startswith("> "):
            kalinli(s.lstrip()[2:].replace("\x00", "|"), 9.6, GRI, 5, girinti=4)
        elif re.match(r"\s*[-*] ", s):
            girinti = 3 + len(s) - len(s.lstrip())
            kalinli("• " + re.sub(r"^\s*[-*] ", "", s).replace("\x00", "|"), girinti=girinti)
        elif re.match(r"\s*\d+\. ", s):
            kalinli(s.strip().replace("\x00", "|"), girinti=2 + len(s) - len(s.lstrip()))
        else:
            kalinli(s.strip().replace("\x00", "|"))


def kirp(yol):
    """Ekran görüntüsünün altındaki boş beyaz alanı kırpar (grafik_rehberi_pdf.py ile aynı)."""
    im = Image.open(yol).convert("RGB")
    a = np.asarray(im).astype(int)
    dolu = np.where((255 - a).sum(axis=(1, 2)) > a.shape[1] * 6)[0]
    alt = dolu.max() + 20 if len(dolu) else a.shape[0]
    return im.crop((0, 0, im.width, min(alt, im.height)))


GECICI = os.path.join(KLASOR, "_gecici_kitapcik")
os.makedirs(GECICI, exist_ok=True)


def ekran_goruntusu(yol, genislik=150):
    im = kirp(yol)
    if im.width > 1400:
        im = im.resize((1400, int(im.height * 1400 / im.width)), Image.LANCZOS)
    gp = os.path.join(GECICI, os.path.basename(yol).replace(".png", ".jpg"))
    im.save(gp, "JPEG", quality=80, optimize=True)
    yukseklik = genislik * im.height / im.width
    if yukseklik > 250:
        genislik, yukseklik = genislik * 250 / yukseklik, 250
    if pdf.get_y() + yukseklik > 280:
        pdf.add_page()
    pdf.image(gp, x=(210 - genislik) / 2, w=genislik)
    pdf.ln(2)


def icindekiler(pdf_, outline):
    pdf_.set_x(SOL)
    pdf_.set_font("Arial", "B", 18)
    pdf_.set_text_color(*LACIVERT)
    pdf_.cell(0, 12, "İçindekiler", new_x="LMARGIN", new_y="NEXT")
    for bolum in outline:
        if bolum.level > 1:
            continue
        pdf_.set_font("Arial", "B" if bolum.level == 0 else "", 10.5 if bolum.level == 0 else 9.5)
        pdf_.set_text_color(*(LACIVERT if bolum.level == 0 else METIN))
        pdf_.set_x(SOL + 6 * bolum.level)
        link = pdf_.add_link(page=bolum.page_number)
        pdf_.cell(150 - 6 * bolum.level, 5.2, temiz(bolum.name), link=link)
        pdf_.cell(0, 5.2, str(bolum.page_number), align="R", link=link, new_x="LMARGIN", new_y="NEXT")


# =====================================================================================================
# Kapak
pdf.add_page()
pdf.ln(30)
yaz("LeadLeaf AI", 30, True, LACIVERT, 14)
yaz("Sunum Kitapçığı", 20, True, LACIVERT, 11)
pdf.ln(4)
yaz("Tek yerden çalışmak için: panonun sayfa sırasıyla her ekranda ne yapılacağı, ne söyleneceği, "
    "grafiklerin nasıl okunacağı, ekrandaki kodun satır satır anlamı, arkasındaki kararlar ve jüri soruları.",
    11, renk=GRI, ara=6)
pdf.ln(8)
yaz("Her pano sayfasında aynı düzen var:", 11, True, LACIVERT, 7)
for baslik, aciklama in [
    ("Sunumda", "Ne zaman, neye tıklanır, ekranda ne gösterilir, ne söylenir (senaryo)."),
    ("Ekranda neler var?", "Sayfadaki her öğe ve ne anlama geldiği."),
    ("Grafikler", "Ekran görüntüsü + ekranda ne var, nasıl okunur, ne anlama geliyor."),
    ("Koddaki karşılığı", "Sayfada gösterilen gerçek kod ve her satırın sade açıklaması."),
    ("Derinlemesine", "Çalışma rehberindeki ilgili sorular (neden, alternatif, sayılar)."),
    ("Kararlar", "Bu konudaki teknik kararlar: nedir, alternatifler, neden, bedeli."),
    ("Jüri sorarsa", "Olası sorular ve kısa cevaplar."),
]:
    kalinli(f"**{baslik}:** {aciklama}", girinti=3)
pdf.ln(6)
yaz("Bu kitapçık sunum/sunum_kitapcigi_pdf.py ile kaynak dosyalardan otomatik üretilir; bir kaynakta yapılan "
    "düzeltme burada da görünür.", 8.5, renk=GRI, ara=4.5)

pdf.add_page()
pdf.insert_toc_placeholder(icindekiler, pages=2)

# =====================================================================================================
# Sunumdan önce
pdf.add_page()
pdf.start_section("Sunumdan önce", 0)
yaz("Sunumdan önce", 20, True, LACIVERT, 11)
pdf.start_section("Hazırlık listesi", 1)
ara_baslik("Hazırlık listesi")
for n, h in enumerate(SENARYO["HAZIRLIK"], 1):
    kalinli(f"{n}. {h}", girinti=2)
ara_baslik("Servisleri başlatma (her biri ayrı PowerShell penceresinde, açık kalmalı)")
kod_kutusu(".venv\\Scripts\\python -m uvicorn inference.app:app --host 127.0.0.1 --port 8000\n"
           "ngrok http --url=enclose-afterglow-sappiness.ngrok-free.dev 5678\n"
           '$env:NODE_OPTIONS="--dns-result-order=ipv4first"; '
           '$env:WEBHOOK_URL="https://enclose-afterglow-sappiness.ngrok-free.dev/"; npx n8n start\n'
           ".venv\\Scripts\\python -m streamlit run ui/sunum.py")
yaz("Kontrol: http://127.0.0.1:8000/health → num_classes: 38 · pano: http://localhost:8501 · "
    "okul (FATİH) ağında ngrok/Telegram/Claude çalışmaz → telefon hotspot'u.", 9, renk=GRI, ara=4.8)
pdf.start_section("Büyük resim", 1)
ara_baslik("Büyük resim")
yaz(PANO["GIRIS"], 10)
pdf.ln(1)
ekran_goruntusu(os.path.join(KLASOR, "sistem_gorselleri", "mimari.png"), genislik=178)
yaz("Bileşenler ve veri depoları: kim kime bağlanıyor, veri nerede duruyor.", 8.5, renk=GRI, ara=4.5)
pdf.ln(1.5)
for p in PANO["BUYUK_RESIM"]:
    kalinli("• " + p, girinti=2)
pdf.start_section("Ezberlenecek sayılar", 1)
ara_baslik("Ezberlenecek sayılar")
tablo([["Konu", "Değer"]] + [list(s) for s in JURI["SAYILAR"]])
pdf.start_section("Önce temel kavramlar", 1)
ara_baslik("Önce temel kavramlar")
for baslik, metin in GRAFIK["TEMEL"]:
    yaz(baslik, 10.5, True, LACIVERT, 6)
    yaz(metin)
    pdf.ln(1.5)

# =====================================================================================================
# Sayfa sayfa
for sira, (dosya, baslik, _ikon) in enumerate(SAYFALAR, 1):
    k = anahtar(dosya)
    pdf.add_page()
    pdf.start_section(f"{sira}. {baslik}", 0)
    yaz(f"Pano sayfası {sira} / {len(SAYFALAR)}", 9, renk=GRI, ara=5)
    yaz(baslik, 20, True, LACIVERT, 11)
    pano_baslik, amac, ogeler = PANO["SAYFALAR"][sira - 1]
    yaz(amac, 10.5, renk=GRI, ara=5.6)

    # Sunumda (senaryo)
    adimlar = [a for a in SENARYO["ADIMLAR"] if SENARYO_SAYFA.get(a[1]) == k]
    if adimlar:
        pdf.start_section("Sunumda (senaryo)", 1)
        ara_baslik("Sunumda (senaryo)")
        for sure, sayfa, tikla, goster, soyle in adimlar:
            if pdf.get_y() > 245:
                pdf.add_page()
            yaz(f"{sure}  ·  {sayfa}", 10.5, True, LACIVERT, 6)
            kalinli(f"**Tıkla:** {tikla}", girinti=3)
            kalinli(f"**Göster:** {goster}", girinti=3)
            pdf.set_fill_color(*ACIK_FON)
            pdf.set_x(SOL + 3)
            pdf.set_font("Arial", "B", 9.8)
            pdf.set_text_color(*LACIVERT)
            pdf.multi_cell(0, 5.3, temiz("Söyle: " + soyle), fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2.5)

    # Ekranda neler var?
    pdf.start_section("Ekranda neler var?", 1)
    ara_baslik("Ekranda neler var?")
    for ne, gorunen, anlam in ogeler:
        if pdf.get_y() > 255:
            pdf.add_page()
        yaz(ne, 10.5, True, METIN, 6)
        kalinli(f"**Ne görünüyor:** {gorunen}", girinti=3)
        kalinli(f"**Ne anlama geliyor:** {anlam}", renk=YESIL, girinti=3)
        pdf.ln(1.5)

    # Grafikler
    grafikler = [b for b in GRAFIK["BOLUMLER"] if b[0][:2] == k]
    if grafikler:
        pdf.start_section("Grafikler", 1)
        ara_baslik("Grafikler")
        for onek, gbaslik, ne, nasil, anlam in grafikler:
            if pdf.get_y() > 200:
                pdf.add_page()
            yaz(gbaslik, 11.5, True, LACIVERT, 7)
            for yol in sorted(glob.glob(os.path.join(EKRAN, f"{onek}_*.png"))):
                ekran_goruntusu(yol)
            if pdf.get_y() > 235:
                pdf.add_page()
            kalinli(f"**Ekranda ne var?** {ne}")
            kalinli(f"**Nasıl okunur?** {nasil}")
            kalinli(f"**Ne anlama geliyor?** {anlam}", renk=YESIL)
            pdf.ln(3)

    # Koddaki karşılığı
    if k in KODLAR:
        pdf.start_section("Koddaki karşılığı", 1)
        ara_baslik("Koddaki karşılığı")
        for kbaslik, kdosya, bas, bit, aciklamalar in KODLAR[k]:
            yaz(kbaslik, 11, True, LACIVERT, 6.5)
            yaz(f"Dosya: {kdosya} (projedeki gerçek kod)", 8.5, renk=GRI, ara=4.5)
            pdf.ln(1)
            kod_kutusu(kod_parcasi(kdosya, bas, bit))
            pdf.ln(2)
            yaz("Satır satır:", 10.5, True, METIN, 6)
            for parca, aciklama in aciklamalar:
                if pdf.get_y() > 255:
                    pdf.add_page()
                kod_kutusu(parca, 8)
                kalinli(aciklama, girinti=3)
                pdf.ln(1.5)

    # Derinlemesine (çalışma rehberi)
    sorular = [REHBER[s] for s in REHBER_SAYFA.get(k, []) if s in REHBER]
    if sorular:
        pdf.start_section("Derinlemesine", 1)
        ara_baslik("Derinlemesine")
        for rbaslik, rmetin in sorular:
            if pdf.get_y() > 240:
                pdf.add_page()
            yaz(rbaslik, 11.5, True, LACIVERT, 6.5)
            markdown(rmetin)
            pdf.ln(2)

    # Kararlar
    kararlar = [b for n, b in enumerate(KARAR["BOLUMLER"]) if KARAR_SAYFA.get(n) == k]
    if kararlar:
        pdf.start_section("Kararlar", 1)
        ara_baslik("Kararlar")
        for _bolum, konular in kararlar:
            for konu, nedir, alternatifler, neden, bedel, juri in konular:
                if pdf.get_y() > 235:
                    pdf.add_page()
                yaz(konu, 11, True, LACIVERT, 6.5)
                kalinli(f"**Nedir?** {nedir}")
                yaz("Alternatifler neydi?", 10, True, METIN, 5.2)
                for a in alternatifler:
                    kalinli("• " + a, girinti=3)
                kalinli(f"**Neden bu seçildi?** {neden}", renk=YESIL)
                if bedel and bedel != "—":
                    kalinli(f"**Bedeli / sınırı:** {bedel}", renk=TURUNCU)
                yaz("Jüri sorarsa: " + juri, 9.5, True, LACIVERT, 5.4, fon=ACIK_FON)
                pdf.ln(3)

    # Jüri sorarsa
    juri_bolum = [b for n, b in enumerate(JURI["BOLUMLER"]) if JURI_SAYFA.get(n) == k]
    if juri_bolum:
        pdf.start_section("Jüri sorarsa", 1)
        ara_baslik("Jüri sorarsa")
        for _bolum, sorular_ in juri_bolum:
            for soru, cevap in sorular_:
                if pdf.get_y() > 255:
                    pdf.add_page()
                yaz("S: " + soru, 10, True, LACIVERT, 5.4)
                yaz("C: " + cevap, 10, False, METIN, 5.2)
                pdf.ln(2)

# =====================================================================================================
# Sonda
pdf.add_page()
pdf.start_section("Sunum ipuçları", 0)
yaz("Sunum ipuçları", 20, True, LACIVERT, 11)
for ip in SENARYO["IPUCLARI"]:
    kalinli("• " + ip, girinti=2)

if "19" in REHBER:
    pdf.add_page()
    pdf.start_section("Projenin değerlendirmesi", 0)
    yaz(REHBER["19"][0], 16, True, LACIVERT, 9)
    markdown(REHBER["19"][1])

pdf.add_page()
pdf.start_section("Sözlük", 0)
yaz("Sözlük", 20, True, LACIVERT, 11)
tablo([["Terim", "Anlamı"]] + [list(s) for s in JURI["SOZLUK"]])
if "Terimler sözlüğü (hızlı tekrar)" in REHBER:
    markdown(REHBER["Terimler sözlüğü (hızlı tekrar)"][1])

# Ek: 26 Eylül hata hikâyesi (kod notlarından)
with open(os.path.join(KOK, "report", "kod_notlarim.md"), encoding="utf-8") as f:
    notlar = f.read()
bas = notlar.find("# 26 Eylül 2026")
if bas >= 0:
    pdf.add_page()
    pdf.start_section("Ek: 26 Eylül — 'Mısır' hatası ve düzeltmesi", 0)
    yaz("Ek: 26 Eylül — 'Mısır' hatası ve düzeltmesi", 18, True, LACIVERT, 10)
    yaz("Gerçek bir Telegram testinde çıkan hatanın nasıl bulunduğu, neden olduğu ve nasıl düzeltildiği "
        "(report/kod_notlarim.md).", 10, renk=GRI)
    markdown(notlar[bas:].split("\n", 1)[1])

pdf.output(CIKTI)
print("Kaydedildi:", CIKTI, "·", pdf.page_no(), "sayfa")

# =====================================================================================================
# İkinci belge: kâğıttaki sorular ve cevapları (report/calisma_rehberi.md'nin tamamı, kendi sırasıyla)
SORU_CEVAP = os.path.join(KLASOR, "soru_cevap.pdf")
pdf = yeni_belge("Sorular ve cevaplar")
with open(os.path.join(KOK, "report", "calisma_rehberi.md"), encoding="utf-8") as f:
    giris = f.read().split("\n## ", 1)[0]
pdf.add_page()
pdf.ln(24)
yaz("LeadLeaf AI", 30, True, LACIVERT, 14)
yaz("Sorularım ve Cevapları", 20, True, LACIVERT, 11)
pdf.ln(4)
yaz("Elle yazılan 16 soru ve sonradan eklenen sorular: RAG, veri bölme, CNN, transfer learning, "
    "fine-tuning, ısı haritaları, n8n (her düğümüyle), projedeki bütün veritabanları, metrikler, "
    "'Mısır' hatası, proje değerlendirmesi. Her cevap: kısa cevap → sade açıklama → projede nerede.",
    11, renk=GRI, ara=6)
pdf.ln(6)
markdown(giris.split("\n", 1)[1] if giris.startswith("# ") else giris)
pdf.add_page()
pdf.insert_toc_placeholder(icindekiler, pages=1)
for anahtar_, (baslik_, metin_) in REHBER.items():
    pdf.add_page()
    no = re.match(r"(\w+)\)\s*(.*)", baslik_)
    # "4) ..." gibi eşsiz ')' yer imi (outline) başlığını bozuyor (fpdf2 ASCII başlıkta kaçışlamıyor)
    pdf.start_section(f"Soru {no.group(1)} — {no.group(2)}" if no else baslik_, 0)
    if no:
        yaz(f"Soru {no.group(1)}", 10, True, GRI, 5.5)
        yaz(no.group(2), 17, True, LACIVERT, 8.5)
    else:
        yaz(baslik_, 17, True, LACIVERT, 8.5)
    pdf.set_draw_color(*LACIVERT)
    pdf.line(SOL, pdf.get_y() + 1, 210 - SOL, pdf.get_y() + 1)
    pdf.ln(4)
    markdown(metin_)
pdf.output(SORU_CEVAP)
print("Kaydedildi:", SORU_CEVAP, "·", pdf.page_no(), "sayfa")

for f in glob.glob(os.path.join(GECICI, "*.jpg")):
    os.remove(f)
os.rmdir(GECICI)
