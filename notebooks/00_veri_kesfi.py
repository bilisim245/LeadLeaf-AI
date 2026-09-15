# =====================================================================
#  VERİ SETİ KEŞFİ (EDA) — "New Plant Diseases Dataset" (vipoooool)
#  Google Colab
# =====================================================================
#
#  AMAÇ: Modele geçmeden ÖNCE veri setini gerçekten incelemek —
#  varsayımla değil, sayılarla. Rapora/sunuma girecek 3 kritik çıktı:
#    1. Sınıf dengesizliği (bazı hastalıklar diğerlerinden çok daha az mı?)
#    2. Görsel kalitesi (lab arka planlı mı, gerçek tarla foto mu, boyutlar tutarlı mı?)
#    3. TRAIN/VALID SIZINTISI (data leakage) riski — bu veri seti "Augmented"
#       yani orijinal fotoğraflar döndürme/aynalama ile çoğaltılmış. Eğer
#       aynı orijinal fotoğrafın farklı augment'leri hem train'de hem valid'de
#       varsa, doğrulama doğruluğu OLDUĞUNDAN YÜKSEK görünür (literatürde
#       PlantVillage tabanlı veri setlerine yöneltilen bilinen bir eleştiri).
#       Bu script perceptual hash ile TRAIN/VALID arasında yakın-kopya arar.
#
#  NASIL ÇALIŞTIRILIR: 01_train_model_colab.py'deki "veri indirme" hücresini
#  önce çalıştır (aynı /content/data yolu kullanılıyor), sonra bunu çalıştır.
#
#  ÇIKTI (/content/eda_ciktilari):
#   - sinif_dagilimi.png         her sınıfta kaç görsel var (train + valid)
#   - ornek_gorseller_izgara.png her sınıftan örnek görseller
#   - boyut_dagilimi.png         görsel genişlik/yükseklik histogramı
#   - sizinti_raporu.txt         train/valid arasında bulunan yakın-kopya sayısı
#   - veri_ozeti.txt             tüm sayısal özet (rapora kopyala-yapıştır)
# =====================================================================

# %% 0) Kurulum + veri yolu (01_train_model_colab.py'deki indirme hücresi çalıştırılmış olmalı)
import os, json, random
os.system("pip -q install imagehash")

CANDIDATES = [
    "/content/data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)",
    "/content/data/New Plant Diseases Dataset(Augmented)",
]
DATA_ROOT = next((p for p in CANDIDATES if os.path.isdir(os.path.join(p, "train"))), None)
assert DATA_ROOT, "Veri seti bulunamadi — once 01_train_model_colab.py'deki indirme hucresini calistir."

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VALID_DIR = os.path.join(DATA_ROOT, "valid")
OUT = "/content/eda_ciktilari"
os.makedirs(OUT, exist_ok=True)

SELECTED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Septoria_leaf_spot",
]

# %% 1) Sınıf dağılımı — TÜM 38 sınıf + domates alt kümesi ayrı ayrı
import matplotlib.pyplot as plt

def sinif_sayilari(klasor):
    return {c: len(os.listdir(os.path.join(klasor, c)))
            for c in sorted(os.listdir(klasor)) if os.path.isdir(os.path.join(klasor, c))}

train_counts_all = sinif_sayilari(TRAIN_DIR)
valid_counts_all = sinif_sayilari(VALID_DIR)

toplam_train = sum(train_counts_all.values())
toplam_valid = sum(valid_counts_all.values())
print(f"TÜM VERİ SETİ — {len(train_counts_all)} sınıf")
print(f"  train: {toplam_train} görsel, valid: {toplam_valid} görsel")

# Domates alt kümesi
tomato_train = {k: v for k, v in train_counts_all.items() if k in SELECTED_CLASSES}
tomato_valid = {k: v for k, v in valid_counts_all.items() if k in SELECTED_CLASSES}
print(f"\nDOMATES ALT KÜMESİ (MVP) — {len(tomato_train)} sınıf")
for c in SELECTED_CLASSES:
    print(f"  {c:35s} train={tomato_train.get(c,0):5d}  valid={tomato_valid.get(c,0):5d}")

# Dengesizlik oranı (en büyük / en küçük sınıf)
if tomato_train:
    max_c, min_c = max(tomato_train.values()), min(tomato_train.values())
    print(f"\nDengesizlik oranı (domates alt kümesi, train): {max_c/min_c:.2f}x "
          f"(en çok: {max_c}, en az: {min_c})")

# Grafik: tüm 38 sınıf (domates olanlar vurgulu)
fig, ax = plt.subplots(figsize=(14, 10))
siniflar = list(train_counts_all.keys())
degerler = [train_counts_all[c] for c in siniflar]
renkler = ["#e74c3c" if c in SELECTED_CLASSES else "#3498db" for c in siniflar]
ax.barh(siniflar, degerler, color=renkler)
ax.set_xlabel("Train görsel sayısı")
ax.set_title("Sınıf dağılımı — tüm 38 sınıf (KIRMIZI = MVP'de kullanılan domates alt kümesi)")
plt.tight_layout(); plt.savefig(f"{OUT}/sinif_dagilimi.png", dpi=120); plt.show()

# %% 2) Örnek görseller — domates 5 sınıftan birer izgara
from PIL import Image

fig, axes = plt.subplots(1, len(SELECTED_CLASSES), figsize=(4 * len(SELECTED_CLASSES), 4))
for ax, cls in zip(axes, SELECTED_CLASSES):
    klasor = os.path.join(TRAIN_DIR, cls)
    dosya = random.choice(os.listdir(klasor))
    img = Image.open(os.path.join(klasor, dosya))
    ax.imshow(img)
    ax.set_title(f"{cls}\n{img.size[0]}x{img.size[1]}px", fontsize=9)
    ax.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/ornek_gorseller_izgara.png", dpi=120); plt.show()

# %% 3) Görsel boyutu dağılımı (örnekleme — 300 görsel, hepsini açmak yavaş)
genislikler, yukseklikler = [], []
ornek_havuzu = []
for cls in SELECTED_CLASSES:
    klasor = os.path.join(TRAIN_DIR, cls)
    dosyalar = os.listdir(klasor)
    ornek_havuzu += [os.path.join(klasor, f) for f in random.sample(dosyalar, min(60, len(dosyalar)))]

for yol in ornek_havuzu:
    with Image.open(yol) as im:
        genislikler.append(im.size[0]); yukseklikler.append(im.size[1])

print(f"\nÖrneklenen {len(ornek_havuzu)} görsel — boyut istatistiği:")
print(f"  genişlik: min={min(genislikler)} max={max(genislikler)} ortalama={sum(genislikler)/len(genislikler):.0f}")
print(f"  yükseklik: min={min(yukseklikler)} max={max(yukseklikler)} ortalama={sum(yukseklikler)/len(yukseklikler):.0f}")
tekboyut = len(set(zip(genislikler, yukseklikler))) == 1
print(f"  Tüm görseller aynı boyutta mı: {'EVET — ' + str(genislikler[0]) + 'x' + str(yukseklikler[0]) if tekboyut else 'HAYIR, karışık'}")

plt.figure(figsize=(7, 4))
plt.hist(genislikler, bins=20, alpha=0.6, label="genişlik")
plt.hist(yukseklikler, bins=20, alpha=0.6, label="yükseklik")
plt.legend(); plt.title("Görsel boyutu dağılımı (piksel, örneklem)")
plt.tight_layout(); plt.savefig(f"{OUT}/boyut_dagilimi.png", dpi=120); plt.show()

# %% 4) TRAIN/VALID SIZINTISI (data leakage) kontrolü — perceptual hash
# "Augmented" veri setinde ayni orijinal fotografin dondurulmus/aynalanmis
# kopyalari hem train'de hem valid'de olabilir. Bunu perceptual hash (pHash)
# ile yakalamaya calisiyoruz: hash'ler cok yakinsa (Hamming mesafesi kucuk),
# muhtemelen ayni fotografin varyasyonu.
import imagehash

def hashle(klasor, sinif, adet=150):
    dosyalar = os.listdir(os.path.join(klasor, sinif))
    secilen = random.sample(dosyalar, min(adet, len(dosyalar)))
    out = {}
    for f in secilen:
        try:
            with Image.open(os.path.join(klasor, sinif, f)) as im:
                out[f] = imagehash.phash(im)
        except Exception:
            pass
    return out

sizinti_rapor = []
for cls in SELECTED_CLASSES:
    train_hash = hashle(TRAIN_DIR, cls, adet=150)
    valid_hash = hashle(VALID_DIR, cls, adet=150)
    yakin_kopya = 0
    for vh in valid_hash.values():
        for th in train_hash.values():
            if (vh - th) <= 4:  # Hamming mesafesi esigi — kucuk = cok benzer
                yakin_kopya += 1
                break
    oran = yakin_kopya / len(valid_hash) if valid_hash else 0
    sizinti_rapor.append((cls, len(train_hash), len(valid_hash), yakin_kopya, oran))
    print(f"{cls:35s} örneklenen train={len(train_hash):3d} valid={len(valid_hash):3d} "
          f"-> valid'deki YAKIN KOPYA: {yakin_kopya} (%{oran*100:.1f})")

with open(f"{OUT}/sizinti_raporu.txt", "w", encoding="utf-8") as f:
    f.write("TRAIN/VALID SIZINTISI (data leakage) — orneklem bazli perceptual hash kontrolu\n")
    f.write("Not: Bu KESIN degil, orneklem uzerinden bir tahmindir. Oran yuksekse\n")
    f.write("(ör. >%10) dogrulama dogrulugu OLDUGUNDAN YUKSEK gorunuyor olabilir —\n")
    f.write("rapora/sunuma bu SINIRLILIK olarak eklenmeli.\n\n")
    for cls, ntr, nva, yk, oran in sizinti_rapor:
        f.write(f"{cls}: train_ornek={ntr} valid_ornek={nva} yakin_kopya={yk} oran=%{oran*100:.1f}\n")

# %% 5) Özet dosyası (rapora kopyala-yapıştır)
ozet = f"""VERİ SETİ ÖZETİ — New Plant Diseases Dataset (vipoooool, Kaggle)

TÜM VERİ SETİ: {len(train_counts_all)} sınıf, {toplam_train} train + {toplam_valid} valid görsel
DOMATES ALT KÜMESİ (MVP, {len(SELECTED_CLASSES)} sınıf):
""" + "\n".join(f"  - {c}: train={tomato_train.get(c,0)}, valid={tomato_valid.get(c,0)}"
                 for c in SELECTED_CLASSES) + f"""

Dengesizlik oranı (train, en çok/en az): {max(tomato_train.values())/min(tomato_train.values()):.2f}x
Görsel boyutu: {'sabit ' + str(genislikler[0]) + 'x' + str(yukseklikler[0]) + 'px' if tekboyut else 'değişken'}

SINIRLILIKLAR (rapora eklenecek):
1. Görseller laboratuvar/kontrollü arka planda çekilmiş (PlantVillage kökenli),
   gerçek tarla fotoğrafları DEĞİL — gerçek koşullarda başarım düşebilir.
2. Veri seti "Augmented" (döndürme/aynalama ile çoğaltılmış). Örneklem bazlı
   perceptual hash kontrolünde train/valid arasında yakın-kopya oranı sınıf
   bazında sizinti_raporu.txt'de listelendi — yüksekse doğrulama doğruluğu
   olduğundan iyimser görünüyor olabilir.
3. Sınıf dengesizliği {max(tomato_train.values())/min(tomato_train.values()):.1f}x — eğitimde
   class_weight veya daha fazla augmentation ile dengelenmesi düşünülebilir.
"""
with open(f"{OUT}/veri_ozeti.txt", "w", encoding="utf-8") as f:
    f.write(ozet)
print("\n" + ozet)

# %% 6) İndir
import shutil
from google.colab import files
shutil.make_archive("/content/eda_ciktilari", "zip", OUT)
files.download("/content/eda_ciktilari.zip")
print("BİTTİ — eda_ciktilari.zip indi. report/ klasörüne koy, rapora görselleri ve veri_ozeti.txt'yi işle.")
