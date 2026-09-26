// n8n "Rapor JSON'unu Ayrıştır" Code düğümünün içeriği (Run Once for All Items).
// Claude'un JSON raporunu ayrıştırır ve Telegram'a gidecek biçimli mesajı (HTML) hazırlar.
// Telegram - Cevap Gönder düğümü: Text = {{ $json.telegram_mesaji }}, Parse Mode = HTML.
const raw = $input.first().json;
let rapor;
try {
  let text = (raw.text ?? raw.response?.text ?? '').trim();
  if (text.startsWith('```')) {
    text = text.replace(/^```(json)?\n?/, '').replace(/```$/, '').trim();
  }
  rapor = JSON.parse(text);
} catch (e) {
  rapor = {
    hastalik: "bilinmiyor",
    guven: 0,
    neden: "-",
    aciklama: "Rapor ayrıştırılamadı, ham yanıt: " + JSON.stringify(raw).slice(0, 500),
    onlem: ["-"],
    uzmana_yonlendir: true,
    uyari: "Teknik hata oluştu, sonuç güvenilir değil."
  };
}

// Claude'un metnindeki <, >, & karakterleri Telegram HTML biçimlendirmesini bozmasın
const esc = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const guven = Number(rapor.guven) || 0;
const guvenEtiketi = guven >= 90 ? 'yüksek' : guven >= 70 ? 'orta' : 'düşük';
const onlemler = (Array.isArray(rapor.onlem) ? rapor.onlem : [rapor.onlem])
  .filter(Boolean)
  .map((o) => '• ' + esc(o))
  .join('\n');

const satirlar = [
  '🌿 <b>LeadLeaf AI — Ön Değerlendirme Raporu</b>',
  '',
  `🔎 <b>Tespit:</b> ${esc(rapor.hastalik)}`,
  `📊 <b>Model güveni:</b> %${guven.toLocaleString('tr-TR')} (${guvenEtiketi})`,
];
if (rapor.uzmana_yonlendir) {
  satirlar.push('', '⚠️ <b>Bu sonuç kesin değil — bir ziraat mühendisine danışmanızı öneririz.</b>');
  // Bitki adı verilmediyse model 38 sınıfın hepsi arasında tahmin etti; adı yazılırsa aynı fotoğraf
  // sadece o bitkinin sınıflarıyla yeniden değerlendirilir (n8n sohbet dalı → /son-foto)
  const tahmin = $('HTTP Request - Predict CNN').first().json;
  if (!tahmin.bitki) {
    satirlar.push('💬 Bitki yanlış mı? Bitkinin adını yazın (ör. <i>domates</i>), aynı fotoğrafı ona göre yeniden değerlendireyim.');
  }
}
if (rapor.aciklama) satirlar.push('', `📝 <b>Değerlendirme</b>\n${esc(rapor.aciklama)}`);
if (rapor.neden && rapor.neden !== '-') satirlar.push('', `🦠 <b>Nedeni</b>\n${esc(rapor.neden)}`);
if (onlemler && onlemler !== '• -') satirlar.push('', `✅ <b>Ne yapmalı?</b>\n${onlemler}`);
// Alt bilgi: Claude'un kendi uyarısı varsa o, yoksa sabit uyarı (ikisi aynı şeyi tekrar etmesin)
satirlar.push(
  '',
  `<i>ℹ️ ${esc(rapor.uyari) || 'Bu rapor yapay zekâ destekli bir ön değerlendirmedir, kesin teşhis değildir.'}</i>`
);

return [{ json: { ...rapor, telegram_mesaji: satirlar.join('\n') } }];
