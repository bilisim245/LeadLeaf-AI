"""n8n akışı — tıklanabilir şema. Bir düğüme tıklanınca ne yaptığı, girdisi, çıktısı ve ayarları görünür.

Veri tek kaynaktan gelir:
  şema, ayarlar, bağlantılar -> n8n/leadleaf_tam_akis.json (canlı akışın dışa aktarımı)
  "ne yapar" açıklamaları    -> report/calisma_rehberi.md, Soru 9'daki 35 satırlık düğüm tablosu
"""
import json
import os
import re
import textwrap

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ortak import KOK, bulgu, gezinme

st.title("n8n Akışı — Telegram botunun beyni")
st.write("Çiftçinin Telegram'dan gönderdiği her mesaj bu akışı bir kez çalıştırır. Kutulara **tıklayarak** "
         "her düğümün ne yaptığı görülebilir; üstten bir senaryo seçilirse mesajın izlediği yol adım adım "
         "ilerletilebilir.")

# --- Veri ------------------------------------------------------------------------------------------
with open(os.path.join(KOK, "n8n", "leadleaf_tam_akis.json"), encoding="utf-8") as f:
    AKIS = json.load(f)
DUGUM = {n["name"]: n for n in AKIS["nodes"]}

# Numara ve açıklama: çalışma rehberindeki tablo ("| 6 | HTTP Request - Predict CNN | ... |")
with open(os.path.join(KOK, "report", "calisma_rehberi.md"), encoding="utf-8") as f:
    TABLO = {m.group(2).strip(): (int(m.group(1)), m.group(3).strip())
             for m in re.finditer(r"^\| (\d+) \| (.+?) \| (.+) \|$", f.read(), re.M)
             if m.group(2).strip() in DUGUM}

TUR = {  # n8n türü -> (Türkçe ad, renk) — renkler kitapçıktaki şemayla aynı
    "telegramTrigger": ("Telegram (tetikleyici)", "#2a78d6"), "telegram": ("Telegram", "#2a78d6"),
    "if": ("Karar (IF)", "#eb6834"), "httpRequest": ("HTTP isteği (FastAPI)", "#1baf7a"),
    "chainLlm": ("Yapay zekâ zinciri (Claude)", "#eda100"), "lmChatAnthropic": ("Claude modeli (alt düğüm)", "#eda100"),
    "code": ("Kod (JavaScript)", "#e87ba4"), "googleSheets": ("Google Sheets", "#008300"), "wait": ("Bekle", "#4a3aa7"),
}
DAL = [(range(1, 4), "Giriş ve yönlendirme"), (range(4, 12), "A · Fotoğraf → teşhis ve rapor"),
       (range(12, 22), "D · Rapordan sonra (paralel işler)"), (range(22, 30), "B · Yazı → selam / düzeltme / sohbet"),
       (range(30, 36), "C · PDF butonu")]
SENARYOLAR = {
    "Serbest (kutulara tıkla)": [],
    "Fotoğraf gönderildi": [1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "Yazı: 'Domates' (bitki düzeltmesi)": [1, 2, 3, 22, 24, 25, 26, 4, 5, 6, 7, 8, 10, 9, 11, 13],
    "Yazı: sohbet sorusu": [1, 2, 3, 22, 24, 25, 28, 27, 29],
    "Yazı: 'merhaba'": [1, 2, 3, 22, 23],
    "PDF butonuna basıldı": [1, 2, 30, 31, 32, 33, 34],
}


def tur_of(n):
    t = n["type"].split(".")[-1]
    return TUR.get(t, (t, "#888888"))


def dal_of(no):
    return next(ad for aralik, ad in DAL if no in aralik)


ADLAR = {no: ad for ad, (no, _) in TABLO.items()}
OLCEK = 1.4  # n8n'deki dikey aralık kutulara dar; şemada açılır
satirlar = []
for ad, n in DUGUM.items():
    no = TABLO.get(ad, (0, ""))[0]
    x, y = n["position"]
    kisa = ad.replace("Telegram - ", "").replace("HTTP Request - ", "").replace("Anthropic Chat Model", "Claude modeli")
    satirlar.append({"no": no, "ad": ad, "kisa": kisa, "x": x, "y": -y * OLCEK, "tur": tur_of(n)[0],
                     "renk": tur_of(n)[1], "dal": dal_of(no) if no else ""})
df = pd.DataFrame(satirlar)

kenarlar = []
for kaynak, v in AKIS["connections"].items():
    for tip, dallar in v.items():
        for di, dal_ in enumerate(dallar):
            for h in dal_:
                a, b = DUGUM[kaynak]["position"], DUGUM[h["node"]]["position"]
                alt_dugum = tip != "main"
                etiket = "model" if alt_dugum else ("evet" if len(dallar) > 1 and di == 0 else
                                                    "hayır" if len(dallar) > 1 else "")
                kenarlar.append({"x": a[0] if alt_dugum else a[0] + 95, "y": -a[1] * OLCEK + (45 if alt_dugum else 0),
                                 "x2": b[0] if alt_dugum else b[0] - 95, "y2": -b[1] * OLCEK - (45 if alt_dugum else 0),
                                 "kaynak": TABLO[kaynak][0], "hedef": TABLO[h["node"]][0], "tur": etiket or "akış"})
dk = pd.DataFrame(kenarlar)

# --- Senaryo seçimi ------------------------------------------------------------------------------
sol, sag = st.columns([2, 1])
senaryo = sol.selectbox("Senaryo", list(SENARYOLAR), key="n8n_senaryo")
yol = SENARYOLAR[senaryo]
if st.session_state.get("n8n_son_senaryo") != senaryo:
    st.session_state["n8n_son_senaryo"] = senaryo
    st.session_state["n8n_adim"] = 0
def _adim_kaydir(fark: int, uzunluk: int) -> None:
    # on_click: sayfa yeniden çizilmeden ÖNCE çalışır -> butonların pasif/aktif durumu hep güncel olur
    st.session_state["n8n_adim"] = max(0, min(st.session_state.get("n8n_adim", 0) + fark, uzunluk - 1))


adim = min(st.session_state.get("n8n_adim", 0), max(len(yol) - 1, 0))
if yol:
    g1, g2, g3 = sag.columns([1, 1.3, 1], vertical_alignment="bottom")
    g1.button("◀", disabled=adim == 0, use_container_width=True, help="Önceki adım",
              on_click=_adim_kaydir, args=(-1, len(yol)))
    g3.button("▶", disabled=adim >= len(yol) - 1, type="primary", use_container_width=True, help="Sonraki adım",
              on_click=_adim_kaydir, args=(1, len(yol)))
    g2.markdown(f"<div style='text-align:center;padding-bottom:8px'>Adım {adim + 1} / {len(yol)}</div>",
                unsafe_allow_html=True)

gidilen = set(yol[: adim + 1]) if yol else set()
df["vurgu"] = df["no"].apply(lambda n: 1.0 if not yol or n in gidilen else 0.18)
df["su_an"] = df["no"].apply(lambda n: bool(yol) and n == yol[adim])
dk["vurgu"] = dk.apply(lambda r: 1.0 if not yol or (r["kaynak"] in gidilen and r["hedef"] in gidilen) else 0.12,
                       axis=1)

# --- Şema (Plotly: tek grafik, tıklanabilir) --------------------------------------------------------
G, Y = 95, 42  # kutunun yarı genişliği / yarı yüksekliği (n8n koordinat birimi)
KENAR_RENK = {"akış": "#8a90a0", "evet": "#008300", "hayır": "#e34948", "model": "#b9bcc4"}
fig = go.Figure()
for _, k in dk.iterrows():  # bağlantılar
    fig.add_trace(go.Scatter(x=[k["x"], k["x2"]], y=[k["y"], k["y2"]], mode="lines", hoverinfo="skip",
                             line=dict(color=KENAR_RENK[k["tur"]], width=2, dash="dash" if k["tur"] == "model" else None),
                             opacity=k["vurgu"], showlegend=False))
    fig.add_annotation(x=k["x2"], y=k["y2"], ax=k["x"], ay=k["y"], xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.6, arrowcolor=KENAR_RENK[k["tur"]],
                       opacity=k["vurgu"], text="")
secili_tik = st.session_state.get("n8n_tik")
for _, d in df.iterrows():  # kutular + yazılar
    kalin = bool(d["su_an"]) or (not yol and d["no"] == secili_tik)
    fig.add_shape(type="rect", x0=d["x"] - G, x1=d["x"] + G, y0=d["y"] - Y, y1=d["y"] + Y,
                  line=dict(color=d["renk"], width=4.5 if kalin else 1.8), fillcolor=d["renk"],
                  opacity=d["vurgu"] * (0.95 if kalin else 0.8), layer="below")
    fig.add_shape(type="rect", x0=d["x"] - G + 3, x1=d["x"] + G - 3, y0=d["y"] - Y + 3, y1=d["y"] + Y - 3,
                  line=dict(width=0), fillcolor="white", opacity=d["vurgu"] * 0.82, layer="below")
    kisa = "<br>".join(textwrap.wrap(d["kisa"], 15)[:2])  # kutuya sığsın: en çok 2 satır
    fig.add_annotation(x=d["x"], y=d["y"] - 10, text=kisa, showarrow=False, font=dict(size=10.5, color="#0b0b0b"),
                       opacity=d["vurgu"])
    fig.add_annotation(x=d["x"] - G + 8, y=d["y"] + Y - 14, text=f"<b>{d['no']}</b>", showarrow=False,
                       xanchor="left", font=dict(size=12, color="#0b0b0b"), opacity=d["vurgu"])
# Tıklama hedefi: her kutunun ortasında kutu boyunda, neredeyse görünmez kare işaret
fig.add_trace(go.Scatter(x=df["x"], y=df["y"], mode="markers", customdata=df[["no", "ad", "tur", "dal"]],
                         marker=dict(symbol="square", size=34, color="rgba(0,0,0,0.01)"),
                         hovertemplate="<b>%{customdata[0]}. %{customdata[1]}</b><br>%{customdata[2]}<br>"
                                       "%{customdata[3]}<extra></extra>", showlegend=False, name="dugum"))
LEJANT = {"Telegram": "#2a78d6", "Karar (IF)": "#eb6834", "HTTP isteği (FastAPI)": "#1baf7a",
          "Yapay zekâ (Claude)": "#eda100", "Kod (JavaScript)": "#e87ba4", "Google Sheets": "#008300",
          "Bekle": "#4a3aa7"}
for tur_ad, renk in LEJANT.items():  # lejant (aynı renkteki türler tek satır)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="square", size=12, color=renk),
                             name=tur_ad, showlegend=True))
fig.update_layout(height=780, margin=dict(l=4, r=4, t=4, b=4), plot_bgcolor="white", paper_bgcolor="white",
                  dragmode=False, clickmode="event+select", hovermode="closest",
                  legend=dict(orientation="h", y=-0.02, x=0, font=dict(size=11)),
                  xaxis=dict(visible=False, range=[df["x"].min() - 130, df["x"].max() + 130], fixedrange=True),
                  yaxis=dict(visible=False, range=[df["y"].min() - 80, df["y"].max() + 80], fixedrange=True))
olay = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="n8n_sema",
                       config={"displayModeBar": False})

# --- Seçilen düğümün açıklaması ------------------------------------------------------------------
secili_no = None
if yol:
    secili_no = yol[adim]
else:
    noktalar = (olay.selection.get("points") if olay and olay.selection else None) or []
    tik = next((pt["customdata"][0] for pt in noktalar if pt.get("customdata")), None)
    if tik is not None and tik != st.session_state.get("n8n_tik"):
        st.session_state["n8n_tik"] = int(tik)
        st.rerun()  # kalın çerçeve seçilen kutuya çizilsin
    secili_no = st.session_state.get("n8n_tik")

if not secili_no:
    st.info("Bir kutuya tıklayın (çift tıklama seçimi kaldırır) ya da yukarıdan bir senaryo seçip ▶ ile "
            "ilerleyin. Dört dal: **A** fotoğraf → rapor · **B** yazı → selam / bitki düzeltmesi / sohbet · "
            "**C** PDF butonu · **D** rapordan sonra paralel işler (kayıt, uzman, takip).")
else:
    ad = ADLAR[secili_no]
    n = DUGUM[ad]
    p = n["parameters"]
    tur_ad, renk = tur_of(n)
    st.markdown(f"<div style='border-left:6px solid {renk};padding:4px 12px;margin-top:4px'>"
                f"<span style='color:#6A7890'>{dal_of(secili_no)} · {tur_ad}</span><br>"
                f"<span style='font-size:1.35rem;font-weight:700'>{secili_no}. {ad}</span></div>",
                unsafe_allow_html=True)
    bulgu(TABLO[ad][1].replace("`", ""), "Ne yapar?")

    gelen = [(TABLO[k][0], k, e) for k, v in AKIS["connections"].items() for t, dallar in v.items()
             for di, d in enumerate(dallar) for h in d if h["node"] == ad
             for e in [("model olarak" if t != "main" else ("evet ise" if len(dallar) > 1 and di == 0 else
                                                            "hayır ise" if len(dallar) > 1 else ""))]]
    giden = [(TABLO[h["node"]][0], h["node"], ("evet ise" if len(dallar) > 1 and di == 0 else
                                               "hayır ise" if len(dallar) > 1 else ""))
             for t, dallar in AKIS["connections"].get(ad, {}).items() for di, d in enumerate(dallar) for h in d]
    k1, k2 = st.columns(2)
    k1.markdown("**Nereden gelir?**\n\n" + ("\n".join(f"- {no}. {a}" + (f" *({e})*" if e else "")
                                                      for no, a, e in sorted(gelen)) or "- Akışın başlangıcı"))
    k2.markdown("**Nereye gider?**\n\n" + ("\n".join(f"- {no}. {a}" + (f" *({e})*" if e else "")
                                                    for no, a, e in sorted(giden)) or "- Akışın sonu"))

    st.markdown("**n8n'deki ayarları** (canlı akıştan)")
    t = n["type"].split(".")[-1]
    if t == "httpRequest":
        st.code(f"{p.get('method', 'GET')} {p.get('url', '')}", language=None, wrap_lines=True)
        alanlar = (p.get("bodyParameters") or p.get("queryParameters") or {}).get("parameters", [])
        if alanlar:
            st.dataframe(pd.DataFrame([{"Alan": a.get("name"), "Değer": a.get("value") or a.get("inputDataFieldName")}
                                       for a in alanlar]), hide_index=True, use_container_width=True)
    elif t == "if":
        kosul = p["conditions"]["conditions"][0]["leftValue"]
        st.code(kosul, language="javascript", wrap_lines=True)
        st.caption("Koşul doğruysa üst çıkış (evet), değilse alt çıkış (hayır) çalışır.")
    elif t == "code":
        st.code(p.get("jsCode", ""), language="javascript")
    elif t == "chainLlm":
        st.markdown("Kullanıcı mesajı (her mesajda doldurulur):")
        st.code(p.get("text", ""), language=None, wrap_lines=True)
        sistem = (p.get("messages") or {}).get("messageValues", [{}])[0].get("message", "")
        with st.expander("Sistem mesajı (Claude'un kuralları)", expanded=True):
            st.code(sistem.lstrip("="), language=None, wrap_lines=True)
    elif t == "lmChatAnthropic":
        st.code(f"Model: {p['model'].get('cachedResultName', p['model'].get('value'))}\n"
                f"Kimlik bilgisi: Anthropic API anahtarı (n8n Credentials'ta şifreli)", language=None)
    elif t == "googleSheets":
        sutunlar = (p.get("columns") or {}).get("value") or {}
        st.dataframe(pd.DataFrame([{"Sütun": k, "Değer (ifade)": v} for k, v in sutunlar.items()]),
                     hide_index=True, use_container_width=True)
    elif t == "wait":
        st.code(f"{p.get('amount')} {p.get('unit')} bekler (sunum için; gerçek kullanımda 3 gün)", language=None)
    elif t == "telegramTrigger":
        st.code(f"Dinlenen güncellemeler: {', '.join(p.get('updates', []))}", language=None)
    else:  # telegram
        islem = p.get("operation") or p.get("resource") or "sendMessage"
        metin = p.get("text") or p.get("fileId") or p.get("queryId") or ""
        st.code(f"İşlem: {islem}\nAlıcı: {p.get('chatId', '-')}\nMetin / değer:\n{metin}", language=None,
                wrap_lines=True)
    st.caption("`{{ ... }}` n8n ifadesidir: değer, önceki düğümlerin çıktısından mesaj anında hesaplanır. "
               "`$json` = bir önceki düğümün çıktısı, `$('Ad').item.json` = adı verilen düğümün çıktısı.")

gezinme(__file__)
