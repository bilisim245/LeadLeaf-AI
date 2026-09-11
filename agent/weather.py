"""
Hava durumu modulu — Open-Meteo (ucretsiz, API anahtari GEREKMEZ).

Kullanim:
    from agent.weather import weather_summary
    w = weather_summary("Antalya")
    print(w["ozet_metni"])

CLI testi:
    python agent/weather.py Antalya
    python agent/weather.py "Serik, Antalya"
"""
from __future__ import annotations

import sys
import datetime as dt
from typing import Optional

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 15


def geocode(yer: str) -> Optional[dict]:
    """Yer adi -> koordinat + il/ilce. Bulamazsa None."""
    r = requests.get(
        GEOCODE_URL,
        params={"name": yer, "count": 1, "language": "tr", "format": "json"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    sonuc = (r.json() or {}).get("results") or []
    if not sonuc:
        return None
    g = sonuc[0]
    return {
        "ad": g.get("name"),
        "il": g.get("admin1"),
        "ilce": g.get("admin2") or g.get("admin3"),
        "lat": g["latitude"],
        "lon": g["longitude"],
    }


def forecast(lat: float, lon: float, gun: int = 3) -> list[dict]:
    """Gunluk tahmin: sicaklik, yagis, ortalama nem."""
    r = requests.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
            "hourly": "relative_humidity_2m",
            "timezone": "auto",
            "forecast_days": gun,
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    j = r.json()
    daily = j.get("daily", {})
    hourly = j.get("hourly", {})

    # saatlik nemi gune gore ortala
    nem_gune: dict[str, list[float]] = {}
    for zaman, nem in zip(hourly.get("time", []), hourly.get("relative_humidity_2m", [])):
        gun_str = zaman[:10]
        if nem is not None:
            nem_gune.setdefault(gun_str, []).append(nem)

    gunler = []
    for i, tarih in enumerate(daily.get("time", [])):
        nemler = nem_gune.get(tarih, [])
        gunler.append({
            "tarih": tarih,
            "tmin": _g(daily, "temperature_2m_min", i),
            "tmax": _g(daily, "temperature_2m_max", i),
            "yagis_mm": _g(daily, "precipitation_sum", i),
            "yagis_olasilik": _g(daily, "precipitation_probability_max", i),
            "nem_ort": round(sum(nemler) / len(nemler), 1) if nemler else None,
        })
    return gunler


def _g(d: dict, anahtar: str, i: int):
    dizi = d.get(anahtar) or []
    return dizi[i] if i < len(dizi) else None


def _mantar_riski(gunler: list[dict]) -> str:
    """Basit sezgisel: yuksek nem + yagis => mantar hastaligi riski."""
    if not gunler:
        return "bilinmiyor"
    nem = [g["nem_ort"] for g in gunler if g["nem_ort"] is not None]
    yagis = [g["yagis_mm"] for g in gunler if g["yagis_mm"] is not None]
    ort_nem = sum(nem) / len(nem) if nem else 0
    top_yagis = sum(yagis) if yagis else 0
    if ort_nem >= 80 or top_yagis >= 10:
        return "yuksek"
    if ort_nem >= 65 or top_yagis >= 2:
        return "orta"
    return "dusuk"


def weather_summary(yer: str, gun: int = 3) -> dict:
    """Agent'in kullanacagi tek fonksiyon."""
    konum = geocode(yer)
    if not konum:
        return {"bulundu": False, "sorgu": yer,
                "ozet_metni": f"'{yer}' icin konum bulunamadi."}

    gunler = forecast(konum["lat"], konum["lon"], gun)
    risk = _mantar_riski(gunler)

    satirlar = [
        f"{konum['ad']}"
        + (f" ({konum['ilce']}/{konum['il']})" if konum.get("ilce") else "")
        + f" — onumuzdeki {len(gunler)} gun:"
    ]
    for g in gunler:
        satirlar.append(
            f"  {g['tarih']}: {g['tmin']}–{g['tmax']}°C, "
            f"nem ~%{g['nem_ort']}, yagis {g['yagis_mm']} mm "
            f"(olasilik %{g['yagis_olasilik']})"
        )
    satirlar.append(f"  Mantar hastaligi riski: {risk.upper()}")

    return {
        "bulundu": True,
        "sorgu": yer,
        "konum": konum,
        "gunler": gunler,
        "mantar_riski": risk,
        "ozet_metni": "\n".join(satirlar),
        "olusturma": dt.datetime.now().isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    yer = " ".join(sys.argv[1:]) or "Antalya"
    w = weather_summary(yer)
    print(w["ozet_metni"])
