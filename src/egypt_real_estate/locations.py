"""Location standardization.

The three sources describe locations very differently:

* OLX:    ``City`` = "New Cairo - El Tagamoa", plus a separate ``Compound`` column
* Bayut:  "Badya Palm Hills Compound, 6th of October, Giza"
* Cairo:  "mivida, 5th settlement compounds, the 5th settlement, new cairo city, cairo"

Every raw string is mapped to one canonical **district** (e.g. "New Cairo"),
its **region** (Greater Cairo, North Coast, ...), and, where the source gives
one, a normalized **compound** name.
"""

from __future__ import annotations

import re

import pandas as pd

# Ordered (district, region, pattern). The FIRST match wins, so specific places
# come before broad ones (e.g. "Sheikh Zayed" before the generic "5th ..." tokens
# used by New Cairo, and every named district before the "Cairo"/"Giza" fallbacks).
DISTRICT_RULES: list[tuple[str, str, str]] = [
    ("New Capital", "Greater Cairo", r"new capital|administrative capital|new administrative"),
    ("Mostakbal City", "Greater Cairo", r"mostakbal|mostaqbal|future city"),
    ("Madinaty", "Greater Cairo", r"madinaty"),
    ("Shorouk City", "Greater Cairo", r"shorouk|sherouk"),
    ("Rehab City", "Greater Cairo", r"rehab"),
    ("Obour City", "Greater Cairo", r"obour"),
    ("Badr City", "Greater Cairo", r"badr city"),
    ("New Heliopolis", "Greater Cairo", r"new heliopolis"),
    ("Sheikh Zayed", "Greater Cairo", r"zayed|beverly hills"),
    ("6th of October", "Greater Cairo", r"october|smart village|somid"),
    ("North Coast", "North Coast", r"north coast|sahel|alamein|hekma|marassi|sidi abdel ?rahman|"
                                   r"dabaa|matrouh|hacienda|fouka|ghazala|marina"),
    ("Ain Sokhna", "Red Sea & Sinai", r"sokhna|sukhna|galala|zafarana"),
    ("Hurghada & El Gouna", "Red Sea & Sinai", r"hurghada|ghardaqa|gouna|sahl hasheesh|makadi|"
                                               r"soma bay|safaga|red sea"),
    ("South Sinai", "Red Sea & Sinai", r"sharm|dahab|sinai|ras sudr|ras sedr"),
    ("Alexandria", "Alexandria", r"alexandria|\balex\b|smouha|smoha|camp caesar|sidi gaber|miami|"
                                 r"sporting|stanley|gleem|roushdy|rushdy|louran|laurent|mandara|"
                                 r"agami|montaza|sidi bishr|kafr abdo|bolkly|cleopatra|ibrahimia|"
                                 r"borg (el|al)-?arab|asafra|san stefano|mariout|moharam bek|nakheel|sidi beshr|glim|saba pasha|"
                                 r"maamoura|zezenia|abu talat|ibrahimiyyah|gianaclis|fleming|"
                                 r"mahatet el raml|raml station|el max|abu qir|seyouf|amreya"),
    ("Heliopolis", "Greater Cairo", r"heliopolis|masr el gedida|misr el gedida|sheraton|nozha|almaza"),
    ("Nasr City", "Greater Cairo", r"nasr city|madinat nasr"),
    ("Maadi", "Greater Cairo", r"maadi|ma'adi|degla|zahraa"),
    ("Mokattam", "Greater Cairo", r"mokattam|moqattam"),
    ("Zamalek", "Greater Cairo", r"zamalek"),
    ("Downtown & Old Cairo", "Greater Cairo", r"garden city|downtown|wust el balad|abdeen|manial|fostat|fustat|"
                                                 r"masr al-?kadema|masr el kadima|qasr al-?nil|sayeda zainab|bulaq|ramses"),
    ("Mohandessin & Dokki", "Greater Cairo", r"mohandes|dokki|doqi|agouza"),
    ("Haram & Faisal", "Greater Cairo", r"haram|faisal|pyramids|hadayek al.?ahram|hadaba"),
    ("Helwan", "Greater Cairo", r"helwan|15th of may|15 may"),
    ("Shubra", "Greater Cairo", r"shubra|shoubra"),
    ("Ain Shams, Matareya & Zeitoun", "Greater Cairo", r"ain shams|matareya|matariya|marg|zaytoun|zeitoun"),
    ("Hadayek El Kobba & Abbasiya", "Greater Cairo", r"kobba|qubba|abb?asi"),
    # New Cairo uses many generic tokens, so it comes after every named district.
    ("New Cairo", "Greater Cairo", r"new cairo|tagamo|settlement|katameya|katamya|qattameya|"
                                   r"investors|golden square|narges|banafsag|yasmeen|lotus|"
                                   r"mivida|hyde park|mountain view|villette|el andalus"),
    ("Damietta & Port Said", "Canal & Delta", r"damietta|port said|ras (el|al)-?bar"),
    ("Ismailia & Suez", "Canal & Delta", r"ismailia|suez"),
    ("Delta Cities", "Canal & Delta", r"mansoura|mansura|10th of ramadan|tanta|zagazig|banha|menouf|kafr el|"
                                      r"gharbia|dakahlia|sharqia|beheira|damanhour|menofia|qalyub"),
    ("Upper Egypt", "Upper Egypt", r"assiut|asyut|minya|sohag|luxor|aswan|qena|fayoum|beni suef"),
    ("Cairo (other)", "Greater Cairo", r"cairo"),
    ("Giza (other)", "Greater Cairo", r"giza"),
]

_COMPILED = [(d, r, re.compile(p)) for d, r, p in DISTRICT_RULES]

OTHER = "Other"


def normalize_text(value: object) -> str:
    """Lowercase, trim, and collapse punctuation/whitespace."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[–—_]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text


def match_district(location: object) -> tuple[str, str]:
    """Return ``(district, region)`` for a raw location string."""
    text = normalize_text(location)
    if not text:
        return OTHER, OTHER
    for district, region, pattern in _COMPILED:
        if pattern.search(text):
            return district, region
    return OTHER, OTHER


_COMPOUND_NOISE = re.compile(
    r"\b(compound|compounds|resort|residence|residences|the|by [a-z ]+ developments?)\b"
)


def normalize_compound(value: object) -> str | None:
    """Normalize a compound name so the same project matches across sources."""
    text = normalize_text(value)
    if not text or text in {"unknown", "nan", "none", "other"}:
        return None
    text = _COMPOUND_NOISE.sub(" ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 3 or re.search(r"settlement|district|neighbo", text):
        return None  # a sub-area label, not a compound
    # A compound name that is really a district name carries no extra information.
    if match_district(text)[0] != OTHER and text in {
        "madinaty", "rehab", "rehab city", "new cairo", "sheikh zayed", "new capital",
        "north coast", "shorouk", "mostakbal city", "6th of october",
    }:
        return None
    return text.title()


def compound_from_hierarchy(location: object, min_parts: int) -> str | None:
    """Take the most specific part of "A, B, C, ..." as the compound.

    Only used when the string has at least ``min_parts`` parts, i.e. when the
    first part is more specific than the district itself.
    """
    parts = [p.strip() for p in str(location).split(",") if p.strip()]
    if len(parts) < min_parts:
        return None
    first = parts[0]
    # Skip generic neighbourhood labels such as "10th District" or "Area A".
    if re.search(r"\b(district|neighbo(u)?rhood|area|zone|street|st\.?|road|block)\b", first, re.I):
        return None
    return normalize_compound(first)


def standardize_locations(
    df: pd.DataFrame, location_col: str, compound: pd.Series | None = None
) -> pd.DataFrame:
    """Add ``district``, ``region`` and ``compound`` columns to ``df``."""
    out = df.copy()
    matched = out[location_col].map(match_district)
    out["district"] = matched.str[0]
    out["region"] = matched.str[1]
    out["compound"] = compound if compound is not None else None
    return out
