import pytest

from egypt_real_estate.locations import compound_from_hierarchy, match_district, normalize_compound


@pytest.mark.parametrize("raw, district", [
    # The same area written three different ways must map to one district.
    ("New Cairo - El Tagamoa", "New Cairo"),
    ("Cairo Festival City, New Cairo, Cairo", "New Cairo"),
    ("mivida, 5th settlement compounds, the 5th settlement, new cairo city, cairo", "New Cairo"),
    ("Madinaty", "Madinaty"),
    ("Madinaty, Madinaty", "Madinaty"),
    ("madinaty, cairo", "Madinaty"),
    ("Rehab City Compound, New Cairo, Cairo", "Rehab City"),
    ("Sheikh Zayed, Giza", "Sheikh Zayed"),
    ("10th District, Sheikh Zayed, Giza", "Sheikh Zayed"),
    ("Badya Palm Hills Compound, 6th of October, Giza", "6th of October"),
    ("Hadayek October", "6th of October"),
    ("Hadayek al Ahram", "Haram & Faisal"),
    ("Hadayek al-Ahram, Giza", "Haram & Faisal"),
    ("El Fostat, Cairo", "Downtown & Old Cairo"),
    ("Zizinia Gardens, Ext North Inves Area, New Cairo City, Cairo", "New Cairo"),
    ("Borg al-Arab", "Alexandria"),
    ("La Vista City, New Capital Compounds, New Capital City, Cairo", "New Capital"),
    ("Mostakbal City, Cairo", "Mostakbal City"),
    ("North Coast", "North Coast"),
    ("Marassi, Sidi Abdel Rahman, North Coast", "North Coast"),
    ("Ain Sukhna", "Ain Sokhna"),
    ("Smoha", "Alexandria"),
    ("Camp Caesar", "Alexandria"),
    ("Nasr City", "Nasr City"),
    ("Hay El Maadi, Cairo", "Maadi"),
])
def test_match_district(raw, district):
    assert match_district(raw)[0] == district


def test_unknown_location_is_other():
    assert match_district("Somewhere Unmapped")[0] == "Other"
    assert match_district(None)[0] == "Other"


def test_compound_normalization_matches_across_sources():
    assert normalize_compound("Mivida") == normalize_compound("mivida compound") == "Mivida"
    assert normalize_compound("Unknown") is None
    assert normalize_compound("Madinaty") is None  # a district, not a compound


def test_compound_from_hierarchy():
    assert compound_from_hierarchy("Badya Palm Hills Compound, 6th of October, Giza", 3) == "Badya Palm Hills"
    assert compound_from_hierarchy("Sheikh Zayed, Giza", 3) is None
    assert compound_from_hierarchy("10th District, Sheikh Zayed, Giza", 3) is None
