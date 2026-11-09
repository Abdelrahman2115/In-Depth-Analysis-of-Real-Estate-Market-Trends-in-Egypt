import pandas as pd

from egypt_real_estate.cleaning import build_dataset


def _write_raw(tmp_path):
    pd.DataFrame({
        "Type": ["Apartment", "Stand Alone Villa", "Apartment", "Apartment"],
        "Price": ["4000000", "20000000", "3000000", "1000"],
        "Bedrooms": ["3", "5", "Unknown", "2"],
        "Bathrooms": ["2", "4", "2", "1"],
        "Area": ["160", "400", "120", "100"],
        "Furnished": "No", "Level": "1", "Payment_Option": "Cash",
        "Delivery_Date": "Ready to move", "Delivery_Term": "Finished",
        "Compound": ["Mivida", "Unknown", "Unknown", "Unknown"],
        "City": ["New Cairo - El Tagamoa", "Sheikh Zayed", "Nasr City", "Nasr City"],
    }).to_csv(tmp_path / "Egypt_Houses_Price.csv", index=False)

    pd.DataFrame({
        "link": ["a", "a", "b", "c"],
        "price": ["4,000,000", "4,000,000", "500,000", "6,000,000"],
        "currency": ["EGP", "EGP", "USD", "EGP"],
        "type": ["Apartment", "Apartment", "Villa", "Townhouse"],
        "beds": [3, 3, 4, 3], "baths": [2, 2, 3, 3],
        "area": ["160 Sq. M.", "160 Sq. M.", "300 Sq. M.", "200 Sq. M."],
        "location": ["Mivida, New Cairo, Cairo"] * 2 + ["Sheikh Zayed, Giza", "Madinaty, Cairo"],
        "down_payment": None, "plan_type": None,
    }).to_csv(tmp_path / "real_estate_data_bayut_full.csv", index=False)

    pd.DataFrame({
        "type": ["Apartment", "Villa"],
        "title": ["x", "y"],
        "location": ["Madinaty, Cairo", "Hyde Park, 5th Settlement Compounds, The 5th Settlement, New Cairo City, Cairo"],
        "bedroom": ["Studio", "4"],
        "bathroom": [1, 4],
        "size_sqm": ["60", "350"],
        "price": ["Ask", "15,000,000"],
    }).to_csv(tmp_path / "properties.csv", index=False)


def test_build_dataset(tmp_path):
    _write_raw(tmp_path)
    df, log = build_dataset(tmp_path)

    # USD listing removed, repeated Bayut URL removed, OLX/Bayut cross-site duplicate merged,
    # 'Ask' price removed, unknown bedrooms removed, EGP 1,000 apartment removed as implausible.
    assert set(df["source"]) == {"OLX", "Bayut", "Cairo"}
    assert len(df) == 4
    assert (df["price_per_sqm"] >= 3_000).all()
    assert df["district"].tolist().count("New Cairo") == 2  # duplicate collapsed, Hyde Park kept
    assert "Madinaty" in set(df["district"])
    kept_new_cairo_apartment = df[(df["district"] == "New Cairo") & (df["type"] == "Apartment")]
    assert kept_new_cairo_apartment["compound"].iloc[0] == "Mivida"

    steps = log.to_frame()
    assert steps["removed"].ge(0).all()
    usd = steps[steps["step"].str.contains("EGP prices only")]
    assert usd["removed"].iloc[0] == 1
