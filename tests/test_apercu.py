"""Vérifie le périmètre de données de la démo publique."""

import json

from tools.publier_apercu import make_preview


def test_preview_keeps_only_the_rounded_surface():
    row = {
        "nominal_start_years": 1.5,
        "fixing_date": "2028-05-19",
        "start_date": "2028-05-23",
        "end_date": "2028-08-23",
        "forward_rate": 0.02919123,
        "issues": ["Diagnostic local"],
        "curve": [
            {"strike": 0.03, "extended_normal_vol": 0.0081225,
             "native_normal_vol": 0.0081234, "coverage": None},
            {"strike": 0.05, "extended_normal_vol": None, "coverage": None},
        ],
    }
    surfaces = {tenor: {"slices": [row]} for tenor in ("3M", "6M")}
    meta = {
        "date_marche": "2026-11-20",
        "correlation_rho": 1,
        "validation_cotations_natives": "passed",
        "fixings": {"3M": {"taux_decimal": 0.0222222}},
        "classeur": "classeur local.xlsx",
    }

    preview = make_preview(surfaces, meta)
    assert preview["strikes"] == [3, 5]
    assert preview["series"]["3M"][0]["vols"] == [81.2, None]
    assert preview["series"]["3M"][0]["origins"] == ["native", None]
    content = json.dumps(preview)
    for token in ("taux_decimal", "0.0222222", "0.02919123", "0.0081234", "classeur local", "Diagnostic local"):
        assert token not in content
