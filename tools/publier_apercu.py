"""Génère la démo 3D avec les seules coordonnées tracées du rejeu."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rapport import render_viewer


def make_preview(surfaces: dict, calibration: dict) -> dict:
    strikes = sorted({
        point["strike"]
        for result in surfaces.values()
        for row in result["slices"]
        for point in row["curve"]
    })
    if not strikes:
        raise ValueError("La grille du rapport ne contient aucun strike.")

    series = {}
    for tenor in ("3M", "6M"):
        rows = []
        for row in surfaces[tenor]["slices"]:
            points = {point["strike"]: point for point in row["curve"]}
            vols, origins = [], []
            for strike in strikes:
                point = points.get(strike, {})
                vol = point.get("extended_normal_vol")
                coverage = point.get("coverage") or {}
                vols.append(None if vol is None else round(vol * 10_000, 1))
                origins.append(None if vol is None else coverage.get("classification", "native"))
            rows.append({
                "time": row["nominal_start_years"],
                "fixing": row["fixing_date"],
                "start": row.get("start_date"),
                "end": row.get("end_date"),
                "vols": vols,
                "origins": origins,
                "errors": [],
            })
        if not rows:
            raise ValueError(f"La série Euribor {tenor} ne contient aucun fixing.")
        series[tenor] = rows
    return {
        "preview": True,
        "meta": {
            "date_marche": calibration["date_marche"],
            "correlation_rho": calibration["correlation_rho"],
            "validation_cotations_natives": calibration["validation_cotations_natives"],
        },
        "strikes": [round(k * 100, 8) for k in strikes],
        "series": series,
        "prices": [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rapport", type=Path, help="Dossier des sorties du rejeu")
    args = parser.parse_args()
    surfaces = json.loads((args.rapport / "surfaces.json").read_text(encoding="utf-8"))
    calibration = json.loads((args.rapport / "calibration.json").read_text(encoding="utf-8"))
    data = make_preview(surfaces, calibration)
    destination = ROOT / "docs" / "index.html"
    destination.write_text(render_viewer(data, public_preview=True), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
