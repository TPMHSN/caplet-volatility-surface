"""Exporte les grilles complètes, le détail du moteur et le rapport interactif."""

import csv
import json
from pathlib import Path
from tempfile import mkdtemp


ASSETS = Path(__file__).resolve().parent / "visuels"


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


def _number(value, scale=1):
    if value is None:
        return ""
    return str(value * scale).replace(".", ",")


def _export_surface(path: Path, surface: dict, strikes: list[float]) -> list[dict]:
    slices = []
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow([
            "indice", "début nominal (ans)", "date de fixing", "début du coupon",
            "fin du coupon", "date de paiement", "expiration ACT/365F (ans)",
            "forward (%)", "strike (%)", "vol native (bp)", "vol prolongée (bp)",
            "provenance", "indice source", "date de fixing source", "statut", "message",
        ])
        for row in surface["slices"]:
            points = {point["strike"]: point for point in row["curve"]}
            view = {
                "time": row["nominal_start_years"],
                "fixing": row["fixing_date"],
                "start": row.get("start_date"),
                "end": row.get("end_date"),
                "forward": row.get("forward_rate"),
                "errors": row["issues"],
                "vols": [],
                "native": [],
                "origins": [],
            }
            for strike in strikes:
                point = points.get(strike, {})
                vol = point.get("extended_normal_vol")
                native = point.get("native_normal_vol")
                coverage = point.get("coverage") or {}
                origin = coverage.get("classification", "native" if vol is not None else None)
                view["vols"].append(vol * 10_000 if vol is not None else None)
                view["native"].append(native * 10_000 if native is not None else None)
                view["origins"].append(origin)
                writer.writerow([
                    surface["index_tenor"], _number(row["nominal_start_years"]), row["fixing_date"],
                    row.get("start_date"), row.get("end_date"), row.get("payment_date"),
                    _number(row.get("expiry_years")), _number(row.get("forward_rate"), 100),
                    _number(strike, 100), _number(native, 10_000), _number(vol, 10_000), origin,
                    coverage.get("source_tenor"), coverage.get("source_fixing_date"),
                    "prêt" if vol is not None else "indisponible",
                    " | ".join(row["issues"]) if vol is None else "",
                ])
            slices.append(view)
    return slices


def write_results(output: Path, metadata: dict, strikes: list[float], surfaces: dict,
                  prices: list[dict]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix=f"surface-{metadata['date_marche']}-", dir=output)).resolve()
    views = {
        tenor: _export_surface(run / f"surface-{tenor}.csv", result, strikes)
        for tenor, result in surfaces.items()
    }
    _write_json(run / "surfaces.json", surfaces)
    _write_json(run / "calibration.json", metadata)
    price_view = []
    if prices:
        _write_json(run / "pricing.json", prices)
        for record in prices:
            result = record.get("resultat") or {}
            price_view.append({
                "nom": record["nom"],
                "prix": result.get("price"),
                "coupons_prolonges": result.get("extended_coupon_count"),
                "erreur": record.get("erreur"),
            })
    payload = {
        "meta": metadata,
        "strikes": [k * 100 for k in strikes],
        "series": views,
        "prices": price_view,
    }
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = (ASSETS / "rapport.html").read_text(encoding="utf-8")
    plotly = (ASSETS / "plotly.min.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
    content = template.replace("__PLOTLY__", plotly, 1).replace("__RESULTATS__", data, 1)
    report = run / "rapport.html"
    report.write_text(content, encoding="utf-8")
    return report
