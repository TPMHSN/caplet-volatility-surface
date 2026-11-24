"""Recalcule depuis un classeur .xlsx les surfaces Euribor 3M et 6M à 30 ans."""

from __future__ import annotations

import argparse
from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from time import monotonic
import webbrowser

from rapport import write_results


ROOT = Path(__file__).resolve().parent
DATE_IN_FILENAME = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2}|\d{8})(?!\d)")


def _market_date(path: Path) -> date:
    found = DATE_IN_FILENAME.search(path.stem)
    if found is None:
        raise ValueError("Date de marché absente du nom du classeur. Indiquer --date AAAA-MM-JJ.")
    value = found.group(1).replace("-", "")
    return datetime.strptime(value, "%Y%m%d").date()


def _pick_workbook():
    import tkinter as tk
    from tkinter import filedialog, simpledialog

    window = tk.Tk()
    window.withdraw()
    try:
        name = filedialog.askopenfilename(
            parent=window,
            title="Classeur de marché",
            initialdir=ROOT,
            filetypes=[("Classeur Excel", "*.xlsx")],
        )
        if not name:
            return None, None
        workbook = Path(name)
        try:
            return workbook, _market_date(workbook)
        except ValueError:
            value = simpledialog.askstring("Date de marché", "Date de marché (AAAA-MM-JJ)", parent=window)
            return (workbook, date.fromisoformat(value.strip())) if value else (None, None)
    finally:
        window.destroy()


def _price_jobs(path: Path | None) -> list[dict]:
    if path is None:
        return []
    inputs = json.loads(path.read_text(encoding="utf-8-sig"))
    jobs = inputs if isinstance(inputs, list) else [inputs]
    if not jobs or any(not isinstance(job, dict) for job in jobs):
        raise ValueError("Le fichier de pricing doit contenir un contrat JSON ou une liste de contrats.")
    return jobs


def replay(workbook: Path, as_of: date, fixings: Path, output: Path, *,
           rho: float = 1.0, strike_step_bp: int = 50, offline: bool = True,
           pricing_jobs: tuple | list = ()) -> tuple[Path, bool]:
    """Convertit le classeur en deux surfaces et, à la demande, en primes de contrats.

    La préparation construit les courbes et la calibration. L’extension
    interroge cette calibration commune pour chaque fixing et chaque strike.
    Le rapport affiche les résultats avec leurs unités de marché.
    """
    if strike_step_bp < 5 or 650 % strike_step_bp:
        raise ValueError("Le pas de strike doit diviser 650 bp et valoir au moins 5 bp.")
    if workbook.suffix.lower() != ".xlsx":
        raise ValueError("Choisir un classeur au format .xlsx.")
    content = workbook.read_bytes()
    sys.path.insert(0, str(ROOT / "moteur"))

    import QuantLib as ql

    from curves.helpers import date_to_ql
    from fixings_io import load_fixings
    from services.exceptions import ServiceError
    from services.latest_normal_candidate_service import prepare_candidate_workbook
    from services.latest_normal_extension import extended_surface, price_extended_model
    from services.latest_normal_extension_schemas import (
        ExtendedCandidatePriceRequest,
        ExtendedCandidateSurfaceRequest,
        ExtensionConfig,
    )

    if not ql.TARGET().isBusinessDay(date_to_ql(as_of)):
        raise ValueError(f"Le {as_of} doit être un jour ouvré TARGET.")
    config = ExtensionConfig(correlation_rho=rho)
    clock = monotonic()
    print(f"Date de marché {as_of} · chargement des fixings.", flush=True)
    observations = load_fixings(as_of, fixings, offline=offline)
    print("Bootstrap des courbes €STR, Euribor 3M et 6M, puis calibration des caplets.", flush=True)
    model = prepare_candidate_workbook(content, as_of)
    print(f"Calibration terminée ({monotonic() - clock:.0f} s).", flush=True)

    # Les vols et les strikes sont stockés en taux décimaux dans le moteur.
    strikes = [bp / 10_000 for bp in range(-150, 501, strike_step_bp)]
    market_context = {"model_id": model["model_id"], "valuation_date": as_of}
    surfaces = {}
    for tenor in ("3M", "6M"):
        print(f"Calcul de la surface {tenor} à 30 ans…", flush=True)
        request = ExtendedCandidateSurfaceRequest(
            **market_context, index_tenor=tenor, grid_to_30y=True, strikes=strikes, extension=config
        )
        result = extended_surface(request)
        count = sum(p["extended_normal_vol"] is not None for row in result["slices"] for p in row["curve"])
        total = len(result["slices"]) * len(strikes)
        print(f"Euribor {tenor} · {count}/{total} points calculés.", flush=True)
        surfaces[tenor] = result

    prices = []
    for number, job in enumerate(pricing_jobs, 1):
        terms = dict(job)
        label = str(terms.pop("nom", f"Contrat {number}"))
        try:
            terms = {**terms, **market_context, "extension": config}
            request = ExtendedCandidatePriceRequest.model_validate(terms)
            print(f"Pricing {label}…", flush=True)
            prices.append({"nom": label, "resultat": price_extended_model(request)})
        except (ServiceError, ValueError) as exc:
            prices.append({"nom": label, "erreur": str(exc)})
            print(f"{label} · {exc}", file=sys.stderr, flush=True)
    metadata = {
        "classeur": workbook.name,
        "sha256_classeur": sha256(content).hexdigest(),
        "date_marche": as_of.isoformat(),
        "modele": model["model"],
        "version": model["version"],
        "correlation_rho": rho,
        "fixings": observations,
        "empreinte_fixings_euribor": model["fixings"]["fingerprint"],
        "validation_cotations_natives": model["validation"]["internal_validation"],
        "diagnostics_cotations_natives": [
            {
                "indice": family["index_tenor"],
                "status": family["status"],
                "validation_repricing": family["quote_fit"]["final"],
                "issues": family["issues"],
            }
            for family in model["families"]
        ],
        "avertissements_calibration": model["warnings"],
    }
    report = write_results(output, metadata, strikes, surfaces, prices)
    complete = all(row["status"] == "ready" for result in surfaces.values() for row in result["slices"])
    return report, complete and all("erreur" not in row for row in prices)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("classeur", type=Path, nargs="?", help="Classeur de marché .xlsx à charger.")
    parser.add_argument("--date", type=date.fromisoformat, help="Date de marché AAAA-MM-JJ.")
    parser.add_argument("--sortie", type=Path, default=ROOT / "resultats", help="Répertoire de sortie.")
    parser.add_argument("--fixings-dir", type=Path, default=ROOT / "local" / "fixings", help="Historiques JSON.")
    parser.add_argument("--hors-ligne", action="store_true", help="Utiliser uniquement les historiques locaux.")
    parser.add_argument("--rho", type=float, default=1.0, help="Corrélation de transfert 3M/6M.")
    parser.add_argument("--pas-strike-bp", type=int, default=50, help="Pas de la grille en bp (50 par défaut).")
    parser.add_argument("--pricing", type=Path, help="Charger un ou plusieurs contrats depuis un JSON.")
    parser.add_argument("--ouvrir", action="store_true", help="Ouvrir le rapport dans le navigateur.")
    parser.add_argument("--choisir", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.classeur is None and args.choisir:
            workbook, picked_date = _pick_workbook()
            if workbook is None:
                return 0
        elif args.classeur is not None:
            workbook, picked_date = args.classeur, None
        else:
            parser.error("Indiquer le chemin du classeur de marché .xlsx.")
        as_of = args.date or picked_date or _market_date(workbook)
        report, complete = replay(
            workbook.resolve(), as_of, args.fixings_dir.resolve(), args.sortie.resolve(),
            rho=args.rho, strike_step_bp=args.pas_strike_bp, offline=args.hors_ligne,
            pricing_jobs=_price_jobs(args.pricing),
        )
        print(f"Rapport {'complet' if complete else 'avec points ou prix indisponibles'}\n{report}", flush=True)
        if args.ouvrir:
            webbrowser.open(report.as_uri())
        return 0 if complete else 2
    except Exception as exc:
        print(f"Calcul interrompu · {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
