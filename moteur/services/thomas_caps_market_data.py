"""Inspect an EUR market workbook without touching the shared pricing state.

The template contract is explicit: rate quotes in percentage points (or actual
Excel percent cells), absolute strike headers, normal cap vols in basis points.
This module does not construct curves, strip caplets, or calibrate SABR.
"""

import math
import re
from datetime import date
from io import BytesIO
from numbers import Real
from types import SimpleNamespace
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile, ZipFile

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.utils.exceptions import InvalidFileException

from services.thomas_caps_schemas import (
    CapQuote,
    CapRow,
    CapStrike,
    CapSurfaceInput,
    CurveInput,
    CurveQuote,
    DataCounts,
    DataIssue,
    SourceSheet,
    TenorPolicy,
    ThomasCapsMarketData,
)

MAX_WORKBOOK_BYTES = 10 * 1024 * 1024
MAX_EXPANDED_BYTES = 40 * 1024 * 1024
ROLES = {
    "ESTR": ("Actualisation EUR (€STR)", None),
    "EUR3": ("Projection Euribor 3M", "3M"),
    "EUR6": ("Projection Euribor 6M", "6M"),
    "VOL CAP": ("Volatilités normales de caps", None),
}


def _label(value):
    return re.sub(r"\s+", " ", str(value or "").strip()).upper()


def _months(value):
    match = re.fullmatch(r"([1-9]\d*)\s*(M|MO|MOS|MONTHS?|Y|YR|YRS|YEARS?)", _label(value))
    if not match:
        return None
    return int(match[1]) * (12 if match[2].startswith("Y") else 1)


def _number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, Real):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip().replace(",", "."))
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def prepare_market_data(
    content: bytes,
    filename: str,
    valuation_date: date | None = None,
    quote_tenor_policy: TenorPolicy = "market",
) -> ThomasCapsMarketData:
    """Return normalized inputs plus cell-level diagnostics; never replace missing data by zero."""
    if quote_tenor_policy not in ("market", "3M", "6M"):
        raise ValueError("Convention de tenor inconnue.")
    if len(content) > MAX_WORKBOOK_BYTES:
        raise ValueError("Le fichier Excel dépasse la limite de 10 Mo.")
    issues: list[DataIssue] = []

    def issue(code, message, sheet=None, cell=None, severity="error"):
        issues.append(
            DataIssue(
                severity=severity,
                code=code,
                message=message,
                sheet=sheet,
                cell=cell,
            )
        )

    # File modification timestamps are not market observation dates.
    filename = re.split(r"[\\/]", filename)[-1]
    match = re.search(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", filename)
    file_date = None
    if match:
        try:
            file_date = date(*map(int, match.groups()))
        except ValueError:
            issue(
                "invalid_filename_date",
                "La date du nom de fichier est invalide.",
                severity="warning",
            )
    selected_date = valuation_date or file_date
    if selected_date is None:
        raise ValueError("Choisir une date de marché : aucune date AAAAMMJJ valide dans le nom.")
    if valuation_date and file_date and valuation_date != file_date:
        issue(
            "date_mismatch",
            "La date choisie diffère de celle du nom du fichier.",
            severity="warning",
        )

    try:
        with ZipFile(BytesIO(content)) as archive:
            if sum(info.file_size for info in archive.infolist()) > MAX_EXPANDED_BYTES:
                raise ValueError("Le classeur Excel décompressé dépasse la limite de 40 Mo.")
        values = load_workbook(BytesIO(content), read_only=True, data_only=True)
        formulas = load_workbook(BytesIO(content), read_only=True, data_only=False)
    except (BadZipFile, InvalidFileException, KeyError, OSError, ValueError, ParseError) as exc:
        raise ValueError("Fichier Excel .xlsx invalide ou trop volumineux.") from exc

    try:
        sheets = {}
        inventory = []
        for sheet in values:
            key = _label(sheet.title)
            inventory.append(
                SourceSheet(
                    name=sheet.title,
                    role=ROLES.get(key, ("Hors périmètre caps EUR", None))[0],
                    used=key in ROLES,
                )
            )
            if key in sheets:
                issue(
                    "duplicate_sheet", "Deux feuilles portent le même nom normalisé.", sheet.title
                )
            else:
                sheets[key] = sheet

        required = ["ESTR", "VOL CAP"]
        if quote_tenor_policy in ("market", "3M"):
            required.append("EUR3")
        if quote_tenor_policy in ("market", "6M"):
            required.append("EUR6")
        for name in required:
            if name not in sheets:
                issue("missing_sheet", f"Feuille requise absente : {name}.", name)

        def rows_for(sheet):
            # Excel dimensions are optional metadata and may understate the
            # actual populated range. Recompute before accepting any quotes.
            for worksheet in (sheet, formulas[sheet.title]):
                worksheet.reset_dimensions()
                worksheet.calculate_dimension(force=True)
            if sheet.max_row > 500 or sheet.max_column > 100:
                issue("sheet_size", "Feuille limitée à 500 lignes et 100 colonnes.", sheet.title)
                return []

            # Materialize once: repeatedly indexing read-only sheets reparses XML.
            def cells(worksheet):
                return [
                    [
                        SimpleNamespace(
                            value=cell.value,
                            data_type=cell.data_type,
                            number_format=cell.number_format or "General",
                            coordinate=f"{get_column_letter(column)}{row_number}",
                        )
                        for column, cell in enumerate(row, start=1)
                    ]
                    for row_number, row in enumerate(worksheet.iter_rows(), start=1)
                ]

            return list(zip(cells(sheet), cells(formulas[sheet.title])))

        def read_numeric(cell, original, name):
            value = _number(cell.value)
            if value is None:
                if original.data_type == "f":
                    code, message = (
                        "formula_without_cache",
                        "Formule sans résultat numérique enregistré ; recalculer et enregistrer dans Excel.",
                    )
                elif cell.value is None:
                    code, message = "missing_value", "Cotation absente."
                else:
                    code, message = "invalid_value", "Cotation non numérique ou non finie."
                issue(code, message, name, cell.coordinate)
            return value

        curves = []
        for name in ("ESTR", "EUR3", "EUR6"):
            if name not in sheets:
                continue
            sheet = sheets[name]
            rows = rows_for(sheet)
            curve = CurveInput(
                sheet=sheet.title, purpose=ROLES[name][0], index_tenor=ROLES[name][1]
            )
            curves.append(curve)
            if not rows or len(rows[0][0]) < 2:
                issue(
                    "curve_header", "Deux colonnes Term et Market Rate sont requises.", sheet.title
                )
                continue
            if [_label(c.value) for c in rows[0][0][:2]] != ["TERM", "MARKET RATE"]:
                issue(
                    "curve_header", "En-têtes attendus : Term et Market Rate.", sheet.title, "A1:B1"
                )
            seen_terms = set()
            for cached, source in rows[1:]:
                term_cell, value_cell = cached[:2]
                if all(c.value is None for c in source[:2]):
                    continue
                term = str(term_cell.value or "").strip()
                normal_term = _label(term)
                months = _months(term)
                fra = re.fullmatch(r"(\d+)\s*MO\s*X\s*(\d+)\s*MO", normal_term)
                hint = "À qualifier"
                if name == "EUR3" and re.fullmatch(r"[1-9]\d*\s*SF", normal_term):
                    hint = "Future Euribor 3M (taux affiché)"
                elif name == "EUR6" and fra and int(fra[2]) - int(fra[1]) == 6:
                    hint = "FRA"
                elif name == "ESTR" and (
                    months or re.fullmatch(r"[1-9]\d*\s*(WK|W|DY|D)", normal_term)
                ):
                    hint = "OIS / dépôt selon convention"
                elif name != "ESTR" and months:
                    index_months = 3 if name == "EUR3" else 6
                    if months == index_months:
                        hint = "Dépôt"
                    elif months > index_months:
                        hint = "Swap de taux"
                if hint == "À qualifier":
                    issue(
                        "unknown_term",
                        "Libellé instrument non reconnu ; aucune échéance inventée.",
                        sheet.title,
                        term_cell.coordinate,
                    )
                if normal_term in seen_terms:
                    issue(
                        "duplicate_term",
                        "Libellé de courbe dupliqué.",
                        sheet.title,
                        term_cell.coordinate,
                    )
                seen_terms.add(normal_term)
                raw = read_numeric(value_cell, source[1], sheet.title)
                is_percent = "%" in value_cell.number_format
                curve.quotes.append(
                    CurveQuote(
                        term=term,
                        term_cell=term_cell.coordinate,
                        value_cell=value_cell.coordinate,
                        raw_value=raw,
                        rate_decimal=(
                            (raw if is_percent else raw / 100) if raw is not None else None
                        ),
                        instrument_hint=hint,
                        raw_unit="excel_percent" if is_percent else "percent_points",
                    )
                )
            if not curve.quotes:
                issue("empty_curve", "Aucune cotation de courbe.", sheet.title)

        surface = CapSurfaceInput()
        if "VOL CAP" in sheets:
            sheet = sheets["VOL CAP"]
            rows = rows_for(sheet)
            fixed_columns = []
            atm_column = None
            seen_strikes = set()
            if not rows or _label(rows[0][0][0].value) != "EXPIRY":
                issue("cap_header", "La première colonne doit être Expiry.", sheet.title, "A1")
            if rows:
                for column, cell in enumerate(rows[0][0][1:], start=1):
                    if cell.value is None and all(
                        source[column].value is None for _, source in rows[1:]
                    ):
                        continue
                    label = str(cell.value).strip()
                    if _label(cell.value) == "ATM":
                        if atm_column is not None:
                            issue(
                                "duplicate_atm",
                                "Plusieurs colonnes ATM.",
                                sheet.title,
                                cell.coordinate,
                            )
                        else:
                            atm_column = column
                        continue
                    if isinstance(cell.value, str) and label.endswith("%"):
                        parsed = _number(label[:-1])
                        strike = parsed / 100 if parsed is not None else None
                    elif isinstance(cell.value, Real) and not isinstance(cell.value, bool):
                        # Numeric template headers are rates already expressed as decimals.
                        strike = _number(cell.value)
                    else:
                        strike = None
                    if strike is None:
                        issue(
                            "invalid_strike",
                            "Strike attendu en pourcentage explicite ou en taux décimal numérique.",
                            sheet.title,
                            cell.coordinate,
                        )
                        continue
                    if strike in seen_strikes:
                        issue("duplicate_strike", "Strike dupliqué.", sheet.title, cell.coordinate)
                    seen_strikes.add(strike)
                    fixed_columns.append(column)
                    surface.strikes.append(
                        CapStrike(label=label, header_cell=cell.coordinate, strike_decimal=strike)
                    )
                if not fixed_columns:
                    issue("missing_strikes", "Aucun strike fixe valide.", sheet.title)
                if atm_column is None:
                    issue(
                        "missing_atm",
                        "Colonne ATM absente ; l'ancrage ATM sera indisponible.",
                        sheet.title,
                        severity="warning",
                    )

                def cap_quote(cached, source, column):
                    cell = cached[column]
                    raw = read_numeric(cell, source[column], sheet.title)
                    normal = raw / 10000 if raw is not None else None
                    if raw is not None and raw < 0:
                        issue(
                            "negative_vol",
                            "Une volatilité ne peut pas être négative.",
                            sheet.title,
                            cell.coordinate,
                        )
                        normal = None
                    if "%" in cell.number_format:
                        issue(
                            "vol_unit",
                            "Cellule au format % incompatible avec le template de vols en bp ; vérifier l'unité.",
                            sheet.title,
                            cell.coordinate,
                        )
                        normal = None
                    return CapQuote(cell=cell.coordinate, raw_value=raw, normal_vol=normal)

                seen_expiries = set()
                for cached, source in rows[1:]:
                    if all(c.value is None for c in source):
                        continue
                    months = _months(cached[0].value)
                    if months is None:
                        issue(
                            "invalid_expiry",
                            "Maturité attendue en mois ou années, par exemple 18Mo ou 2Yr.",
                            sheet.title,
                            cached[0].coordinate,
                        )
                        continue
                    if months in seen_expiries:
                        issue(
                            "duplicate_expiry",
                            "Maturité dupliquée.",
                            sheet.title,
                            cached[0].coordinate,
                        )
                    seen_expiries.add(months)
                    tenor = (
                        ("3M" if months <= 24 else "6M")
                        if quote_tenor_policy == "market"
                        else quote_tenor_policy
                    )
                    surface.rows.append(
                        CapRow(
                            expiry=str(cached[0].value),
                            expiry_cell=cached[0].coordinate,
                            expiry_months=months,
                            quote_index_tenor=tenor,
                            atm=(
                                cap_quote(cached, source, atm_column)
                                if atm_column is not None
                                else None
                            ),
                            quotes=[cap_quote(cached, source, col) for col in fixed_columns],
                        )
                    )
                if not surface.rows:
                    issue("empty_caps", "Aucune maturité de cap valide.", sheet.title)

        issue(
            "template_conventions",
            "Convention de lecture du template : volatilités normales de caps en bp/√an, strikes absolus et taux de courbes en %. Ces conventions ne sont pas documentées dans les cellules ; les confirmer auprès du fournisseur avant valorisation.",
            severity="warning",
        )
        issue(
            "cap_not_caplet",
            "Les lignes sont des maturités de caps entiers. Elles ne sont ni des volatilités de caplets ni des volatilités locales. ATM est un strike à calculer à partir du calendrier et des courbes, pas K = 0.",
            severity="info",
        )
        if quote_tenor_policy == "market":
            issue(
                "tenor_transition",
                "Convention héritée du pricer : Euribor 3M jusqu'à 2 ans inclus, puis 6M. Deux familles d'indices à traiter explicitement lors du stripping.",
                severity="warning",
            )
        if any("Future" in q.instrument_hint for c in curves for q in c.quotes):
            issue(
                "futures_convention",
                "Les labels SF sont conservés : dates de contrats et ajustement de convexité seront nécessaires pour construire la courbe EUR3.",
                sheet="EUR3",
                severity="info",
            )

        return ThomasCapsMarketData(
            source_filename=filename,
            valuation_date=selected_date,
            date_source="user" if valuation_date else "filename",
            quote_tenor_policy=quote_tenor_policy,
            status=(
                "needs_attention"
                if any(i.severity == "error" for i in issues)
                else "ready_for_curves"
            ),
            sheets=inventory,
            curves=curves,
            cap_surface=surface,
            issues=issues,
            counts=DataCounts(
                curve_quotes=sum(q.rate_decimal is not None for c in curves for q in c.quotes),
                expiries=len(surface.rows),
                fixed_strikes=len(surface.strikes),
                cap_quotes=sum(
                    q.normal_vol is not None for row in surface.rows for q in row.quotes
                ),
                atm_quotes=sum(
                    row.atm is not None and row.atm.normal_vol is not None for row in surface.rows
                ),
            ),
        )
    except ParseError as exc:
        raise ValueError("Fichier Excel invalide : contenu XML illisible.") from exc
    finally:
        values.close()
        formulas.close()
