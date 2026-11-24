"""
Excel template loader for the MD.xlsx multi-sheet format.

Parses the multi-sheet template used by the xva-pricer. The loader is deliberately
sheet-driven: it imports only datasets represented by sheets present in the workbook.
"""

import re
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Literal, TypedDict

import pandas as pd

from config_swaption import SWAPTION_MARKET_CONVENTIONS
from curves.helpers import index_tenor_months, tenor_to_years
from volatility.vol_convention import get_vol_convention

from .curve_quote_typing import (
    OVERNIGHT_ANCHOR_TENOR,
    curve_quote_metadata,
    resolve_curve_quote_convention,
    typed_point,
)
from .futures_parser import (
    futures_code_to_imm_date,
    sf_to_imm_date,
)
from .schemas import (
    CapFloorVolData,
    CurveInstrumentType,
    ForwardMarketData,
    FuturesDataPoint,
    FuturesRateBasis,
    MarketDataPoint,
    OISMarketData,
    SwaptionVolData,
    SwaptionVolDataPoint,
)
from .swaption_contract import (
    BASIS_POINT,
    cell_address,
    parse_strike_headers,
    parse_vol_cell,
    swaption_convention_id,
    validate_swaption_quotes,
)
from .tenor_normalizer import classify_forward_instrument, normalize_tenor

SheetType = Literal["ois", "forward", "capfloor", "swaption", "swaption_otm"]


class SheetConfig(TypedDict, total=False):
    """Static parsing config for one Excel sheet."""

    type: SheetType
    currency: str
    index_tenor: str


class ParsedTemplateData(TypedDict):
    """Parsed market-data buckets keyed by dataset type and currency/tenor."""

    ois: dict[str, OISMarketData]
    forward: dict[str, ForwardMarketData]
    capfloor: dict[str, CapFloorVolData]
    swaption: dict[str, SwaptionVolData]
    swaption_otm: dict[str, SwaptionVolData]
    warnings: list[str]


# Sheet configuration: sheet_name -> parsing config
_EUR_SWAPTION_KEY = SWAPTION_MARKET_CONVENTIONS["EUR"].default_index_tenor
SHEET_CONFIG: dict[str, SheetConfig] = {
    "ESTR": {"type": "ois", "currency": "EUR"},
    "EUR1": {"type": "forward", "currency": "EUR", "index_tenor": "1M"},
    "EUR3": {"type": "forward", "currency": "EUR", "index_tenor": "3M"},
    "EUR6": {"type": "forward", "currency": "EUR", "index_tenor": "6M"},
    "VOL CAP": {"type": "capfloor", "currency": "EUR"},
    "VOL SWAPTION ATM": {"type": "swaption", "currency": "EUR", "index_tenor": _EUR_SWAPTION_KEY},
    "VOL SWAPTION OTM": {
        "type": "swaption_otm",
        "currency": "EUR",
        "index_tenor": _EUR_SWAPTION_KEY,
    },
    "SOFR": {"type": "ois", "currency": "USD"},
    "SONIA": {"type": "ois", "currency": "GBP"},
}

DEFAULT_SWAPTION_INDEX_TENOR_BY_CURRENCY = {
    currency: profile.default_index_tenor
    for currency, profile in SWAPTION_MARKET_CONVENTIONS.items()
}

SWAPTION_SHEET_PATTERN = re.compile(
    r"^VOL\s+SWAPTION\s+(ATM|OTM)(?:\s+([A-Z]{3})(?:\s+([A-Z0-9]+))?)?$"
)

_EUR_ESTR_CASH_RATE_LABEL = "ESTR ON"
_EURIBOR_1M_FIXING_LABEL = "E1M"
_RATE_BASIS_COLUMN = "RATE BASIS"
_QUOTE_TENOR_COLUMN = "QUOTE TENOR"
_FUTURES_PRICE_BASIS = "futures_price"
_MIN_PLAUSIBLE_FUTURES_PRICE = 0.0
# Broad sanity guardrail: prices above 100 represent negative implied rates and are valid.
_MAX_PLAUSIBLE_FUTURES_PRICE = 200.0


def _normalize_sheet_name(sheet_name: str) -> str:
    return re.sub(r"\s+", " ", sheet_name.strip().upper())


def _normalize_curve_source_label(label: str) -> str:
    """Normalize spacing/case for explicit non-tenor market-data labels."""
    return re.sub(r"\s+", " ", label.strip().upper())


def _optional_column(columns: list[object], normalized_name: str) -> object | None:
    """Return an optional column by normalized label, rejecting ambiguous duplicates."""
    matches = [
        column
        for column in columns
        if _normalize_curve_source_label(str(column)) == normalized_name
    ]
    if len(matches) > 1:
        raise ValueError(f"Duplicate '{normalized_name.title()}' columns")
    return matches[0] if matches else None


def _sheet_rate(raw_rate: object) -> float | None:
    """Return a sheet rate as a decimal, or ``None`` for a blank or non-numeric cell."""
    if pd.isna(raw_rate):
        return None
    if str(raw_rate).strip().upper() in ("N/A", "NA", "#N/A", "-", ""):
        return None
    try:
        return float(raw_rate) / 100.0
    except (ValueError, TypeError):
        return None


def _future_rate_and_basis(
    raw_rate: object,
    raw_basis: object,
    *,
    has_basis_column: bool,
    source_label: str,
) -> tuple[float, FuturesRateBasis]:
    """Parse one futures quote without applying any convexity adjustment in the loader."""
    if has_basis_column:
        if pd.isna(raw_basis) or not str(raw_basis).strip():
            raise ValueError(f"Future '{source_label}' requires a non-blank Rate Basis")
        normalized_basis = str(raw_basis).strip().lower()
        if normalized_basis == _FUTURES_PRICE_BASIS:
            try:
                price = float(raw_rate)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Future '{source_label}' has an invalid futures price") from exc
            if not isfinite(price) or not (
                _MIN_PLAUSIBLE_FUTURES_PRICE <= price <= _MAX_PLAUSIBLE_FUTURES_PRICE
            ):
                raise ValueError(
                    f"Future '{source_label}' futures price must be finite and within the "
                    "plausibility range [0, 200]"
                )
            return (100.0 - price) / 100.0, FuturesRateBasis.FUTURES_RATE
        try:
            basis = FuturesRateBasis(normalized_basis)
        except ValueError as exc:
            raise ValueError(
                f"Future '{source_label}' has unsupported Rate Basis '{raw_basis}'"
            ) from exc
    else:
        basis = FuturesRateBasis.CONVEXITY_ADJUSTED_RATE

    rate = _sheet_rate(raw_rate)
    if rate is None:
        raise ValueError(f"Future '{source_label}' has an invalid Market Rate")
    return rate, basis


def _typed_row(
    currency: str,
    curve_key: str,
    instrument_type: CurveInstrumentType,
    *,
    tenor: str,
    rate: float,
    source_label: str,
) -> MarketDataPoint:
    """Type one sheet row: the canonical table decides its convention or rejects it."""
    try:
        return typed_point(
            currency,
            curve_key,
            instrument_type,
            tenor=tenor,
            rate=rate,
            source_label=source_label,
        )
    except ValueError as error:
        raise ValueError(f"{currency} {curve_key} sheet, row '{source_label}': {error}") from error


def _has_eur_context(normalized_sheets: set[str]) -> bool:
    return bool({"ESTR", "EUR1", "EUR3", "EUR6", "VOL CAP"}.intersection(normalized_sheets))


def _has_non_eur_context(normalized_sheets: set[str]) -> bool:
    return bool(
        {"SOFR", "SONIA"}.intersection(normalized_sheets)
        or any(
            name.startswith("VOL SWAPTION ATM USD")
            or name.startswith("VOL SWAPTION OTM USD")
            or name.startswith("VOL SWAPTION ATM GBP")
            or name.startswith("VOL SWAPTION OTM GBP")
            for name in normalized_sheets
        )
    )


def _resolve_sheet_config(
    sheet_name: str,
    normalized_sheets: set[str],
    warnings: list[str],
) -> SheetConfig | None:
    normalized = _normalize_sheet_name(sheet_name)
    static_config = SHEET_CONFIG.get(normalized)
    if static_config is not None:
        if normalized in {"VOL SWAPTION ATM", "VOL SWAPTION OTM"} and (
            not _has_eur_context(normalized_sheets) and _has_non_eur_context(normalized_sheets)
        ):
            warnings.append(
                f"Skipped ambiguous unqualified sheet '{sheet_name}'. "
                "Use an explicit suffix such as 'VOL SWAPTION ATM USD SOFR'."
            )
            return None
        return static_config

    match = SWAPTION_SHEET_PATTERN.match(normalized)
    if match is None:
        return None

    smile_type, currency, explicit_tenor = match.groups()
    if currency is None:
        return None

    currency = currency.upper()
    if currency not in DEFAULT_SWAPTION_INDEX_TENOR_BY_CURRENCY:
        warnings.append(f"Skipped swaption sheet '{sheet_name}': unsupported currency {currency}.")
        return None

    index_tenor = (explicit_tenor or DEFAULT_SWAPTION_INDEX_TENOR_BY_CURRENCY[currency]).upper()
    sheet_type: SheetType = "swaption" if smile_type == "ATM" else "swaption_otm"
    return {"type": sheet_type, "currency": currency, "index_tenor": index_tenor}


class ExcelTemplateLoader:
    """Loader for the multi-sheet Excel market data template."""

    def load_template(
        self, file_path: str | Path | bytes, valuation_date: date
    ) -> ParsedTemplateData:
        """
        Load and parse all sheets from the Excel template.

        Args:
            file_path: Path to the Excel file or bytes content
            valuation_date: Valuation date for all market data

        Returns:
            Dictionary with parsed market data organized by type:
            {
                "ois": {"EUR": OISMarketData, "USD": ..., "GBP": ...},
                "forward": {"EUR_1M": ForwardMarketData, "EUR_3M": ..., "EUR_6M": ...},
                "capfloor": {"EUR": CapFloorVolData},
                "swaption": {"EUR_6M": SwaptionVolData},
            }
        """
        import io

        if isinstance(file_path, bytes):
            file_path = io.BytesIO(file_path)

        xls = pd.ExcelFile(file_path)
        available_sheets = xls.sheet_names

        result: ParsedTemplateData = {
            "ois": {},
            "forward": {},
            "capfloor": {},
            "swaption": {},
            "swaption_otm": {},
            "warnings": [],
        }

        normalized_sheets = {_normalize_sheet_name(sheet_name) for sheet_name in available_sheets}
        for sheet_name in available_sheets:
            config = _resolve_sheet_config(
                sheet_name,
                normalized_sheets,
                result["warnings"],
            )
            if config is None:
                continue

            df = pd.read_excel(xls, sheet_name=sheet_name)
            sheet_type = config["type"]
            currency = config["currency"]

            if sheet_type == "ois":
                data = self._parse_ois_sheet(df, currency, valuation_date)
                result["ois"][currency] = data

            elif sheet_type == "forward":
                index_tenor = config["index_tenor"]
                data = self._parse_forward_sheet(df, currency, valuation_date, index_tenor)
                result["warnings"].extend(data.import_warnings)
                key = f"{currency}_{index_tenor}"
                result["forward"][key] = data

            elif sheet_type == "capfloor":
                data = self._parse_capfloor_sheet(df, currency, valuation_date)
                result["warnings"].extend(data.import_warnings)
                result["capfloor"][currency] = data

            elif sheet_type == "swaption":
                index_tenor = config.get("index_tenor", "6M")
                data = self._parse_swaption_sheet(
                    df, currency, valuation_date, index_tenor=index_tenor, sheet_name=sheet_name
                )
                result["warnings"].extend(validate_swaption_quotes(data, "atm"))
                key = f"{currency}_{index_tenor}"
                if key in result["swaption"]:
                    result["warnings"].append(
                        f"Duplicate swaption ATM dataset {key}; last sheet wins."
                    )
                result["swaption"][key] = data

            elif sheet_type == "swaption_otm":
                index_tenor = config.get("index_tenor", "6M")
                data = self._parse_swaption_otm_sheet(
                    df, currency, valuation_date, index_tenor=index_tenor, sheet_name=sheet_name
                )
                result["warnings"].extend(validate_swaption_quotes(data, "smile"))
                key = f"{currency}_{index_tenor}"
                if key in result["swaption_otm"]:
                    result["warnings"].append(
                        f"Duplicate swaption OTM dataset {key}; last sheet wins."
                    )
                result["swaption_otm"][key] = data

        return result

    def _parse_ois_sheet(
        self, df: pd.DataFrame, currency: str, valuation_date: date
    ) -> OISMarketData:
        """
        Parse an OIS sheet (ESTR, SOFR, SONIA).

        Columns: Term, Market Rate. Rates are in percentage (1.9332 = 1.9332%).
        The ``ESTR ON`` label and a one-day SONIA quote are overnight deposit anchors; every other
        row is an outright OIS swap of one week or more. Each quote takes its convention from the
        canonical table (``curve_quote_typing``), and a row the table cannot place is an error.
        """
        deposits = []
        swaps = []

        for _, row in df.iterrows():
            term = str(row["Term"]).strip()
            rate = _sheet_rate(row["Market Rate"])
            if rate is None:
                continue

            if (
                currency == "EUR"
                and _normalize_curve_source_label(term) == _EUR_ESTR_CASH_RATE_LABEL
            ):
                tenor = OVERNIGHT_ANCHOR_TENOR
            else:
                tenor = normalize_tenor(term)
            instrument_type = (
                CurveInstrumentType.DEPOSIT
                if tenor == OVERNIGHT_ANCHOR_TENOR
                else CurveInstrumentType.OIS_SWAP
            )
            point = _typed_row(
                currency, "OIS", instrument_type, tenor=tenor, rate=rate, source_label=term
            )
            if instrument_type == CurveInstrumentType.DEPOSIT:
                deposits.append(point)
            else:
                swaps.append(point)

        return OISMarketData(
            currency=currency,
            valuation_date=valuation_date,
            fixings=[],
            deposits=deposits,
            swaps=swaps,
        )

    def _parse_forward_sheet(
        self,
        df: pd.DataFrame,
        currency: str,
        valuation_date: date,
        index_tenor: str,
    ) -> ForwardMarketData:
        """
        Parse a forward curve sheet (EUR1, EUR3, EUR6).

        Columns: Term, Market Rate. Rates are in percentage.
        Handles:
        - Futures codes (ERF26..., N SF): the futures strip of the EURIBOR 3M curve, bootstrapped
          with ``FuturesRateHelper`` on IMM dates. ``Rate Basis`` states whether a quote is a raw
          futures rate, a convexity-adjusted rate, or a futures price converted once on import
        - Bloomberg FRA tickers and ``X MO x Y MO`` rows: the FRAs of the EURIBOR 6M curve
        - Regular tenors: the deposit of the curve's own tenor, or a swap
        Each quote takes its convention from the canonical table (``curve_quote_typing``): an
        instrument that the curve does not admit, such as a future on the 6M curve or a deposit of
        another tenor, is an error, never an untyped point.
        """
        index_months = index_tenor_months(index_tenor)
        fixings = []
        deposits = []
        fras = []
        futures = []
        swaps = []
        rate_basis_column = _optional_column(list(df.columns), _RATE_BASIS_COLUMN)
        import_warnings: list[str] = []
        used_legacy_future_basis = False

        for _, row in df.iterrows():
            term = str(row["Term"]).strip()
            raw_rate = row["Market Rate"]
            classification = classify_forward_instrument(term, index_tenor=index_tenor)
            inst_type = classification["type"]
            is_future = inst_type in ("future", "sf")
            is_blank_rate = pd.isna(raw_rate) or str(raw_rate).strip().upper() in (
                "N/A",
                "NA",
                "#N/A",
                "-",
                "",
            )
            if is_future and rate_basis_column is not None:
                rate, rate_basis = _future_rate_and_basis(
                    raw_rate,
                    row[rate_basis_column],
                    has_basis_column=True,
                    source_label=term,
                )
            elif is_blank_rate:
                continue
            elif is_future:
                rate, rate_basis = _future_rate_and_basis(
                    raw_rate,
                    None,
                    has_basis_column=False,
                    source_label=term,
                )
                used_legacy_future_basis = True
            else:
                rate = _sheet_rate(raw_rate)
                if rate is None:
                    continue

            if _normalize_curve_source_label(term) == _EURIBOR_1M_FIXING_LABEL:
                fixings.append(
                    _typed_row(
                        currency,
                        index_tenor,
                        CurveInstrumentType.FIXING,
                        tenor="1M",
                        rate=rate,
                        source_label=term,
                    )
                )
                continue

            if inst_type in ("future", "sf"):
                if inst_type == "future":
                    imm_date = futures_code_to_imm_date(classification["code"])
                else:
                    imm_date = sf_to_imm_date(classification["n"], valuation_date, index_months)
                convention = resolve_curve_quote_convention(
                    currency, index_tenor, CurveInstrumentType.FUTURE
                )
                futures.append(
                    FuturesDataPoint(
                        imm_date=imm_date,
                        rate=rate,
                        rate_basis=rate_basis,
                        quote_metadata=curve_quote_metadata(
                            term, CurveInstrumentType.FUTURE, convention
                        ),
                    )
                )
            elif inst_type == "fra":
                fras.append(
                    _typed_row(
                        currency,
                        index_tenor,
                        CurveInstrumentType.FRA,
                        tenor=classification["fra_tenor"],
                        rate=rate,
                        source_label=term,
                    )
                )
            elif inst_type == "deposit":
                deposits.append(
                    _typed_row(
                        currency,
                        index_tenor,
                        CurveInstrumentType.DEPOSIT,
                        tenor=classification["tenor"],
                        rate=rate,
                        source_label=term,
                    )
                )
            else:
                swaps.append(
                    _typed_row(
                        currency,
                        index_tenor,
                        CurveInstrumentType.IRS_SWAP,
                        tenor=classification["tenor"],
                        rate=rate,
                        source_label=term,
                    )
                )

        if used_legacy_future_basis:
            import_warnings.append(
                f"{currency}_{index_tenor} futures have no Rate Basis column; legacy assumption "
                "'convexity_adjusted_rate' was applied. The source did not explicitly declare "
                "the quote basis."
            )

        return ForwardMarketData(
            currency=currency,
            valuation_date=valuation_date,
            index_tenor=index_tenor,
            fixings=fixings,
            deposits=deposits,
            fras=fras,
            futures=futures,
            swaps=swaps,
            import_warnings=import_warnings,
        )

    def _parse_capfloor_sheet(
        self,
        df: pd.DataFrame,
        currency: str,
        valuation_date: date,
    ) -> CapFloorVolData:
        """
        Parse cap/floor volatility sheet (VOL CAP).

        Auto-detects old vs new format:
        - Old format: strike columns are numeric (0, 0.005, 0.01, ...)
        - New format: strike columns contain '%' (e.g., '-1,50%', '0,00%', '5,00%')
        """
        # Auto-detect: if any non-Expiry/ATM column contains '%', it's the new format
        all_cols = list(df.columns)
        quote_tenor_column = _optional_column(all_cols, _QUOTE_TENOR_COLUMN)
        non_meta_cols = [
            column
            for column in all_cols
            if str(column) not in ("Expiry", "ATM") and column != quote_tenor_column
        ]
        is_new_format = any("%" in str(c) for c in non_meta_cols)

        if is_new_format:
            return self._parse_capfloor_new_format(df, currency, valuation_date)
        else:
            return self._parse_capfloor_old_format(df, currency, valuation_date)

    def _parse_capfloor_old_format(
        self,
        df: pd.DataFrame,
        currency: str,
        valuation_date: date,
    ) -> CapFloorVolData:
        """
        Parse old cap/floor format.

        Columns: Expiry, ATM, 0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05
        Vols are in bps. ATM is treated as strike 0.0.
        """
        df = df.dropna(subset=["Expiry"]).copy()

        all_cols = list(df.columns)
        quote_tenor_column = _optional_column(all_cols, _QUOTE_TENOR_COLUMN)
        strike_cols = [
            column for column in all_cols if column not in ("Expiry", "ATM", quote_tenor_column)
        ]
        strikes = [float(c) for c in strike_cols]

        expiries = []
        for exp in df["Expiry"]:
            expiries.append(normalize_tenor(str(exp)))
        quote_tenors, import_warnings = self._resolve_capfloor_quote_tenors(
            df, currency, expiries, quote_tenor_column
        )

        volatilities = []
        for _, row in df.iterrows():
            vol_row = []
            for col in strike_cols:
                val = row[col]
                if pd.notna(val):
                    vol_row.append(float(val) / 10000.0)
                else:
                    vol_row.append(0.0)
            volatilities.append(vol_row)

        return CapFloorVolData(
            currency=currency,
            valuation_date=valuation_date,
            expiries=expiries,
            strikes=strikes,
            volatilities=volatilities,
            quote_tenors=quote_tenors,
            import_warnings=import_warnings,
        )

    def _parse_capfloor_new_format(
        self,
        df: pd.DataFrame,
        currency: str,
        valuation_date: date,
    ) -> CapFloorVolData:
        """
        Parse new cap/floor format.

        Strike columns: '-1,50%', '-1,00%', '0,00%', '0,50%', ... (absolute % with European comma)
        ATM column: separate ATM vol in bps
        Expiries: 16 tenors (1Yr, 18Mo, 2Yr, ..., 30Yr)
        Vols: in bps (normal)
        """
        df = df.dropna(subset=["Expiry"]).copy()

        all_cols = list(df.columns)
        quote_tenor_column = _optional_column(all_cols, _QUOTE_TENOR_COLUMN)

        # Identify strike columns (contain '%')
        strike_cols = [c for c in all_cols if "%" in str(c)]

        # Parse strike values: strip '%', replace ',' with '.', convert to decimal
        strikes = []
        for col in strike_cols:
            s = str(col).strip().replace("%", "").replace(",", ".").strip()
            strikes.append(float(s) / 100.0)  # percentage to decimal

        # Normalize expiries
        expiries = []
        for exp in df["Expiry"]:
            expiries.append(normalize_tenor(str(exp)))
        quote_tenors, import_warnings = self._resolve_capfloor_quote_tenors(
            df, currency, expiries, quote_tenor_column
        )

        # Build volatilities matrix [expiry][strike], converting bps to decimal
        volatilities = []
        for _, row in df.iterrows():
            vol_row = []
            for col in strike_cols:
                val = row[col]
                if pd.notna(val):
                    vol_row.append(float(val) / 10000.0)  # bps to decimal
                else:
                    vol_row.append(0.0)
            volatilities.append(vol_row)

        # Extract ATM vols if column exists
        atm_vols = None
        if "ATM" in all_cols:
            atm_vols = []
            for _, row in df.iterrows():
                val = row["ATM"]
                if pd.notna(val):
                    atm_vols.append(float(val) / 10000.0)  # bps to decimal
                else:
                    atm_vols.append(0.0)

        return CapFloorVolData(
            currency=currency,
            valuation_date=valuation_date,
            expiries=expiries,
            strikes=strikes,
            volatilities=volatilities,
            atm_vols=atm_vols,
            quote_tenors=quote_tenors,
            import_warnings=import_warnings,
        )

    @staticmethod
    def _resolve_capfloor_quote_tenors(
        df: pd.DataFrame,
        currency: str,
        expiries: list[str],
        quote_tenor_column: object | None,
    ) -> tuple[list[str], list[str]]:
        """Read explicit native tenors, or retain the legacy maturity-based assumption."""
        if quote_tenor_column is None:
            return ExcelTemplateLoader._capfloor_quote_tenors(currency, expiries), [
                (
                    "VOL CAP has no Quote Tenor column; the legacy implicit composite tenor "
                    "split was inferred from the cap-expiry convention."
                )
            ]

        quote_tenors: list[str] = []
        for row_index, raw_tenor in enumerate(df[quote_tenor_column], start=2):
            if pd.isna(raw_tenor) or not str(raw_tenor).strip():
                raise ValueError(f"VOL CAP row {row_index} requires a non-blank Quote Tenor")
            tenor = normalize_tenor(str(raw_tenor))
            if currency == "EUR" and tenor not in ("3M", "6M"):
                raise ValueError(
                    f"VOL CAP row {row_index} Quote Tenor '{raw_tenor}' is unsupported for EUR; "
                    "use 3M or 6M"
                )
            quote_tenors.append(tenor)
        return quote_tenors, []

    @staticmethod
    def _capfloor_quote_tenors(currency: str, expiries: list[str]) -> list[str]:
        """Resolve the native quote tenor of each cap expiry from market convention."""
        convention = get_vol_convention(currency)
        return [convention.get_quoting_tenor(tenor_to_years(expiry)) for expiry in expiries]

    def _parse_swaption_sheet(
        self,
        df: pd.DataFrame,
        currency: str,
        valuation_date: date,
        index_tenor: str = "6M",
        sheet_name: str = "VOL SWAPTION ATM",
    ) -> SwaptionVolData:
        """
        Parse swaption volatility sheet (VOL SWAPTION ATM).

        Columns: Expiry, 1Yr, 2Yr, 3Yr, 4Y, 5Yr, ... 30Yr
        Vols are in bps (normal volatilities).
        All points are ATM (strike=None). A non-numeric or non-positive cell is refused with its
        address; an empty cell is a missing quote.
        """
        columns = list(df.columns)
        tenor_cols = [c for c in columns if c != "Expiry"]
        convention_id = swaption_convention_id(currency)

        data_points = []
        for row_index, (_, row) in enumerate(df.iterrows()):
            expiry_raw = str(row["Expiry"]).strip()
            expiry = normalize_tenor(expiry_raw)

            for col in tenor_cols:
                address = cell_address(sheet_name, columns.index(col) + 1, row_index + 2)
                vol_bps = parse_vol_cell(row[col], address)
                if vol_bps is None:
                    continue
                data_points.append(
                    SwaptionVolDataPoint(
                        expiry=expiry,
                        tenor=normalize_tenor(str(col)),
                        strike=None,  # ATM
                        volatility=vol_bps * BASIS_POINT,
                        vol_type="normal",
                        raw_value=vol_bps,
                        raw_unit="bp",
                        convention_id=convention_id,
                        source_cell=address,
                    )
                )

        return SwaptionVolData(
            currency=currency,
            index_tenor=index_tenor,
            valuation_date=valuation_date,
            data_points=data_points,
        )

    def _parse_swaption_otm_sheet(
        self,
        df: pd.DataFrame,
        currency: str,
        valuation_date: date,
        index_tenor: str = "6M",
        sheet_name: str = "VOL SWAPTION OTM",
    ) -> SwaptionVolData:
        """
        Parse swaption OTM volatility sheet (VOL SWAPTION OTM).

        Columns: Concat, KEY OPTION TENORS, KEY SWAP TENORS, -200bps, -150bps, ..., 200bps
        Vols are in bps (normal volatilities).
        Strike = spread to ATM in decimal (e.g., -200bps -> -0.02). The strike headers must be
        strictly increasing and unique; a header, cell or vol that breaks the contract is refused
        with its address, an empty cell is a missing quote.
        """
        meta_cols = {"Concat", "KEY OPTION TENORS", "KEY SWAP TENORS"}
        columns = list(df.columns)
        strike_positions = [
            index for index, column in enumerate(columns) if str(column) not in meta_cols
        ]
        # Pandas names empty header cells 'Unnamed: n': blank filler columns are not strikes.
        strike_positions = [
            index
            for index in strike_positions
            if not (str(columns[index]).startswith("Unnamed") and df.iloc[:, index].isna().all())
        ]
        headers = parse_strike_headers(
            [(columns[index], index + 1) for index in strike_positions], sheet=sheet_name
        )
        convention_id = swaption_convention_id(currency)

        data_points = []
        for row_index, (_, row) in enumerate(df.iterrows()):
            expiry_raw = row.get("KEY OPTION TENORS")
            tenor_raw = row.get("KEY SWAP TENORS")
            if pd.isna(expiry_raw) or pd.isna(tenor_raw):
                continue

            expiry = normalize_tenor(str(expiry_raw).strip())
            tenor = normalize_tenor(str(tenor_raw).strip())

            for column, position, spread in headers:
                address = cell_address(sheet_name, position, row_index + 2)
                vol_bps = parse_vol_cell(row[column], address)
                if vol_bps is None:
                    continue
                data_points.append(
                    SwaptionVolDataPoint(
                        expiry=expiry,
                        tenor=tenor,
                        strike=spread,
                        volatility=vol_bps * BASIS_POINT,
                        vol_type="normal",
                        raw_value=vol_bps,
                        raw_unit="bp",
                        convention_id=convention_id,
                        source_cell=address,
                    )
                )

        return SwaptionVolData(
            currency=currency,
            index_tenor=index_tenor,
            valuation_date=valuation_date,
            data_points=data_points,
        )
