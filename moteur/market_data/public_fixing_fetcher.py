"""
Public fixing source registry and CSV fetch/parsing helpers.
"""

import csv
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date, timedelta
from io import StringIO
from typing import Callable, Iterable, Optional
from urllib.parse import urlencode

import httpx

from market_data.schemas import FixingDataPoint


@dataclass(frozen=True)
class PublicFixingSource:
    """Configuration for one public fixing time series."""

    index_key: str
    label: str
    currency: str
    index_tenor: str
    provider: str
    series_id: str
    source_url: str
    default_start_date: date
    parser: str
    stale_after_days: int = 5


_ECB_ESTR_URL = "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT"
_BUNDESBANK_URL = "https://api.statistiken.bundesbank.de/rest/data/BBIG1/{series_id}"
_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_DEFAULT_FETCH_TIMEOUT = httpx.Timeout(30.0, connect=10.0, read=30.0)
_FETCH_TIMEOUT_ATTEMPTS = 2
_FETCH_WINDOW_DAYS_BY_PROVIDER = {
    "FRED": 366,
    "ECB": 5 * 366,
    "Bundesbank": 5 * 366,
}
_WINDOWS_SOCKET_ACCESS_DENIED = 10013


class PublicFixingFetchError(Exception):
    """Actionable error raised when a public fixing source cannot be reached."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


PUBLIC_FIXING_SOURCES: dict[str, PublicFixingSource] = {
    "EUR_OIS": PublicFixingSource(
        index_key="EUR_OIS",
        label="ESTR",
        currency="EUR",
        index_tenor="OIS",
        provider="ECB",
        series_id="EST.B.EU000A2X2A25.WT",
        source_url=_ECB_ESTR_URL,
        default_start_date=date(2019, 10, 1),
        parser="ecb",
    ),
    "EUR_1M": PublicFixingSource(
        index_key="EUR_1M",
        label="EURIBOR 1M",
        currency="EUR",
        index_tenor="1M",
        provider="Bundesbank",
        series_id="D.D0.EUR.MMKT.EURIBOR.M01.BID._Z",
        source_url=_BUNDESBANK_URL.format(series_id="D.D0.EUR.MMKT.EURIBOR.M01.BID._Z"),
        default_start_date=date(1998, 12, 30),
        parser="bundesbank",
    ),
    "EUR_3M": PublicFixingSource(
        index_key="EUR_3M",
        label="EURIBOR 3M",
        currency="EUR",
        index_tenor="3M",
        provider="Bundesbank",
        series_id="D.D0.EUR.MMKT.EURIBOR.M03.BID._Z",
        source_url=_BUNDESBANK_URL.format(series_id="D.D0.EUR.MMKT.EURIBOR.M03.BID._Z"),
        default_start_date=date(1998, 12, 30),
        parser="bundesbank",
    ),
    "EUR_6M": PublicFixingSource(
        index_key="EUR_6M",
        label="EURIBOR 6M",
        currency="EUR",
        index_tenor="6M",
        provider="Bundesbank",
        series_id="D.D0.EUR.MMKT.EURIBOR.M06.BID._Z",
        source_url=_BUNDESBANK_URL.format(series_id="D.D0.EUR.MMKT.EURIBOR.M06.BID._Z"),
        default_start_date=date(1998, 12, 30),
        parser="bundesbank",
    ),
    "USD_OIS": PublicFixingSource(
        index_key="USD_OIS",
        label="SOFR",
        currency="USD",
        index_tenor="OIS",
        provider="FRED",
        series_id="SOFR",
        source_url=_FRED_URL,
        default_start_date=date(2018, 4, 3),
        parser="fred",
    ),
    "GBP_OIS": PublicFixingSource(
        index_key="GBP_OIS",
        label="SONIA",
        currency="GBP",
        index_tenor="OIS",
        provider="FRED",
        series_id="IUDSOIA",
        source_url=_FRED_URL,
        default_start_date=date(2018, 4, 23),
        parser="fred",
    ),
}


def get_public_fixing_sources() -> dict[str, PublicFixingSource]:
    """Return configured public fixing sources keyed by store index."""
    return dict(PUBLIC_FIXING_SOURCES)


def get_public_fixing_source(index_key: str) -> PublicFixingSource:
    """Return source config for an index key."""
    return PUBLIC_FIXING_SOURCES[index_key]


def fetch_public_fixing_series(
    source: PublicFixingSource,
    start_date: date,
    end_date: date,
    *,
    client: Optional[httpx.Client] = None,
) -> list[FixingDataPoint]:
    """Fetch and parse one public fixing series over an inclusive date window."""
    if start_date > end_date:
        return []

    owns_client = client is None
    http_client = client or httpx.Client(timeout=_DEFAULT_FETCH_TIMEOUT, follow_redirects=True)
    try:
        parser = _PARSERS[source.parser]
        points: list[FixingDataPoint] = []
        for window_start, window_end in _iter_fetch_windows(source, start_date, end_date):
            points.extend(
                _fetch_public_fixing_window(
                    http_client,
                    source,
                    window_start,
                    window_end,
                    parser,
                )
            )
        return _dedupe_fixing_points(points)
    finally:
        if owns_client:
            http_client.close()


def parse_ecb_csv(text: str, source: PublicFixingSource) -> list[FixingDataPoint]:
    """Parse ECB SDMX CSV with TIME_PERIOD/OBS_VALUE percent columns."""
    values: list[tuple[date, float]] = []
    reader = csv.DictReader(StringIO(text))
    for row in reader:
        time_period = (row.get("TIME_PERIOD") or "").strip()
        raw_value = (row.get("OBS_VALUE") or "").strip()
        parsed_date = _parse_iso_date(time_period)
        parsed_value = _parse_percent(raw_value)
        if parsed_date is not None and parsed_value is not None:
            values.append((parsed_date, parsed_value))
    return _decimal_points_from_percent_values(values)


def parse_bundesbank_csv(text: str, source: PublicFixingSource) -> list[FixingDataPoint]:
    """Parse Bundesbank bbk_csv output with semicolon delimiter and decimal comma."""
    values: list[tuple[date, float]] = []
    reader = csv.reader(StringIO(text.lstrip("\ufeff")), delimiter=";")
    for row in reader:
        if len(row) < 2:
            continue
        parsed_date = _parse_iso_date(row[0].strip())
        if parsed_date is None:
            continue
        raw_value = row[1].strip().replace(",", ".")
        parsed_value = _parse_percent(raw_value)
        if parsed_value is not None:
            values.append((parsed_date, parsed_value))
    return _decimal_points_from_percent_values(values)


def parse_fred_csv(text: str, source: PublicFixingSource) -> list[FixingDataPoint]:
    """Parse FRED graph CSV with observation_date and series value columns."""
    values: list[tuple[date, float]] = []
    reader = csv.DictReader(StringIO(text))
    for row in reader:
        parsed_date = _parse_iso_date((row.get("observation_date") or "").strip())
        parsed_value = _parse_percent((row.get(source.series_id) or "").strip())
        if parsed_date is not None and parsed_value is not None:
            values.append((parsed_date, parsed_value))
    return _decimal_points_from_percent_values(values)


def _request_url(source: PublicFixingSource) -> str:
    if source.provider == "Bundesbank":
        return _BUNDESBANK_URL.format(series_id=source.series_id)
    return source.source_url


def _request_params(source: PublicFixingSource, start_date: date, end_date: date) -> dict[str, str]:
    start = start_date.isoformat()
    end = end_date.isoformat()
    if source.provider == "ECB":
        return {"startPeriod": start, "endPeriod": end, "format": "csvdata"}
    if source.provider == "Bundesbank":
        return {"startPeriod": start, "endPeriod": end, "format": "bbk_csv"}
    return {"id": source.series_id, "cosd": start, "coed": end}


def _iter_fetch_windows(
    source: PublicFixingSource,
    start_date: date,
    end_date: date,
) -> Iterable[tuple[date, date]]:
    max_days = _FETCH_WINDOW_DAYS_BY_PROVIDER.get(source.provider, 366)
    current_start = start_date
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=max_days - 1), end_date)
        yield current_start, current_end
        current_start = current_end + timedelta(days=1)


def _fetch_public_fixing_window(
    http_client: httpx.Client,
    source: PublicFixingSource,
    start_date: date,
    end_date: date,
    parser: Callable[[str, PublicFixingSource], list[FixingDataPoint]],
) -> list[FixingDataPoint]:
    last_timeout: Optional[httpx.TimeoutException] = None
    for _ in range(_FETCH_TIMEOUT_ATTEMPTS):
        try:
            response = http_client.get(
                _request_url(source),
                params=_request_params(source, start_date, end_date),
            )
            response.raise_for_status()
            return parser(response.text, source)
        except httpx.TimeoutException as exc:
            last_timeout = exc
        except httpx.RequestError as exc:
            if _is_socket_access_denied(exc):
                return _fetch_with_windows_curl(source, start_date, end_date, parser, exc)
            raise _public_fixing_request_error(source, start_date, end_date, exc) from exc
    if last_timeout is not None:
        raise TimeoutError(
            f"timed out fetching {source.label} from {start_date.isoformat()} "
            f"to {end_date.isoformat()} after {_FETCH_TIMEOUT_ATTEMPTS} attempts: {last_timeout}"
        ) from last_timeout
    return []


def _fetch_with_windows_curl(
    source: PublicFixingSource,
    start_date: date,
    end_date: date,
    parser: Callable[[str, PublicFixingSource], list[FixingDataPoint]],
    original_error: httpx.RequestError,
) -> list[FixingDataPoint]:
    """Use the Windows HTTPS client only when endpoint policy blocks Python sockets."""
    if os.name != "nt":
        raise _public_fixing_request_error(source, start_date, end_date, original_error)

    curl_path = shutil.which("curl.exe")
    if curl_path is None:
        raise PublicFixingFetchError(
            "network_access_denied",
            "Accès réseau sortant refusé par Windows (WinError 10013) et le client HTTPS "
            "Windows curl.exe est indisponible. Autorisez backend\\venv\\Scripts\\python.exe "
            "à accéder à Internet en HTTPS (TCP 443), ou configurez le proxy d'entreprise.",
        ) from original_error

    request_url = (
        f"{_request_url(source)}?{urlencode(_request_params(source, start_date, end_date))}"
    )
    try:
        completed = subprocess.run(
            [
                curl_path,
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "10",
                "--max-time",
                "35",
                request_url,
            ],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=40,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PublicFixingFetchError(
            "network_access_denied",
            f"Accès réseau refusé à Python (WinError 10013) et le fallback HTTPS Windows "
            f"n'a pas pu démarrer pour {source.label}. Vérifiez curl.exe ou le proxy.",
        ) from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"code curl {completed.returncode}"
        raise PublicFixingFetchError(
            "network_error",
            f"Le fallback HTTPS Windows n'a pas pu récupérer {source.label} depuis "
            f"{source.provider} ({detail}). Vérifiez la connexion réseau ou le proxy.",
        )
    return parser(completed.stdout, source)


def _public_fixing_request_error(
    source: PublicFixingSource,
    start_date: date,
    end_date: date,
    exc: httpx.RequestError,
) -> PublicFixingFetchError:
    if _is_socket_access_denied(exc):
        return PublicFixingFetchError(
            "network_access_denied",
            "Accès réseau sortant refusé par Windows (WinError 10013). "
            "Autorisez backend\\venv\\Scripts\\python.exe à accéder à Internet en HTTPS "
            "(TCP 443), ou configurez le proxy d'entreprise, puis réessayez.",
        )
    return PublicFixingFetchError(
        "network_error",
        f"Impossible de joindre {source.provider} pour {source.label} du "
        f"{start_date.isoformat()} au {end_date.isoformat()} ({type(exc).__name__}). "
        "Vérifiez la connexion réseau ou le proxy, puis réessayez.",
    )


def _is_socket_access_denied(exc: BaseException) -> bool:
    """Return whether an exception chain contains Windows WSAEACCES / WinError 10013."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "winerror", None) == _WINDOWS_SOCKET_ACCESS_DENIED:
            return True
        if "WinError 10013" in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def _dedupe_fixing_points(points: Iterable[FixingDataPoint]) -> list[FixingDataPoint]:
    deduped: dict[date, float] = {}
    for point in points:
        deduped[point.date] = point.rate
    return [
        FixingDataPoint(date=fixing_date, rate=deduped[fixing_date])
        for fixing_date in sorted(deduped.keys())
    ]


def _decimal_points_from_percent_values(
    values: Iterable[tuple[date, float]],
) -> list[FixingDataPoint]:
    deduped: dict[date, float] = {}
    for fixing_date, percent_value in values:
        if math.isfinite(percent_value):
            deduped[fixing_date] = percent_value / 100.0
    return [
        FixingDataPoint(date=fixing_date, rate=deduped[fixing_date])
        for fixing_date in sorted(deduped.keys())
    ]


def _parse_iso_date(value: str) -> Optional[date]:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_percent(value: str) -> Optional[float]:
    if value in {"", "."}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


_PARSERS: dict[str, Callable[[str, PublicFixingSource], list[FixingDataPoint]]] = {
    "ecb": parse_ecb_csv,
    "bundesbank": parse_bundesbank_csv,
    "fred": parse_fred_csv,
}
