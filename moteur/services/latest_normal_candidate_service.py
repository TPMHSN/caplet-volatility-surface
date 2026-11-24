"""Candidate entry points using the shared market-state and coupon lifecycle."""

from datetime import date

from pydantic import ValidationError as PydanticValidationError

from market_data.store import MarketDataStore
from services.exceptions import ServiceError
from services.latest_normal_candidate import CandidatePricingError
from services.latest_normal_candidate_schemas import CANDIDATE_MODEL, CandidatePriceResponse
from services.latest_normal_model import (
    LatestNormalValidationError,
    _prepare_latest_normal_model,
    _prepare_latest_normal_workbook,
    _price_latest_normal,
)
from services.latest_normal_schemas import LatestNormalModelRequest, LatestNormalPriceRequest


def prepare_candidate_model(
    request: LatestNormalModelRequest, market_store: MarketDataStore, curve_store: object
) -> dict:
    try:
        return _prepare_latest_normal_model(request, market_store, curve_store, use_candidate=True)
    except ServiceError:
        raise
    except ValueError as exc:
        raise LatestNormalValidationError(str(exc)) from exc
    except (RuntimeError, OverflowError, KeyError, TypeError) as exc:
        raise CandidatePricingError(str(exc)) from exc


def prepare_candidate_workbook(content: bytes, valuation_date: date) -> dict:
    try:
        return _prepare_latest_normal_workbook(content, valuation_date, use_candidate=True)
    except ServiceError:
        raise
    except ValueError as exc:
        raise LatestNormalValidationError(str(exc)) from exc
    except (RuntimeError, OverflowError, KeyError, TypeError) as exc:
        raise CandidatePricingError(str(exc)) from exc


def price_candidate_model(
    request: LatestNormalPriceRequest, market_store: MarketDataStore | None, curve_store: object
) -> dict:
    try:
        result = _price_latest_normal(
            request, market_store, curve_store, expected_model=CANDIDATE_MODEL
        )
        CandidatePriceResponse.model_validate(result)
        return result
    except ServiceError:
        raise
    except PydanticValidationError as exc:
        raise CandidatePricingError(str(exc)) from exc
    except ValueError as exc:
        raise LatestNormalValidationError(str(exc)) from exc
    except (RuntimeError, OverflowError, KeyError, TypeError) as exc:
        raise CandidatePricingError(str(exc)) from exc
