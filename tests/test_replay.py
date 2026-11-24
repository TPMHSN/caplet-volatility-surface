"""Contrôles rapides du parcours de rejeu sans données de marché privées."""

from datetime import date
from math import sqrt
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest
import QuantLib as ql

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "moteur")]

from fixings_io import load_fixings
from market_data.fixing_store import fixing_store
from reproduire import _market_date
from services.latest_normal_coupon import coupon_forward_contract
from services.thomas_stripping import _intrinsic, _time_value


AS_OF = date(2026, 11, 20)


@pytest.fixture
def histories(tmp_path):
    for tenor, rate in (("3M", 0.025), ("6M", 0.027)):
        payload = {
            "currency": "EUR",
            "index_tenor": tenor,
            "fixings": [
                {"date": "2026-11-19", "rate": rate - 0.001},
                {"date": "2026-11-20", "rate": rate},
                {"date": "2026-11-23", "rate": rate + 0.001},
            ],
        }
        (tmp_path / f"EUR_{tenor}.json").write_text(json.dumps(payload))
    yield tmp_path
    fixing_store.clear()


def test_replay_uses_observed_rates_without_lookahead(histories):
    audit = load_fixings(AS_OF, histories, offline=True)
    assert audit["3M"]["taux_decimal"] == 0.025
    assert audit["6M"]["taux_decimal"] == 0.027
    assert fixing_store.get_fixing("EUR", "3M", date(2026, 11, 19)) == 0.024
    assert fixing_store.get_fixing("EUR", "3M", date(2026, 11, 23)) is None


def test_missing_today_fixing_has_an_explicit_error(histories):
    with pytest.raises(ValueError, match="Fixing Euribor 3M observé du 2026-11-24 absent"):
        load_fixings(date(2026, 11, 24), histories, offline=True)


@pytest.mark.parametrize("name", ["20261120 - md.xlsx", "market-2026-11-20.xlsx"])
def test_market_date_from_workbook_name(name):
    assert _market_date(Path(name)) == AS_OF


def test_forward_reconstructed_from_projection_curve():
    valuation = ql.Date(20, 11, 2026)
    fixing = ql.Date(22, 2, 2027)
    curve = ql.YieldTermStructureHandle(ql.FlatForward(valuation, 0.03, ql.Actual365Fixed()))
    contract = coupon_forward_contract("3M", fixing)
    start, end = contract.index_value_date, contract.index_maturity_date
    forward = (curve.discount(start) / curve.discount(end) - 1) / contract.projection_accrual_fraction
    assert contract.forward_rate(curve) == pytest.approx(forward, abs=1e-14)


def test_native_caplet_uses_the_bachelier_price():
    coupon = SimpleNamespace(forward_rate=0.032, expiry_years=2,
                             accrual_fraction=0.25, discount_factor=0.96)
    strike, vol = 0.03, 0.008
    option_pv = _intrinsic(coupon, strike) + _time_value(coupon, strike, vol)
    expected = ql.bachelierBlackFormula(
        ql.Option.Call, strike, coupon.forward_rate, vol * sqrt(coupon.expiry_years),
        coupon.accrual_fraction * coupon.discount_factor,
    )
    assert option_pv == pytest.approx(expected, abs=1e-14)
