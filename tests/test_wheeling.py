from __future__ import annotations

from itertools import combinations

import pytest

from vietlott_engine.core.games import MEGA_645
from vietlott_engine.wheeling.cover import CoveringDesigner, WheelRequest, build_wheel, schonheim_bound


@pytest.mark.parametrize(("v", "k", "t", "optimum"), [(7, 3, 2, 7), (9, 3, 2, 12), (10, 4, 3, 30)])
def test_reaches_known_covering_numbers(v: int, k: int, t: int, optimum: int) -> None:
    """Fano plane C(7,3,2)=7, STS(9) C(9,3,2)=12, Steiner S(3,4,10) C(10,4,3)=30."""
    res = build_wheel(WheelRequest(pool=list(range(1, v + 1)), ticket_size=k, guarantee=t, condition=t), seed=1)
    assert res.verified
    assert res.n_tickets == optimum
    assert res.schonheim_lower_bound == schonheim_bound(v, k, t) <= optimum


def test_lotto_design_guarantee_holds_exhaustively() -> None:
    pool = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 42]
    res = build_wheel(WheelRequest(pool=pool, ticket_size=6, guarantee=3, condition=4), MEGA_645, seed=0, exact=False)
    assert res.verified
    for drawn in combinations(pool, 4):  # every way 4 drawn numbers can land in the pool
        assert max(len(set(drawn) & set(t)) for t in res.tickets) >= 3
    assert res.stats is not None and res.stats.cost == res.n_tickets * 10_000


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        CoveringDesigner(5, 6, 3, 3)
    with pytest.raises(ValueError):
        build_wheel(WheelRequest(pool=[1, 2, 3, 4, 5, 6, 99], ticket_size=6, guarantee=3, condition=3), MEGA_645)
