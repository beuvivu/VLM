from __future__ import annotations

from math import comb

import numpy as np
import pytest
from pydantic import ValidationError

from vietlott_engine.core.games import MEGA_645, POWER_655, get_game
from vietlott_engine.core.history import DrawHistory
from vietlott_engine.core.models import Draw, Ticket


def test_tier_probabilities_known_odds() -> None:
    assert MEGA_645.total_combinations == 8_145_060
    assert POWER_655.total_combinations == 28_989_675
    p = POWER_655.tier_probabilities
    assert p["jackpot1"] == pytest.approx(1 / 28_989_675)
    assert p["jackpot2"] == pytest.approx(6 / comb(55, 6))  # ≈ 1 / 4,831,613
    assert p["first"] == pytest.approx(6 * 48 / comb(55, 6))
    assert MEGA_645.tier_probabilities["third"] == pytest.approx(comb(6, 3) * comb(39, 3) / comb(45, 6))


def test_match_distribution_sums_to_one() -> None:
    for g in (MEGA_645, POWER_655):
        assert sum(g.match_distribution()) == pytest.approx(1.0)


def test_classify_prefers_highest_tier() -> None:
    assert POWER_655.classify(5, bonus_hit=True).name == "jackpot2"
    assert POWER_655.classify(5, bonus_hit=False).name == "first"
    assert MEGA_645.classify(2) is None
    assert get_game("645") is MEGA_645


def test_draw_validation() -> None:
    d = Draw(game="power655", draw_id=1, draw_date="2017-08-01", numbers=[38, 5, 10, 14, 23, 24], bonus=35)
    assert d.numbers == (5, 10, 14, 23, 24, 38)
    with pytest.raises(ValidationError):
        Draw(game="power655", draw_id=1, draw_date="2017-08-01", numbers=[1, 2, 3, 4, 5, 6])  # missing bonus
    with pytest.raises(ValidationError):
        Draw(game="power655", draw_id=1, draw_date="2017-08-01", numbers=[1, 2, 3, 4, 5, 6], bonus=6)
    with pytest.raises(ValidationError):
        Draw(game="mega645", draw_id=1, draw_date="2017-08-01", numbers=[1, 2, 3, 4, 5, 46])
    with pytest.raises(ValidationError):
        Ticket(game="mega645", numbers=[1, 1, 2, 3, 4, 5])


def test_history_prefix_is_readonly_and_detached(mega_history: DrawHistory) -> None:
    view = mega_history.upto(100)
    assert len(view) == 100 and not view.incidence.flags.writeable
    with pytest.raises(ValueError):
        view.incidence[0, 0] = True
    det = view.detached()
    assert det.incidence.base is None or det.incidence.base.shape[0] == 100
    assert np.array_equal(det.numbers, mega_history.numbers[:100])
    assert mega_history.incidence.sum(axis=1).tolist() == [6] * len(mega_history)


def test_history_detects_duplicates() -> None:
    with pytest.raises(ValueError):
        DrawHistory.from_arrays(MEGA_645, np.array([[1, 1, 2, 3, 4, 5]]))
