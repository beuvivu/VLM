"""One-call inferential summary ("fairness certificate") for a game's history."""

from __future__ import annotations

from pydantic import BaseModel

from vietlott_engine.core.history import DrawHistory
from vietlott_engine.inference.changepoint import ChangepointReport, changepoint_report
from vietlott_engine.inference.hierarchical import HierarchicalReport, hierarchical_report
from vietlott_engine.inference.multiple_testing import PerNumberReport, per_number_deviations
from vietlott_engine.inference.power import PowerEquivalenceReport, power_equivalence_report
from vietlott_engine.inference.sequential import SequentialReport, sequential_report


class InferenceReport(BaseModel):
    game: str
    draws: int
    first_date: str
    last_date: str
    per_number: PerNumberReport
    power: PowerEquivalenceReport
    hierarchical: HierarchicalReport
    sequential: SequentialReport
    changepoints: ChangepointReport
    findings: list[str]


def inference_report(h: DrawHistory, sims: int = 1000, alpha: float = 0.05, seed: int | None = 0) -> InferenceReport:
    per = per_number_deviations(h, sims=sims, seed=seed)
    pw = power_equivalence_report(h, alpha)
    hier = hierarchical_report(h)
    seq = sequential_report(h, alpha)
    cp = changepoint_report(h, sims=max(200, sims // 2), seed=seed)
    top = per.numbers[0]
    findings = [
        f"Per-number: most deviant is {top.number} (z = {top.z:+.2f}); Westfall–Young FWER-adjusted p = {top.p_westfall_young:.2f}"
        + (" — significant." if per.any_significant_fwer else " — not significant after accounting for all numbers."),
        pw.interpretation,
        hier.interpretation,
        seq.interpretation,
        cp.interpretation,
    ]
    return InferenceReport(
        game=h.spec.code.value,
        draws=len(h),
        first_date=str(h.dates[0]),
        last_date=str(h.dates[-1]),
        per_number=per,
        power=pw,
        hierarchical=hier,
        sequential=seq,
        changepoints=cp,
        findings=findings,
    )
