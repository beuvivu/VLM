# Walk-forward backtest — mega645
Draws 2019-09-27 → 2026-09-27 (1071 draws, 5 ticket(s)/draw unless the strategy fixes its own count)

| Strategy | Tickets | Mean matches (null) | z (cluster) | deff | q (>chance) | max e-value (Holm p) | edge < (95%) | MDE | ROI | ROI 95% CI | Popularity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 5355 | 0.7892 (0.8000) | -1.02 | 0.98 | 0.917 | 1.2 (1.00) | 3.5% | 3.3% | -92.4% | [-94.1%, -90.5%] | 1.01 |
| hot_50 | 5355 | 0.8032 (0.8000) | +0.18 | 2.57 | 0.768 | 73.6 (0.12) | 3.9% | 5.3% | -87.5% | [-91.9%, -83.6%] | 1.15 |
| cold_50 | 5355 | 0.7765 (0.8000) | -1.38 | 2.52 | 0.917 | 0.7 (1.00) | 6.4% | 5.3% | -91.6% | [-93.9%, -88.5%] | 1.16 |
| overdue | 5355 | 0.8043 (0.8000) | +0.25 | 2.48 | 0.768 | 1.0 (1.00) | 4.0% | 5.2% | -91.5% | [-93.9%, -88.3%] | 1.02 |
| bayes_decay0.99 | 5355 | 0.8037 (0.8000) | +0.36 | 0.96 | 0.768 | 10.7 (0.75) | 2.6% | 3.3% | -89.7% | [-92.5%, -86.6%] | 0.99 |
| markov | 5355 | 0.8166 (0.8000) | +1.46 | 1.12 | 0.643 | 3.1 (1.00) | 4.4% | 3.5% | -89.2% | [-92.2%, -85.7%] | 1.03 |
| gcn | 5355 | 0.7927 (0.8000) | -0.69 | 0.97 | 0.917 | 1.8 (1.00) | 3.1% | 3.3% | -89.1% | [-92.0%, -86.0%] | 0.99 |
| anti_popularity | 5355 | 0.8045 (0.8000) | +0.29 | 2.06 | 0.768 | 1.5 (1.00) | 3.7% | 4.8% | -90.3% | [-92.7%, -87.4%] | 0.42 |
| wheel_10_4if3 | 4284 | 0.7880 (0.8000) | -0.67 | 2.23 | 0.917 | 6.1 (1.00) | 5.2% | 5.6% | -92.3% | [-95.0%, -88.5%] | 1.08 |

Superior Predictive Ability (Hansen 2005, 2000 stationary-bootstrap draws): best = markov, p (consistent) = 0.418, p (lower/upper) = 0.269/0.418, White Reality Check p = 0.685.

No strategy matched more numbers than chance: every BH q ≥ 0.05, no anytime-valid e-process is significant after Holm across the 9 bettors, and the best strategy (data-snooping-adjusted SPA) has p = 0.42. Equivalence (TOST, 95%): every strategy's edge is below 6.4% of the chance level of 0.800 matches/ticket. ROI differences come from a handful of rare prize hits; the only structural lever is jackpot sharing (Popularity column). Temporary lucky streaks (e-value ≥ 20× at some point, not significant once all 9 bettors are accounted for): hot_50 (peak 74×, final 0.15×).
