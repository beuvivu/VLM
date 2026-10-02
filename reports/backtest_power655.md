# Walk-forward backtest — power655
Draws 2019-07-06 → 2026-09-26 (1102 draws, 5 ticket(s)/draw unless the strategy fixes its own count)

| Strategy | Tickets | Mean matches (null) | z (cluster) | deff | q (>chance) | max e-value (Holm p) | edge < (95%) | MDE | ROI | ROI 95% CI | Popularity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 5510 | 0.6691 (0.6545) | +1.47 | 1.02 | 0.635 | 32.9 (0.27) | 4.7% | 3.8% | -90.7% | [-93.2%, -87.5%] | 1.00 |
| hot_50 | 5510 | 0.6497 (0.6545) | -0.30 | 2.72 | 0.853 | 0.9 (1.00) | 4.8% | 6.1% | -91.5% | [-94.7%, -87.5%] | 1.13 |
| cold_50 | 5510 | 0.6675 (0.6545) | +0.78 | 2.85 | 0.650 | 9.7 (0.83) | 6.1% | 6.3% | -90.9% | [-93.7%, -86.9%] | 1.29 |
| overdue | 5510 | 0.6691 (0.6545) | +0.88 | 2.87 | 0.650 | 1.1 (1.00) | 6.4% | 6.3% | -90.4% | [-94.2%, -84.4%] | 1.08 |
| bayes_decay0.99 | 5510 | 0.6477 (0.6545) | -0.68 | 1.05 | 0.853 | 1.0 (1.00) | 3.6% | 3.8% | -92.7% | [-94.6%, -90.2%] | 0.98 |
| markov | 5510 | 0.6368 (0.6545) | -1.66 | 1.18 | 0.952 | 1.1 (1.00) | 5.4% | 4.1% | -89.5% | [-93.0%, -84.9%] | 0.97 |
| gcn | 5510 | 0.6523 (0.6545) | -0.23 | 1.05 | 0.853 | 1.2 (1.00) | 2.9% | 3.8% | -91.2% | [-93.9%, -87.7%] | 1.02 |
| anti_popularity | 5510 | 0.6448 (0.6545) | -0.70 | 2.01 | 0.853 | 0.9 (1.00) | 5.0% | 5.3% | -91.3% | [-94.7%, -86.7%] | 0.44 |
| wheel_10_4if3 | 4408 | 0.6515 (0.6545) | -0.18 | 2.33 | 0.853 | 0.7 (1.00) | 4.7% | 6.4% | -94.6% | [-96.4%, -92.5%] | 1.02 |

Superior Predictive Ability (Hansen 2005, 2000 stationary-bootstrap draws): best = random, p (consistent) = 0.392, p (lower/upper) = 0.279/0.419, White Reality Check p = 0.662.

No strategy matched more numbers than chance: every BH q ≥ 0.05, no anytime-valid e-process is significant after Holm across the 9 bettors, and the best strategy (data-snooping-adjusted SPA) has p = 0.39. Equivalence (TOST, 95%): every strategy's edge is below 6.4% of the chance level of 0.655 matches/ticket. ROI differences come from a handful of rare prize hits; the only structural lever is jackpot sharing (Popularity column). Temporary lucky streaks (e-value ≥ 20× at some point, not significant once all 9 bettors are accounted for): random (peak 33×, final 1.19×).
