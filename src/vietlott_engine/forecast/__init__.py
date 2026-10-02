"""Self-learning next-draw forecaster for all eight Vietlott products (v3.5).

What it does, honestly
----------------------
A set of *experts* — uniform (the fair-machine null), "hot" and "cold" numbers over the
whole history and over recent draws, overdue / anti-overdue, repeat / anti-repeat of the last
draw, an online logistic model on those features, Dirichlet learners for digit games, and a
bank of sparse "one number / one digit is tilted" hypotheses — each issues a full probability
distribution for the next draw from past draws only. A fixed-share Bayesian mixture
re-weights the experts after every draw by how well each one actually predicted (this is the
self-learning part; it also re-learns if the machine's behaviour changes).

Because every forecast is a proper, predictable distribution, the mixture's likelihood ratio
against the fair-machine law is a test martingale: an *anytime-valid* e-value that says,
after any number of draws, whether the forecaster has found something real (≥ 1/α) or not.
If the draws are fair, the mixture learns to put its weight on the uniform expert and its
"most likely numbers" are no more likely than any others — the reports say so in numbers.
"""
