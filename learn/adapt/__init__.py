"""
Few-shot adaptation of the MPC to what the low-fidelity world lacks, by a
learned CORRECTION of its model rather than by retuning its weights.

The low-fidelity world has no speed loss in waves and almost no impacts, so
the MPC designed there pushes everywhere; the target wants to ease off in
head seas and push 45 deg off the bow (DEFECTS I). The knowledge that is
missing is carried here by two small corrections, learned online from the
boat's own measurements (speed log, compass, accelerometer) -- no new
sensor, and no knowledge of the wave direction: the heading is the context.

    features.py  feature maps: hand-chosen (speed polynomial x heading
                 harmonics) or random Fourier features, optionally with the
                 recent motion intensity as a stand-in for "how big the
                 waves are" (the confounder of speed and impacts)
    blr.py       Bayesian linear regression on those features: mean,
                 uncertainty, posterior samples, and how much a planned
                 observation would shrink the uncertainty of the slope the
                 decision depends on
    cmpc.py      the MPC that plans against the operator's cost with the
                 corrected model, and the ways it can probe: none, dither,
                 explore-then-commit, optimism, Thompson sampling, an
                 information bonus

Experiment driver: studies/adapt_matrix.py.
"""
