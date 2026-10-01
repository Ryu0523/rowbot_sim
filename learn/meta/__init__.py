"""
General (mechanism-agnostic) adaptation, step 1: learn to infer an UNKNOWN
residual of the MPC's model from the boat's own recent history.

Nothing here names a physical mechanism. The low-fidelity world is given
random extra accelerations in all five modelled degrees of freedom, drawn
from a broad family of random functions of the boat's state, its actuators
and time (residuals.py). A network learns, from a history of what the boat
did against what the model predicted, a POSTERIOR over the residual
function (model.py): a learned basis, a transformer history encoder and a
flow-matching posterior over the basis coefficients -- samples, so the
uncertainty can be multimodal.

    residuals.py  the random-function family and the input features
    data.py       episodes in the low-fidelity world with a random residual,
                  randomly excited (speed and heading targets), and the
                  residual OBSERVED the way it can be on the water: the
                  boat's velocity change minus a nominal twin's
    model.py      basis, encoder, flow matching; training and inference

Later steps: the MPC plans with posterior samples; RL learns when and how to
probe (DEFECTS J, the user's direction of 2026-09-29).
Driver: studies/meta_step1.py.
"""
