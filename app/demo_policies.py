"""Policies for the demonstration videos (python -m app run ... --policy
'{"type": "callable", "target": "app.demo_policies:SpeedHold", "kwargs": {...}}').

SpeedHold(v_kn, ...)   holds a set speed with the test scaffolding's P+I
                       throttle (studies/envelope_run) and follows the
                       scenario's track with the project's heading autopilot
                       (action mode 'heading', line-of-sight guidance).
                       Optionally the set speed is switched to v_lo_kn by one
                       of the preview-study rules (studies/preview_value_run:
                       'learned_future' = the impact predictor fed the exact
                       future wave, 'preview_group', 'preview_h', ...), so the
                       same policy shows "constant speed" and "slow down
                       before the waves the predictor flags" on the same sea.
"""
import math

import numpy as np

from .policies import Policy

KN = 0.514444
K_P, K_I, I_CLIP = 0.15, 0.02, 10.0


class SpeedHold(Policy):
    name = "speed_hold"
    action_mode = "heading"

    def __init__(self, v_kn=20.0, rule=None, v_lo_kn=15.0, thr=3.0,
                 horizon=10.0, hold=1.0, net="future+motion", guidance="los",
                 lookahead=None, thr_lo=0.0):
        self.v_hi, self.v_lo = float(v_kn) * KN, float(v_lo_kn) * KN
        self.rule, self.thr, self.horizon, self.hold = rule, float(thr), \
            float(horizon), float(hold)
        self.net, self.guidance, self.lookahead = net, guidance, lookahead
        self.thr_lo = float(thr_lo)
        self.label = f"speed hold {v_kn:g} kn" + (
            f" / pred -> {v_lo_kn:g} kn" if rule else "")

    def reset(self, env):
        self.env = env
        self.thr0 = float(env.s[12] / env.t_max)
        self.acc_err = 0.0
        self.ref = None
        self.log = []
        if self.rule:
            from studies.preview_value_run import (Encounter, LearnedTrigger,
                                                   SpeedRef)
            hs = float(env.sea_params["hs"])
            enc = Encounter(env.sea, env.x_station, self.horizon, env.dt_step)
            learned = None
            if self.rule.startswith("learned"):
                learned = LearnedTrigger(self.net, self.rule, env, enc, hs)
            self.ref = SpeedRef(self.rule, self.v_hi, self.v_lo, self.thr,
                                hs, enc, env.dt_step, self.horizon,
                                hold=self.hold, learned=learned)

    def act(self, obs, info):
        env = self.env
        s = env.s
        v_ref = self.v_hi
        if self.ref is not None:
            rows = getattr(env, "_demo_rows", None)
            v_ref = self.ref(s, env.t, 0.0)
        err = v_ref - float(s[6])
        self.acc_err = min(I_CLIP, max(-I_CLIP, self.acc_err + err))
        thr = min(1.0, max(self.thr_lo, self.thr0 + K_P * err
                           + K_I * self.acc_err))
        h = env.track.guidance(s[0], s[1], self.guidance, self.lookahead,
                               env.L)
        self.log.append((env.t, v_ref, float(s[6]), thr))
        return {"throttle": thr, "heading": h}

    def summary(self):
        lg = np.asarray(self.log, float) if self.log else np.zeros((0, 4))
        return dict(v_kn=self.v_hi / KN, rule=self.rule, v_lo_kn=self.v_lo / KN,
                    thr=self.thr, net=self.net if self.rule else None,
                    low_frac=float(np.mean(lg[:, 1] < self.v_hi - 1e-9))
                    if len(lg) else 0.0,
                    v_ref=lg[:, 1].tolist(), t=lg[:, 0].tolist())
