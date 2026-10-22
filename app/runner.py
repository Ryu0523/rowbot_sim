"""
Run a policy on a scenario: one episode, one record.

    rec = run("demo_scarab195.json", {"type": "hand"}, seed=0)
    rec = run(sc, my_function)                    # any f(obs, info) -> action
    rec = run(sc, "learned:cat_p_prior", duration=20)

rate > 0 paces the loop to `rate` x real time (live demos); 0 runs as fast
as it can. on_step(env, info) is called after every control step (the live
server streams frames from it); stop() -> True ends the run early.

RESUME. run(..., resume=(record, k)) continues `record`'s episode after
its control step k (UsvEnv.resume: plant, hidden state, sensors, track,
autopilot as recorded) with `policy`; the scenario must be the record's
(header['scenario']). A policy whose action depends only on (t, state,
track) -- scripted -- reproduces the original record bit for bit; any
other starts its own memory fresh at k (a branch from a recorded moment).

A POLICY THAT FAILS MID-RUN (act raises, returns a non-finite or malformed
action, observe raises) does not lose the run: the episode ends with
reason "policy_error: <type>: <message>" (or "env_error: ..." when the
plant itself raised inside env.step), the traceback goes into the record's
header["error"], the policy's close() still runs (CUDA memory of learned
rows), and the partial record is returned -- raise_on_error=True re-raises
after that. A failure while SETTING UP (an unknown policy type,
pol.reset refusing a hull) raises at once: there is nothing to record.
"""
import time
import traceback

from .env import ActionError, UsvEnv
from .policies import make_policy, spec_of
from .scenario import load as load_scenario


def run(scenario, policy=None, *, seed=None, duration=None, model=None,
        rate=0.0, on_step=None, stop=None, env=None, verbose=False,
        raise_on_error=False, resume=None, test_switches=None,
        randomize=None, **usv):
    """randomize: None / False, True (draw this episode's domain
    randomisation, app/randomize.py, from the scenario's usv.randomization
    or the env's config) or a draw dict (applied as given)."""
    sc = load_scenario(scenario)
    if usv or seed is not None or duration is not None or model is not None:
        sc = sc.with_overrides(seed=seed, duration=duration, model=model,
                               **usv)
    spec = policy if policy is not None else sc.policy_spec()
    pol = make_policy(spec)
    if resume is not None and not hasattr(resume[0], "header"):
        from .record import Record
        resume = (Record.load(resume[0]), resume[1])
    if env is None:
        if resume is not None and test_switches is None:
            test_switches = resume[0].header.get("test_switches")
        env = UsvEnv(sc, action_mode=getattr(pol, "action_mode", "direct"),
                     test_switches=test_switches)
    if resume is not None:
        obs, info = env.resume(*resume)
    else:
        opts = None if not randomize else {"randomize": randomize}
        obs, info = env.reset(seed=sc.seed, options=opts)
    t0 = time.perf_counter()
    t_pol = 0.0
    failure = None
    try:
        pol.reset(env)
        while True:
            stage = "policy"
            try:
                tp = time.perf_counter()
                a = pol.act(obs, info)
                t_pol += time.perf_counter() - tp
                stage = "env"
                obs, r, term, trunc, info = env.step(a)
                stage = "policy"
                if hasattr(pol, "observe"):
                    pol.observe(info)
            except Exception as ex:                  # noqa: BLE001
                # a refused action (ActionError: malformed, non-finite) is
                # the policy's fault; anything else from env.step is the
                # plant's ("env_error")
                if stage == "env" and not isinstance(ex, ActionError):
                    stage = "env"
                else:
                    stage = "policy"
                failure = ex
                env.reason = (f"{stage}_error: {type(ex).__name__}: "
                              f"{ex}")[:500]
                env.error = dict(stage=stage, type=type(ex).__name__,
                                 message=str(ex),
                                 traceback=traceback.format_exc()[-4000:],
                                 k=env.k, t=env.t)
                break
            if on_step is not None:
                on_step(env, info)
            if verbose and env.k % 40 == 0:
                print(f"  t {env.t:6.1f} s  u {env.s[6]:5.2f} m/s  leg "
                      f"{info['leg']}/{info['n_legs']}  reward {r:+.3f}",
                      flush=True)
            if term or trunc:
                break
            if stop is not None and stop():
                env.reason = "stopped"
                break
            if rate > 0:
                ahead = env.t / rate - (time.perf_counter() - t0)
                if ahead > 0:
                    time.sleep(ahead)
        summ = {}
        if hasattr(pol, "summary"):
            try:
                summ = pol.summary()
            except Exception as ex:                  # noqa: BLE001
                summ = dict(summary_error=f"{type(ex).__name__}: {ex}")
    finally:
        if hasattr(pol, "close"):
            try:
                pol.close()
            except Exception:                        # noqa: BLE001
                traceback.print_exc()
    rec = env.record(policy=dict(spec=spec_of(spec),
                                 name=getattr(pol, "name", None),
                                 label=getattr(pol, "label", None),
                                 summary=summ,
                                 t_policy_per_step=t_pol / max(env.k, 1)))
    if failure is not None and raise_on_error:
        raise failure
    return rec
