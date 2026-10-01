#!/usr/bin/env python3
"""
learn/meta/train_resume.py through model3.train and model_preview.train_p
on a small random dataset (no cache needed, CPU, about a minute): a run
interrupted after a saved step and then restarted ends with exactly the
same weights as an uninterrupted run; a resume file from another
configuration is ignored; the file is gone after a normal end.

    python -m studies.test_train_resume
"""
import os
import sys
import tempfile
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np   # noqa: E402
import torch   # noqa: E402

from learn.meta import model3 as M   # noqa: E402
from learn.meta import train_resume as TR   # noqa: E402

TR.Resume.__init__.__defaults__ = (5, lambda *a: None)   # save every 5


def fake_data(n=12, T=80, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = dict(dtype=torch.float32)
    D = SimpleNamespace(dev=torch.device("cpu"), n=n, T=T)
    D.X = torch.randn(n, T, M.D_X, generator=g, **f)
    D.U = torch.randn(n, T, 2, generator=g, **f)
    D.E = torch.randn(n, T, M.C7, generator=g, **f)
    D.valid = torch.ones(n, T, dtype=torch.bool)
    D.len = torch.full((n,), T, dtype=torch.long)
    D.stats = dict(e_sd=np.ones(M.C7, np.float32))
    return D


class Stop(Exception):
    pass


def run(kind, path, stop_at=None, steps=12, L=32):
    """stop_at: raise Stop right after the resume file is written at that
    step count (a crash between two saves repeats those steps anyway)."""
    D = fake_data()
    calls = dict(n=0)
    orig_save = TR.Resume.save

    def save(self, it, *a):
        orig_save(self, it, *a)
        if stop_at is not None and it + 1 == stop_at:
            raise Stop
    TR.Resume.save = save
    try:
        return _run(kind, path, D, steps, L)
    finally:
        TR.Resume.save = orig_save


def _run(kind, path, D, steps, L):
    stop_at = None
    calls = dict(n=0)
    if kind == "a":
        orig = M.fm_loss

        def loss(*a, **k):
            calls["n"] += 1
            if stop_at is not None and calls["n"] > stop_at:
                raise Stop
            return orig(*a, **k)
        M.fm_loss = loss
        try:
            return M.train(D, steps=steps, batch=4, L=L, log=lambda *a: None,
                           resume=path)
        finally:
            M.fm_loss = orig
    # w / p: the wave data cannot be faked here; a stand-in loss that
    # draws from the training generator exercises the same loop (resume,
    # schedule, generator, early-stopping record)
    from learn.meta import model_preview as MP
    orig, orig_v = MP.fm_loss_p, MP.fm_validate_p

    def loss_p(net, D, ii, a, L, gen, variant):
        z = torch.randn(8, generator=gen)
        return sum(((p_.float().mean() - z.mean()) ** 2)
                   for p_ in net.parameters())

    MP.fm_loss_p = loss_p
    MP.fm_validate_p = lambda *a, **k: 1.0
    try:
        return MP.train_p(D, kind, steps=steps, batch=4, L=L,
                          log=lambda *a: None, resume=path)
    finally:
        MP.fm_loss_p, MP.fm_validate_p = orig, orig_v


def same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def main():
    torch.use_deterministic_algorithms(True)
    ok_all = True
    for kind in ("a", "w", "p"):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "resume.pt")
        full = run(kind, None)
        try:
            run(kind, p, stop_at=5)          # stops right after saving step 5
        except Stop:
            pass
        had = os.path.exists(p)
        st = torch.load(p, weights_only=False)["it"] if had else None
        resumed = run(kind, p)
        gone = not os.path.exists(p)
        ok = had and st == 5 and same(full, resumed) and gone
        # a file from another configuration is ignored
        try:
            run(kind, p, stop_at=5)
        except Stop:
            pass
        other = run(kind, p, steps=14)
        ref = run(kind, None, steps=14)
        ok2 = same(other, ref)
        print(f"[{'PASS' if ok and ok2 else 'FAIL'}] model{kind}: saved at "
              f"step {st}, resumed == uninterrupted {same(full, resumed)}, "
              f"file removed at the end {gone}, other config ignored {ok2}")
        ok_all &= ok and ok2
    print("all passed" if ok_all else "FAILED")


if __name__ == "__main__":
    main()
