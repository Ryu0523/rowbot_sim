#!/usr/bin/env python3
"""
Resumable training state for model3.train / model_preview.train_p: every
`every` steps the network, optimiser, schedule, the data generator, the
global RNG states and the early-stopping record go to one file (temp file
+ os.replace); a restarted run with the same configuration continues from
the last saved step. A file saved under another configuration (steps,
batch, context, lr, seed, variant, episode count) is ignored with a
message, never loaded. The file is deleted when training ends normally.
"""
import os

import torch


class Resume:
    def __init__(self, path, config, every=1000, log=print):
        self.path, self.config, self.every, self.log = (path, dict(config),
                                                        int(every), log)

    def load(self, net, opt, sched, gen, best):
        """Restore into the given objects; returns the step to start at
        (0 when there is nothing to resume)."""
        if not self.path or not os.path.exists(self.path):
            return 0
        st = torch.load(self.path, map_location="cpu", weights_only=False)
        if st.get("config") != self.config:
            self.log(f"    resume file {self.path} is for {st.get('config')}"
                     f", not {self.config}: starting from step 0")
            return 0
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        gen.set_state(st["gen"])
        torch.set_rng_state(st["rng"])
        if torch.cuda.is_available() and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        best.clear()
        best.update(st["best"])
        self.log(f"    resumed from step {st['it']} ({self.path})")
        return int(st["it"])

    def save(self, it, net, opt, sched, gen, best):
        """Call after step `it` finished (it + 1 steps done)."""
        if not self.path or (it + 1) % self.every:
            return
        st = dict(config=self.config, it=it + 1, net=net.state_dict(),
                  opt=opt.state_dict(), sched=sched.state_dict(),
                  gen=gen.get_state(), rng=torch.get_rng_state(),
                  cuda_rng=(torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available() else None),
                  best=dict(best))
        tmp = self.path + ".tmp"
        torch.save(st, tmp)
        os.replace(tmp, self.path)

    def done(self):
        if self.path and os.path.exists(self.path):
            os.remove(self.path)
