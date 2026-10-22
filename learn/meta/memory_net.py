#!/usr/bin/env python3
"""
Long-term memory for the one-step error model (ROADMAP_2026-09-30 section
11.3, candidates B, D and E): a pretrained model_preview.NetP keeps its
256-step causal window unchanged (the same local context as today) and
gains a memory of what happened BEFORE the window -- the earlier missions
of the same boat (and, with long enough missions, the earlier part of the
current one) -- written segment by segment.

Every kind starts as exactly the pretrained net (the read is a residual
whose output projection starts at zero), so fine-tuning starts from the
pretrained predictions and the memory only adds what it learns to add.

Kinds (`kind`):
  recurrent   memory tokens (Recurrent Memory Transformer style): n_mem
              vectors. A write runs the trunk over one past segment with
              the current memory in front and n_mem write queries at the
              end; the write queries' outputs, gated per channel, become
              the next memory (fixed size, constant cost)
  compress    context compression: every past segment is compressed into
              n_sum summary tokens (the write queries' outputs; the trunk
              reads the earlier summaries in the write too); the memory
              is a learned start token plus the summaries, the row's
              newest max_sum - 1 kept. Entries a row does not have (it sat
              out a batched write) are masked as keys and never count
              towards max_sum, so a row's memory never depends on the rest
              of its batch; the start token is never masked, so no
              attention row is ever empty
  ttt         test-time training of a fast-weight matrix W (n_fast x
              n_fast per row; TTT-Linear / delta-rule form): a write runs
              the plain trunk over the segment, projects its outputs to
              keys k (unit length) and values v and takes ONE gradient
              step of sum_t |v_t - W k_t|^2 over the segment's valid
              tokens, W <- W + eta / n_valid sum_t (v_t - W k_t) k_t^T,
              eta learned (sigmoid-bounded), W from a learned W0. The
              projections, eta and W0 are learned through the writes: the
              update rule is meta-learned, not set by hand
  none        no memory (the control: the same fine-tuning without it)

Read (recurrent, compress): the window runs through the base trunk
unchanged; each window output h_t then attends to the memory entries
(multi-head attention, keys / values from the memory, masked entries
excluded) and the result goes through a zero-initialised projection added
to h_t. ttt: h_t + Wo (W Wq h_t), Wo zero-initialised. The window stays
causal (the base trunk is causal; the read adds per-position terms only).

Write masks (recurrent, compress; boolean, True = not allowed): [memory |
segment tokens | write queries]: memory rows see memory; token t sees
memory and tokens <= t; write queries see everything; padded (invalid)
segment tokens and masked memory entries are excluded as keys.

A memory is a pair (entries (B, M, d) or W (B, f, f), present (B, M) bool
or None); MN.expand(mem, n) repeats a one-row memory for n windows. A
write only ever reads a segment that ends at or before the start of the
window that will read its result (the caller's responsibility, see
studies/train_mem.py and studies/onestep_gz.py).
"""
import torch
from torch import nn

KINDS = ("recurrent", "compress", "ttt", "none")


class MemNet(nn.Module):
    def __init__(self, base, kind="recurrent", n_mem=16, n_sum=4,
                 max_sum=64, n_fast=64, read_heads=4):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind {kind!r}: one of {KINDS}")
        self.base, self.kind, self.d = base, kind, base.d
        self.n_mem, self.n_sum, self.max_sum = int(n_mem), int(n_sum), \
            int(max_sum)
        self.n_fast, self.read_heads = int(n_fast), int(read_heads)
        self.head = base.head
        d = self.d
        if kind == "ttt":
            f = self.n_fast
            self.Wk = nn.Linear(d, f, bias=False)
            self.Wv = nn.Linear(d, f, bias=False)
            self.Wq = nn.Linear(d, f, bias=False)
            self.Wo = nn.Linear(f, d, bias=False)
            nn.init.zeros_(self.Wo.weight)
            self.W0 = nn.Parameter(torch.zeros(f, f))
            self.eta = nn.Parameter(torch.tensor(-1.0))
        elif kind in ("recurrent", "compress"):
            nw = self.n_mem if kind == "recurrent" else self.n_sum
            n0 = self.n_mem if kind == "recurrent" else 1
            self.mem0 = nn.Parameter(torch.randn(n0, d) * 0.02)
            self.wq = nn.Parameter(torch.randn(nw, d) * 0.02)
            self.tag_mem = nn.Parameter(torch.zeros(d))
            self.tag_wq = nn.Parameter(torch.zeros(d))
            if kind == "recurrent":
                self.gate = nn.Parameter(torch.full((d,), -2.0))
            self.read = nn.MultiheadAttention(d, self.read_heads,
                                              batch_first=True)
            self.read_ln = nn.LayerNorm(d)
            self.read_out = nn.Linear(d, d)
            nn.init.zeros_(self.read_out.weight)
            nn.init.zeros_(self.read_out.bias)

    # ---------------------------------------------------------- helpers
    def config(self):
        return dict(kind=self.kind, n_mem=self.n_mem, n_sum=self.n_sum,
                    max_sum=self.max_sum, n_fast=self.n_fast,
                    read_heads=self.read_heads)

    def empty(self, B, dev=None):
        """The memory before anything was written (module docstring)."""
        if self.kind == "none":
            return None, None
        if self.kind == "ttt":
            return self.W0[None].expand(B, -1, -1), None
        m = self.mem0[None].expand(B, -1, -1)
        return m, torch.ones(B, m.shape[1], dtype=torch.bool,
                             device=m.device)

    def _embed(self, tok):
        L = tok.shape[1]
        return self.base.inp(tok) + self.base.pos[:L][None]

    # ------------------------------------------------------------- read
    def encode(self, tok, mem=None):
        """Window tokens (B, L, D) -> trunk outputs (B, L, d) with the
        memory's residual read added (module docstring)."""
        h = self.base.encode(tok)
        if self.kind == "none":
            return h
        B = tok.shape[0]
        if mem is None:
            mem = self.empty(B, tok.device)
        m, present = mem
        if self.kind == "ttt":
            return h + self.Wo(torch.einsum("bij,blj->bli", m,
                                            self.Wq(h)))
        kv = self.read_ln(m)
        r, _ = self.read(h, kv, kv, key_padding_mask=~present,
                         need_weights=False)
        return h + self.read_out(r)

    def velocity(self, y, tau, h):
        return self.base.velocity(y, tau, h)

    def sample(self, h, n, steps=24, gen=None):
        return self.base.sample(h, n, steps=steps, gen=gen)

    def sample_grad(self, h, y0, steps=24):
        return self.base.sample_grad(h, y0, steps=steps)

    # ------------------------------------------------------------ write
    def write(self, tok, valid, mem, active=None):
        """One write: segment tokens (B, S, D) with validity (B, S); rows
        with active False (or no valid token) keep their memory (compress:
        get masked entries). Returns the new memory pair."""
        if self.kind == "none":
            return mem
        on = valid.any(1)
        if active is not None:
            on = on & active
        # only the rows that write are computed (the others are scattered
        # back unchanged), so rows that sat out cost nothing
        idx = on.nonzero(as_tuple=True)[0]
        if self.kind == "ttt":
            W = mem[0]
            if len(idx) == 0:
                return W, None
            new = self._write_ttt(tok[idx], valid[idx], W[idx])
            return W.index_copy(0, idx, new), None
        B, S, _ = tok.shape
        dev = tok.device
        m, present = mem
        M = m.shape[1]
        nw = self.wq.shape[0]
        w = m.new_zeros(B, nw, self.d)
        if len(idx):
            b = len(idx)
            wq = (self.wq + self.tag_wq)[None].expand(b, -1, -1)
            x = torch.cat([m[idx] + self.tag_mem, self._embed(tok[idx]), wq],
                          1)
            n = M + S + nw
            attn = torch.zeros(n, n, dtype=torch.bool, device=dev)
            attn[:M, M:] = True
            attn[M:M + S, M:M + S] = torch.triu(
                torch.ones(S, S, dtype=torch.bool, device=dev), 1)
            attn[M:M + S, M + S:] = True                  # tokens: no queries
            pad = torch.cat([~present[idx], ~valid[idx],
                             torch.zeros(b, nw, dtype=torch.bool,
                                         device=dev)], 1)
            h = self.base.out(self.base.tf(x, mask=attn,
                                           src_key_padding_mask=pad))
            w = w.index_copy(0, idx, h[:, M + S:])         # (B, nw, d)
        if self.kind == "recurrent":
            g = torch.sigmoid(self.gate)
            new = m + g * (w - m)
            return torch.where(on[:, None, None], new, m), present
        m = torch.cat([m, w], 1)
        present = torch.cat([present, on[:, None].expand(B, nw)], 1)
        if m.shape[1] > self.max_sum:
            # per row the start token and its newest max_sum - 1 PRESENT
            # entries (a row that sat out writes has absent entries at the
            # end, which must not push its real ones out); the memory is a
            # set (no positions), so packing the kept ones first is fine
            keep = self.max_sum - 1
            body, pb = m[:, 1:], present[:, 1:]
            newer = torch.flip(torch.cumsum(torch.flip(pb.long(), [1]), 1),
                               [1])                # present at >= this slot
            kept = pb & (newer <= keep)
            idx = torch.sort((~kept).long(), dim=1, stable=True).indices[
                :, :keep]
            m = torch.cat([m[:, :1], torch.gather(
                body, 1, idx[..., None].expand(-1, -1, body.shape[-1]))], 1)
            present = torch.cat([present[:, :1],
                                 torch.gather(kept, 1, idx)], 1)
        return m, present

    def _write_ttt(self, tok, valid, W):
        return self._ttt_step(W, *self._ttt_kv(tok), valid)

    def _ttt_kv(self, tok):
        h = self.base.encode(tok)           # causal; padded steps weighted 0
        k = self.Wk(h)
        return k / (k.norm(dim=-1, keepdim=True) + 1e-6), self.Wv(h)

    def _ttt_step(self, W, k, v, valid):
        w = valid.float()
        err = v - torch.einsum("bij,blj->bli", W, k)
        nv = w.sum(1).clamp(min=1.0)[:, None, None]
        return W + torch.sigmoid(self.eta) * torch.einsum(
            "bli,blj->bij", err * w[..., None], k) / nv

    def write_seq(self, tok, valid, step, row, mem):
        """Several writes from a flat list of segments: segment i (tokens
        tok[i], validity valid[i]) is written into row row[i] at write
        step step[i]; steps run in order, a row has at most one segment per
        step. The same as one write() per step with the rows that have a
        segment active; ttt runs the trunk over all segments in one call
        (its keys and values do not depend on W)."""
        if self.kind == "none" or len(step) == 0:
            return mem
        n = int(step.max()) + 1
        if self.kind == "ttt":
            k_all, v_all = self._ttt_kv(tok)
            W = mem[0]
            for j in range(n):
                i = (step == j).nonzero(as_tuple=True)[0]
                if len(i):
                    r = row[i]
                    W = W.index_copy(0, r, self._ttt_step(
                        W[r], k_all[i], v_all[i], valid[i]))
            return W, None
        B = mem[0].shape[0]
        for j in range(n):
            i = (step == j).nonzero(as_tuple=True)[0]
            t = tok.new_zeros(B, *tok.shape[1:])
            t[row[i]] = tok[i]
            v = torch.zeros(B, tok.shape[1], dtype=torch.bool,
                            device=tok.device)
            v[row[i]] = valid[i]
            act = torch.zeros(B, dtype=torch.bool, device=tok.device)
            act[row[i]] = True
            mem = self.write(t, v, mem, active=act)
        return mem


def expand(mem, n):
    """A one-row memory repeated for n rows."""
    m, p = mem
    if m is None:
        return mem
    return (m.expand(n, *m.shape[1:]),
            None if p is None else p.expand(n, *p.shape[1:]))


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def segments(n, S):
    """[(a, b)] covering [0, n) in segments of S steps (the last shorter)."""
    return [(a, min(n, a + S)) for a in range(0, n, S)]


def mem_lr_groups(net, lr_base, lr_new):
    """AdamW groups: the pretrained base at lr_base, the new memory
    parameters at lr_new (none for kind 'none')."""
    base = set(id(p) for p in net.base.parameters())
    new = [p for p in net.parameters() if id(p) not in base]
    groups = [dict(params=list(net.base.parameters()), lr=lr_base)]
    if new:
        groups.append(dict(params=new, lr=lr_new))
    return groups
