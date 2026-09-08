#!/usr/bin/env python3
"""
Teaching figure: why a floating body needs a MEMORY term, and how long the
memory actually is.

NOTE: the radiation damping B(omega) here is a plausible SHAPE, not a real
hull. Real B(omega) comes from a BEM solver (Capytaine / NEMOH / WAMIT).
The point of the figure is the mechanism and the timescale, not the numbers.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 1. radiation damping vs frequency (typical single-humped heave shape) ---
w = np.linspace(1e-4, 12.0, 4000)
w0, B0 = 1.4, 1.0                       # peak location, scale
B = B0 * (w / w0) ** 2 * np.exp(-(w / w0) ** 2)

# --- 2. retardation function via the Ogilvie relation -----------------------
#        K(t) = (2/pi) * integral_0^inf  B(omega) * cos(omega t) d omega
t = np.linspace(0, 12.0, 1200)
K = (2.0 / np.pi) * np.trapezoid(B[None, :] * np.cos(np.outer(t, w)), w, axis=1)

env = np.abs(K)
thr = 0.05 * env.max()
t_mem = t[np.where(env > thr)[0][-1]]   # when memory has decayed to 5%

# --- 3. what the memory does: response to a sudden velocity change ----------
# Compare the radiation force history for (a) constant damping b_eq, and
# (b) the true convolution, when velocity steps from 0 to 1 m/s at t=0.
b_eq = B[np.argmin(np.abs(w - w0))]                    # "equivalent" constant
F_const = np.full_like(t, b_eq)
F_mem = np.array([np.trapezoid(K[:i + 1][::-1], t[:i + 1]) if i else 0.0
                  for i in range(len(t))])

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

ax[0].plot(w, B, "tab:blue", lw=2)
ax[0].axvline(w0, ls=":", c="grey")
ax[0].set_xlim(0, 8)
ax[0].set_xlabel("$\\omega$  (rad/s)")
ax[0].set_ylabel("$B(\\omega)$  radiation damping")
ax[0].set_title("1. Damping DEPENDS ON FREQUENCY\n"
                "(this is the whole problem)", fontsize=10)
ax[0].grid(alpha=.3)

ax[1].plot(t, K, "tab:red", lw=2)
ax[1].axhline(0, c="k", lw=.6)
ax[1].axvline(t_mem, ls="--", c="tab:green", lw=1.6)
ax[1].annotate(f"memory ~{t_mem:.1f} s", (t_mem, K.max() * .55),
               xytext=(t_mem + 1.2, K.max() * .75), fontsize=9,
               color="tab:green",
               arrowprops=dict(arrowstyle="->", color="tab:green"))
ax[1].set_xlim(0, 10)
ax[1].set_xlabel("$t$  (s)")
ax[1].set_ylabel("$K(t)$  retardation function")
ax[1].set_title("2. In the TIME domain that becomes MEMORY\n"
                "$K(t)=\\frac{2}{\\pi}\\int B(\\omega)\\cos(\\omega t)\\,d\\omega$",
                fontsize=10)
ax[1].grid(alpha=.3)

ax[2].plot(t, F_const, "k--", lw=1.8, label="constant damping $b\\,\\dot z$")
ax[2].plot(t, F_mem, "tab:purple", lw=2, label="convolution $\\int K\\,\\dot z$")
ax[2].set_xlim(0, 8)
ax[2].set_xlabel("time since the velocity step  (s)")
ax[2].set_ylabel("radiation force")
ax[2].set_title("3. Response to a SUDDEN speed change\n"
                "the two disagree exactly on the preview timescale", fontsize=10)
ax[2].legend(fontsize=9); ax[2].grid(alpha=.3)

fig.suptitle("Why the Cummins equation has a convolution "
             "(illustrative $B(\\omega)$, not a real hull)", fontsize=12)
fig.tight_layout()
fig.savefig("fig6_cummins.png", dpi=140)
print(f"memory length (K decayed to 5%) = {t_mem:.2f} s")
print(f"peak of K at t = {t[np.argmax(K)]:.2f} s")
print("\nfigure: fig6_cummins.png")
