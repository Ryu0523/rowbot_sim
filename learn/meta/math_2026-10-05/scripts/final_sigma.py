"""How many standard deviations of the untruncated Gaussian smooth-in-speed
coefficients (operators_rb._smooth, rho = 0.8, the most favourable) a gz value
needs beyond the bounded offset range. Hand inputs from comparison.md A.5."""
import numpy as np
KN = 0.514444; umid, uhalf = 23 * KN, 7 * KN
def basis(kn):
    xi = np.clip((kn * KN - umid) / uhalf, -1, 1); return np.array([xi, 1.5 * xi * xi - 0.5])
def sd(sig, kn, rho=0.8):
    b = basis(kn); return np.sqrt((sig * rho * b[0]) ** 2 + (sig * rho ** 2 * b[1]) ** 2)
wh, zh, wp, zp = 5.83486, 0.372324, 3.71347, 0.155060; whp = wh * wp; D4 = 2 * zp * wp
rq = (0.18 * 5.4, 0.27 * 5.4)
gz = {16: dict(dwdth=-42.0, dqdz=17.7, dqdq=-6.2, dqdth=7.2, drdr=-1.734),
      20: dict(dwdth=-60.1, dqdz=20.7, dqdq=-8.1, dqdth=3.1, drdr=-1.465),
      25: dict(dwdth=-99.4, dqdz=13.1, dqdq=-11.3, dqdth=2.6, drdr=-1.399)}
for kn, g in gz.items():
    # T4 coupling: ks+ka = kzq/(whp rho_q) with kzq = -dwdth; bounded |ks+ka| <= 0.45
    need = -g["dwdth"] / (whp * rq[1]); s = np.hypot(sd(0.1, kn), sd(0.05, kn))
    # pitch damping: e^g = -dqdq / D4, bounded c0 <= ln 2.4, sigma 0.2
    gp = np.log(-g["dqdq"] / D4); sp_ = sd(0.2, kn)
    # yaw damping jointly with x_j/I_r: e^g = |drdr| tau_r / f_r, f_r in [1.466, 2.31]; bounded c0 >= ln 0.6
    gy = np.log(-g["drdr"] * 0.42 / 1.466)
    print(f"{kn} kn: T4 ks+ka need {need:.2f} (bound 0.45) -> {(need - 0.45) / s:.1f} sd; "
          f"pitch damping factor {np.exp(gp):.2f} (bound 2.4) -> {(gp - np.log(2.4)) / sp_:.1f} sd; "
          f"yaw S1 factor need <= {np.exp(gy):.3f} (bound 0.6) -> {(np.log(0.6) - gy) / sd(0.2, kn):.1f} sd")
