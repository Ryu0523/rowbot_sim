import numpy as np
from learn.meta.cat_water_a import _flutter_kcrit
wh, zh, wp, zp = 5.83486, 0.372324, 3.71347, 0.155060
kc = _flutter_kcrit(2*zh*wh, wh**2, 2*zp*wp, wp**2)
print("kappa*^2", kc)
for rf in (0.3, 2.5):
    kap = np.sqrt(rf*kc)
    for lf in (0.1*5.4, 1.0*5.4):
        print(f"rho_f {rf} l_f {lf:.2f}: k35 {kap*lf:.1f} k53 {kap/lf:.2f} product {-kap**2:.0f}")
# gz needs (25 kn): k35 = 99.4 (dwdot/dtheta=-99.4), k53 = -13.1 ; product
for sp,(a,b) in {"16":(42.0,17.7),"20":(60.1,20.7),"25":(99.4,13.1),"30":(125.9,6.3)}.items():
    print(sp,"kn need kappa^2", a*b, "rho_f", a*b/kc, "l_f", np.sqrt(a/b))
# W1 moment dq/dz = W l_x/(hbar_run I_q)
W=1400*9.81; hbar=np.mean([0.251,0.196,0.142,0.087,0.033])
print("hbar_run", hbar)
for lx in (0.15*5.4,):
    for Iq in (3037, 9439):
        print("W1 dqdot/dz max", W*lx/(hbar*Iq), "Iq", Iq)
# yaw joint: e^gS1 f_r = 0.5876 (25kn), x_j/I_r = -7.438e-4
Ir0=5253.5
for xj in (-2.665,-1.585):
    Ir=-xj/7.438e-4; fr=Ir0/Ir
    print("x_j",xj,"I_r",Ir,"f_r",fr,"e^gS1 needed",1.399/2.381/fr)
