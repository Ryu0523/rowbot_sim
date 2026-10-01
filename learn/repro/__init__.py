"""
Reproductions of five published sim-to-real / adaptive-control methods on
this project's own task (learn/repro/task.py): the Scarab 195 planing boat,
source = the low-fidelity world (sim/lofi.py), target = the full plant
(sim/planing_vessel.py), which stands in for the real boat.

    sgrl.py     SG-RL (Zarrouki et al., arXiv 2609.01061): PPO sets the MPC
                weights every control step, guided by the MPC's own
                sensitivity of a tracking loss to the weights
    berg.py     Berg et al. (Sci. Rep. 2025): PPO sets the MPC weights and
                the MPC model's mass and damping ("digital-twin syncing")
    latent.py   RMA (Kumar et al., RSS 2021) and Jiang et al. (arXiv
                2607.02037): a direct-control policy that sees the vessel's
                parameters in training (teacher / base policy) and a
                history module that estimates them at run time
    bilevel.py  Anand et al. (arXiv 2510.17709, RLJ 2026): PPO in the source,
                whose dynamics and reward parameters are moved so that the
                source-trained policy does better on the target

What each keeps from its paper and what had to change for a single-jet
planing boat is written at the top of each file. Driver and report:
studies/repro_baselines.py.
"""
