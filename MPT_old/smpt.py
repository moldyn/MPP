#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from MPT import MPTBase

traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)

tlag = 50

mpt_det = MPTBase(traj, tlag)
mpt_det.mpt(2)

mpt_in_s1_det = np.zeros(mpt_det.n_states)
mpt_in_s1_det[mpt_det.macrostates[1]] = 1

n = 100
mpt_in_s1 = np.zeros((mpt_det.n_states, n))

for i in tqdm(range(n)):
    # mpt = MPTBase(traj, tlag, method="smpt", params={"%": 0.5})
    mpt = MPTBase(traj, tlag, method="smpt", params={"n": 2})
    mpt.mpt(2)
    mpt_in_s1[mpt.macrostates[1], i] = 1

mpt_in_s1_mean = mpt_in_s1.mean(axis=1)
order = np.argsort(mpt_in_s1_mean)
pops = np.cumsum(mpt.pop[order])

plt.figure(figsize=(8, 4))

plt.scatter(pops, mpt_in_s1_det[order], c="tab:orange", marker="+", label="Deterministic MPT")
plt.scatter(pops, mpt_in_s1_mean[order], marker="x", label=f"Stochastic MPT from {n} runs")

plt.title("Performance of Stochastic MPT algorithm (n 2)")
plt.xlabel("Cumulative Population")
plt.ylabel("Fraction in state 1")
plt.xlim(0, 1)
plt.legend()

plt.savefig("/home/fg149/Dokumente/data_production/MPT/MPT/performance_smpt_n2_test.pdf")
