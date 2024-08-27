import numpy as np
import msmhelper as mh

import MPT

traj = np.array([int(i) for i in "112225555566664422331122255252111445231231234444441121222222"])

tmat, states = mh.msm.estimate_markov_model(traj, lagtime=1)
_, pop = np.unique(traj, return_counts=True)

traj12 = mh.shift_data(traj, np.arange(1, 7), [0, 0, 1, 2, 3, 4])
tmat12, states12 = mh.msm.estimate_markov_model(traj12, lagtime=1)

tmat_mpt = np.zeros((7, 7))
tmat_mpt[:-1][:, :-1] = tmat
tmat_mpt, pop_mpt = MPT.utils.merge_states(tmat_mpt, [0, 1], -1, pop.copy())
o = [6, 2, 3, 4, 5]
tmat_mpto = tmat_mpt[o][:, o]

print(tmat12)
print(tmat_mpto)
