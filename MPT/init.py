#!/usr/bin/env python3

import time
from plot_dendrogram import plot_tree
import numpy as np
#from MPT import MPTBase
#from MPT import BinaryTreeNode
import MPT
import msmhelper as mh
from anytree import RenderTree

def print_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Function '{func.__name__}' executed in {runtime:.4f} seconds")
        return result
    return wrapper

traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)

feature_traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")

out = "/home/fg149/Dokumente/data_production/MPT/MPT/dendrogram_test"

root, Z = MPT.mpt(traj, 50)
froot = MPT.add_feature(traj, feature_traj, root)
mroot = MPT.define_macrostates(root, 4)

plot_tree(root, out)



# tmat, states = mh.msm.estimate_markov_model(traj, 50)
# _, pop = np.unique(traj, return_counts=True)
#
# Z, full_pop = MPT._cluster(tmat, pop)
# nodes = MPT.build_tree(Z, full_pop)


# r = BinaryTreeNode(1)
# n1 = BinaryTreeNode(2, population=5, q=0.2)
# n2 = BinaryTreeNode(2, population=10, q=0.3)
#
# r.add_node(n1)
# r.add_node(n2)



# mpt = MPTBase(traj, 50, method="smpt", params={"n": 2})
#
# mpt.add_feature("fnc", feature_traj)
#
# mpt.mpt(4)
#dd = plot_dendrogram_mpt(mpt, "/home/fg149/Dokumente/data_production/MPT/MPT/h35_fg_dendrogram_smpt_n_6")

