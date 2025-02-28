#!/usr/bin/env python

import os
import yaml
from pathlib import Path
import argparse

import numpy as np
import MPT
import msmhelper as mh
import matplotlib.pyplot as plt
import prettypyplot as pplt
from scipy.stats import pearsonr
from itertools import combinations
from tqdm import tqdm



source_root = "/data/evaluation/MPP/stochastic_MPP_Felix/data_source/"


### HP35 #####################################################################

limits = None
multi_feature_raw = np.loadtxt(source_root + "hp35.mindists2")
microtraj = np.loadtxt(source_root + "hp35.selected_contacts.gaussian10f_microstates_pcs5_p153")
cluster_file = source_root + "hp35.mindist2.mosaic_clusters"

multi_feature_bool = multi_feature_raw < 0.45
feature_traj = multi_feature_bool.mean(axis=1)

topology_file = source_root + "villin_crystal_number360K.pdb"
xtc_file = source_root + "pnas2012-2f4k-360K-protein-fit.xtc"

tlag = 50
pop_min = 0.005
q_min = 0.5

root = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/HP35/"


### PDZ3 #####################################################################

# limits = np.loadtxt(source_root + "PDZ3/limits", dtype=int)
# multi_feature_raw = np.loadtxt(source_root + "PDZ3/dist_all")
# microtraj = np.loadtxt(source_root + "PDZ3/microstates_p100")
# cluster_file = source_root + "PDZ3/clusters"
#
# multi_feature_bool = multi_feature_raw < 0.45
# feature_traj = multi_feature_bool.mean(axis=1)
#
# tlag = 50
# pop_min = 0.05
# q_min = 0.8
#
# root = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/PDZ3/"


class Data:
    def __init__(self, yaml_file):
        with open(yaml_file, "r") as f:
            self.d = yaml.safe_load(f)

        self.source = self.d["source"]
        self.out = self.d["out"]

        self.microtraj = np.loadtxt(os.path.join(
            self.source,
            self.d["microstate trajectory"]
        ), dtype=np.uint32)
        self.mtraj_raw = np.loadtxt(os.path.join(
            self.source,
            self.d["multi feature trajectory"],
        ))
        self.limits = None if self.d["limits"] is None else np.loadtxt(self.d["limits"], dtype=np.uint64)
        self.mfeature_traj = self.mtraj_raw < 0.45
        self.feature_traj = self.mfeature_traj.mean(axis=1)
        self.cluster = os.path.join(self.source, self.d["cluster file"])

        self.top = os.path.join(self.source, self.d["topology file"])
        self.xtc = os.path.join(self.source, self.d["xtc file"])

        self.tlag = self.d["tlag"]
        self.pop_min = self.d["pop_min"]
        self.q_min = self.d["q_min"]

        self.lumping_dir = None
        self.kernel = None
        self.feature_kernel = None

        self.n_random_frames = 20
        self.use_ref = True



### RUN ######################################################################

def process_lumpings(lumpings, data, func, mpts=None):
    """Perform lumpings"""
    if mpts is None:
        mpts = [None] * len(lumpings)
    for i, lumping in tqdm(enumerate(lumpings)):
        kernel = MPT.kernel.MPTKernel(similarity=lumpings[lumping]["kernel similarity"])
        data.use_ref = True
        if lumpings[lumping]["feature kernel"] is None:
            feature_kernel = 1
            if lumpings[lumping]["kernel similarity"] == "P":
                data.use_ref = False
        elif lumpings[lumping]["feature kernel"] == "fnc":
            feature_kernel = MPT.kernel.FeatureKernel(
                feature_traj,
                microtraj,
            )
        elif lumpings[lumping]["feature kernel"] == "JS":
            feature_kernel = MPT.kernel.MultiFeatureKernel(
                multi_feature_bool,
                microtraj,
            )
        else:
            raise ValueError("feature kernel must be None, fnc or JS.")

        data.lumping_dir = os.path.join(data.out, lumping)
        Path(data.lumping_dir).mkdir(parents=True, exist_ok=True)
        data.kernel = kernel
        data.feature_kernel = feature_kernel


        if mpts[i] is None:
            mpts[i] = MPT.MPT(
                data.microtraj,
                data.tlag,
                data.feature_traj,
                macrostate_thresholds=(data.pop_min, data.q_min),
                limits=data.limits,
                quiet=True,
            )
        mpts[i] = func(mpts[i], data)
    return mpts


def mpp(mpt, data):
    """Performs MPP and saves Z matrix"""
    mpt.mpt(
        data.kernel,
        feature_kernel=data.feature_kernel,
    )
    mpt.save_Z(os.path.join(data.lumping_dir, "Z.npy"))
    return mpt


def standard_plots(mpt, data):
    if mpt.Z is None:
        mpt.from_Z(os.path.join(data.lumping_dir, "Z.npy"))
    out = data.lumping_dir
    Path(out).mkdir(parents=True, exist_ok=True)
    print("Plotting dendrogram...")
    mpt.plot(os.path.join(out + "dendrogram.pdf"), scale=1)
    print("Plotting implied timescales...")
    mpt.plot_implied_timescales(os.path.join(out + "timescales.pdf"), use_ref=data.use_ref, scale=1)
    print("Plotting Sankey diagram...")
    mpt.plot_sankey(os.path.join(out + "sankey.pdf"), scale=1)
    print("Plotting contact representation...")
    mpt.plot_contact_rep(data.mtraj_raw, data.cluster, os.path.join(out + "contact_rep.pdf"), scale=1.4)
    print("Plotting trajectory...")
    mpt.plot_macrotraj(os.path.join(out + "macrotraj.pdf"), row_length=0.1)
    print("Performing Chapman Kolmogorov test...")
    mpt.plot_ck_test(os.path.join(out + "ck_test.pdf"), frame_length=0.2)
    return mpt

def draw_random_frames(mpt, data):
    if mpt.Z is None:
        mpt.from_Z(os.path.join(data.lumping_dir, "Z.npy"))
    Path(os.path.join(data.lumping_dir + "random_frames/")).mkdir(parents=True, exist_ok=True)
    mpt.topology_file = data.top
    mpt.xtc_trajectory_file = data.xtc
    mpt.draw_random_frames(os.path.join(data.lumping_dir + "random_frames/"), n=data.n_random_frames)
    return mpt


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Perform MPP on MD simulation data",
        description=(
            "This program allows for the analysis of MD data utilizing the "
            "most probable path algorithm. It allows for easy plotting of "
            "different quality measures."
        ),
    )
    parser.add_argument(
        "data_specification",
        help=(
            "yaml file containing specification of files and parameters of "
            "the simulation"
        ),
        type=argparse.FileType('r', encoding='latin-1'),
    )
    parser.add_argument(
        "lumping_grid",
        help=(
            "yaml file defining the lumpings to perform and where to store "
            "them."
        ),
        type=argparse.FileType('r', encoding='latin-1'),
    )
    parser.add_argument(
        "-Z",
        help="Perform MPP and write the Z matrix.",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--standard-plots",
        help="Plot standard plots for specified lumpings",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--draw-random",
        help="Draw N random frames for each macrostate",
        metavar="N",
        type=int,
    )
    return parser.parse_args()
    

def main():
    args = parse_args()

    # Parse input files
    data = Data(args.data_specification.name)
    with open(args.lumping_grid.name, "r") as f:
        lumpings = yaml.safe_load(f)

    mpts = [None] * len(lumpings)
    if args.Z:
        mpts = process_lumpings(lumpings, data, mpp, mpts)
    if args.standard_plots:
        mpts = process_lumpings(lumpings, data, standard_plots, mpts)
    if args.draw_random:
        data.n_random_frames = args.draw_random
        mpts = process_lumpings(lumpings, data, standard_plots, mpts)

if __name__ == "__main__":
    main()


# lumpings = {
#     "ref/": {
#         "kernel similarity": "P",
#         "feature kernel": None,
#     },
#
#     "fnc/": {
#         "kernel similarity": "F",
#         "feature kernel": "fnc",
#     },
#
#     "kl/": {
#         "kernel similarity": "KL",
#         "feature kernel": None,
#     },
#
#     "js/": {
#         "kernel similarity": "F",
#         "feature kernel": "JS",
#     },
#
#     "ref_fnc/": {
#         "kernel similarity": "P",
#         "feature kernel": "fnc",
#     },
#
#     "ref_js/": {
#         "kernel similarity": "P",
#         "feature kernel": "JS",
#     },
#
#     "kl_fnc/": {
#         "kernel similarity": "KL",
#         "feature kernel": "fnc",
#     },
#
#     "kl_js/": {
#         "kernel similarity": "KL",
#         "feature kernel": "JS",
#     },
# }



# for lumping in lumpings:
#     kernel = MPT.kernel.MPTKernel(similarity=lumpings[lumping]["kernel similarity"])
#     use_ref = True
#     if lumpings[lumping]["feature kernel"] is None:
#         feature_kernel = 1
#         if lumpings[lumping]["kernel similarity"] == "P":
#             use_ref = False
#     elif lumpings[lumping]["feature kernel"] == "fnc":
#         feature_kernel = MPT.kernel.FeatureKernel(
#             feature_traj,
#             microtraj,
#         )
#     elif lumpings[lumping]["feature kernel"] == "JS":
#         feature_kernel = MPT.kernel.MultiFeatureKernel(
#             multi_feature_bool,
#             microtraj,
#         )
#     else:
#         raise ValueError("feature kernel must be 1, single or multi.")
#     mpt = MPT.MPT(
#         microtraj,
#         tlag,
#         feature_traj,
#         macrostate_thresholds=(pop_min, q_min),
#         limits=limits,
#     )
#     mpt.mpt(
#         kernel,
#         feature_kernel=feature_kernel,
#     )
#     mpt.topology_file = topology_file
#     mpt.xtc_trajectory_file = xtc_file
#     plots(mpt, root + lumping, use_ref=lumpings[lumping]["use_ref"])



# kernel = MPT.kernel.MPTKernel()
# mpt = MPT.MPT(
#     microtraj,
#     tlag,
#     feature_traj,
#     macrostate_thresholds=(pop_min, q_min),
#     limits=limits,
# )
# mpt.mpt(kernel)
# mpt.topology_file = topology_file
# mpt.xtc_trajectory_file = xtc_file
# plots(mpt, root + "ref/", use_ref=False)



# kernel = MPT.kernel.MPTKernel()
# feature_kernel = MPT.kernel.FeatureKernel(
#     feature_traj,
#     microtraj,
# )
# mpt_rfnc = MPT.MPT(
#     microtraj,
#     tlag,
#     feature_traj,
#     macrostate_thresholds=(pop_min, q_min),
#     limits=limits,
# )
# mpt_rfnc.mpt(
#     kernel,
#     feature_kernel=feature_kernel,
# )
# plots(mpt, root + "ref_fnc/")
#
#
# kernel = MPT.kernel.MPTKernel(
#     similarity="KL",
# )
# multi_feature_kernel = MPT.kernel.MultiFeatureKernel(
#     multi_feature_bool,
#     microtraj,
#     similarity="JS",
# )
# mpt_kljs = MPT.MPT(
#     microtraj,
#     tlag,
#     feature_traj,
#     macrostate_thresholds=(pop_min, q_min),
#     limits=limits,
# )
# mpt_kljs.mpt(
#     kernel,
#     feature_kernel=multi_feature_kernel,
# )
# plots(mpt, root + "kl_js/")
#
#
# kernel = MPT.kernel.MPTKernel(
#     similarity="KL",
#     term="*",
# )
# mpt_kl = MPT.MPT(
#     microtraj,
#     tlag,
#     feature_traj,
#     macrostate_thresholds=(pop_min, q_min),
#     limits=limits,
# )
# mpt_kl.mpt(kernel)
# plots(mpt, root + "kl/")
#
#
# kernel = MPT.kernel.MPTKernel(
#     similarity="F",
# )
# feature_kernel = MPT.kernel.FeatureKernel(
#     feature_traj,
#     microtraj,
# )
# mpt_fnc = MPT.MPT(
#     microtraj,
#     tlag,
#     feature_traj,
#     macrostate_thresholds=(pop_min, q_min),
#     limits=limits,
# )
# mpt_fnc.mpt(
#     kernel,
#     feature_kernel=feature_kernel,
# )
# plots(mpt_fnc, root + "fnc/")
#
#
# kernel = MPT.kernel.MPTKernel(
#     similarity="F",
# )
# multi_feature_kernel = MPT.kernel.MultiFeatureKernel(
#     multi_feature_bool,
#     microtraj,
#     similarity="JS",
# )
# mpt_js = MPT.MPT(
#     microtraj,
#     tlag,
#     feature_traj,
#     macrostate_thresholds=(pop_min, q_min),
#     limits=limits,
# )
# mpt_js.mpt(
#     kernel,
#     feature_kernel=multi_feature_kernel,
# )
# plots(mpt_fnc, root + "js/")


# kernel = MPT.kernel.MPTKernel(
#     param=2,
#     similarity="KL",
#     term="*",
# )
# multi_feature_kernel = MPT.kernel.MultiFeatureKernel(
#     multi_feature_bool,
#     microtraj,
#     similarity="JS",
# )
# mpt_kljs_n2 = MPT.MPT(
#     microtraj,
#     tlag,
#     feature_traj,
#     macrostate_thresholds=(pop_min, q_min),
#     limits=limits,
# )
# mpt_kljs_n2.mpt(
#     kernel,
#     feature_kernel=multi_feature_kernel,
#     n=100,
# )
#
# its_max = np.argmax(mpt_kljs_n2.timescales[:, 0])
# mpt_kljs_n2.n_i = its_max
# plots(mpt_fnc, root + "kl_js_n2/")







### Plot Pearson correlation coefficients in the course of a lumping ofr all lumpings

# for mpp, dir, lab in tqdm(zip(
#     [mpt, mpt_fnc, mpt_rfnc, mpt_kl, mpt_js, mpt_kljs],
#     ["ref/", "fnc/", "ref_fnc/", "kl/", "js/", "kl_js/"],
#     ["Ref", "fnc", "Ref + fnc", r"$D_\mathrm{KL}$", r"$D_\mathrm{JS}$", r"$D_\mathrm{KL} + D_\mathrm{JS}$"]
# )):
#     Z = mpp.Z
#     tmat = mpp.tmat
#     pop = mpp.pop
#
#     full_tmat, full_pop = MPT.utils.calc_full_tmat(tmat, pop, Z)
#     dq_full_tmat = MPT.utils.dq(full_tmat[0], Z[0])
#     # dq_full_pop = MPT.utils.dq(full_pop[0], Z[0], similarity="pop")
#     dq_full_klp = MPT.utils.dq(full_tmat[0], Z[0], similarity="KLP")
#     dq_full_feature = MPT.utils.dq(
#         feature_kernel.full_feature_from_Z(Z)[0],
#         Z[0],
#         similarity="fnc",
#     )
#     dq_full_multi_feature = MPT.utils.dq(
#         multi_feature_kernel.full_feature_from_Z(Z)[0],
#         Z[0],
#         similarity="JSC",
#     )
#     name = ["P", "fnc", "KLP", "JSC"]
#     labels = ["P", r"$\Delta$fnc", r"$D_\mathrm{KL}$", r"$D_\mathrm{JS}$"]
#     features = [
#         dq_full_tmat,
#         dq_full_feature,
#         dq_full_klp,
#         dq_full_multi_feature,
#     ]
#     iter = combinations(zip(name, labels, features), 2)
#     for (n1, l1, f1), (n2, l2, f2) in iter:
#         print(f"{lab}: {l1} - {l2}")
#         out = root + dir + f"pearson_{n1}_{n2}.pdf"
#         MPT.plot.plot_pearson(
#             f1,
#             f2,
#             out,
#             title=f"{lab}: {l1} - {l2} Correlation",
#             clip_to_greater_zero=dq_full_tmat,
#         )



### Correlation Scatter Plot

# for mpp, dir, lab in tqdm(zip(
#     [mpt, mpt_fnc, mpt_rfnc, mpt_kl, mpt_js, mpt_kljs],
#     ["ref/", "fnc/", "ref_fnc/", "kl/", "js/", "kl_js/"],
#     ["Ref", "fnc", "Ref + fnc", r"$D_\mathrm{KL}$", r"$D_\mathrm{JS}$", r"$D_\mathrm{KL} + D_\mathrm{JS}$"]
# )):
#     Z = mpp.Z
#     tmat = mpp.tmat
#     pop = mpp.pop
#     macro_tmat = mpp.macro_tmat[0]
#     macro_pop = mpp.macro_pop[0]
#     macro_traj = mpp.macrotraj[:, 0]
#
#     dq_full_tmat = MPT.utils.dq(tmat)
#     dq_full_pop = MPT.utils.dq(pop, similarity="origin pop")
#     dq_full_klp = MPT.utils.dq(tmat, similarity="KLP")
#     dq_full_feature = MPT.utils.dq(
#         feature_kernel.full_feature[:feature_kernel.n_states],
#         similarity="fnc",
#     )
#     dq_full_multi_feature = MPT.utils.dq(
#         multi_feature_kernel.full_feature[:multi_feature_kernel.n_states],
#         similarity="JSC",
#     )
#
#     dq_macro_tmat = MPT.utils.dq(macro_tmat)
#     dq_macro_pop = MPT.utils.dq(macro_pop, similarity="origin pop")
#     dq_macro_klp = MPT.utils.dq(macro_tmat, similarity="KLP")
#     mfk = MPT.kernel.FeatureKernel(feature_traj, macro_traj)
#     dq_macro_feature = MPT.utils.dq(
#         mfk.full_feature[:mfk.n_states],
#         similarity="fnc",
#     )
#     mmfk = MPT.kernel.MultiFeatureKernel(multi_feature_bool, mpp.macrotraj[:, 0])
#     dq_macro_multi_feature = MPT.utils.dq(
#         mmfk.full_feature[:mmfk.n_states],
#         similarity="JSC",
#     )
#
#
#     name = ["P", "fnc", "KLP", "JSC"]
#     labels = ["P", r"$\Delta$fnc", r"$D_\mathrm{KL}$", r"$D_\mathrm{JS}$"]
#     features = [
#         dq_full_tmat,
#         dq_full_feature,
#         dq_full_klp,
#         dq_full_multi_feature,
#     ]
#     macro_features = [
#         dq_macro_tmat,
#         dq_macro_feature,
#         dq_macro_klp,
#         dq_macro_multi_feature,
#     ]
#     iter = combinations(zip(name, labels, features, macro_features), 2)
#     for (n1, l1, f1, mf1), (n2, l2, f2, mf2) in iter:
#         print(f"{lab}: {l1} - {l2}")
#         out = root + dir + f"correlation_scatter_{n1}_{n2}.pdf"
#         MPT.plot.plot_correlation_scatter(
#             f1,
#             f2,
#             out,
#             macro_feature1=mf1,
#             macro_feature2=mf2,
#             weights=dq_full_pop,
#             macro_weights=dq_macro_pop,
#             label1=l1,
#             label2=l2,
#             clip_to_greater_zero=dq_full_tmat,
#             clip_to_greater_zero_macro=dq_macro_tmat,
#             title=f"{lab}: {l1} - {l2} Correlation",
#         )




### Microstates correlations

# mpp = mpt_fnc
#
# Z = mpp.Z
# tmat = mpp.tmat
# pop = mpp.pop
#
# dq_full_tmat = MPT.utils.dq(tmat)
# dq_full_pop = MPT.utils.dq(pop, similarity="origin pop")
# dq_full_klp = MPT.utils.dq(tmat, similarity="KLP")
# dq_full_feature = MPT.utils.dq(
#     feature_kernel.full_feature[:feature_kernel.n_states],
#     similarity="fnc",
# )
# dq_full_multi_feature = MPT.utils.dq(
#     multi_feature_kernel.full_feature[:multi_feature_kernel.n_states],
#     similarity="JSC",
# )
#
# name = ["P", "fnc", "KLP", "JSC"]
# labels = ["P", r"$\Delta$fnc", r"$D_\mathrm{KL}$", r"$D_\mathrm{JS}$"]
# features = [
#     dq_full_tmat,
#     dq_full_feature,
#     dq_full_klp,
#     dq_full_multi_feature,
# ]
#
# dir = "microstates/"
#
# iter = combinations(zip(name, labels, features), 2)
# for (n1, l1, f1), (n2, l2, f2) in iter:
#     # print(f"{lab}: {l1} - {l2}")
#     out = root + dir + f"correlation_scatter_{n1}_{n2}.pdf"
#     MPT.plot.plot_correlation_scatter(
#         f1,
#         f2,
#         out,
#         weights=dq_full_pop,
#         label1=l1,
#         label2=l2,
#         clip_to_greater_zero=dq_full_tmat,
#         title=f"{l1} - {l2} Correlation",
#     )
