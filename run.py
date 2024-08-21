#!/usr/bin/env python

import os
import time
import inspect

import numpy as np
import scipy as scy
import matplotlib.pyplot as plt

import MPT

def run(out):
    traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
    feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
    multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")
    #multi_feature_traj_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2"
    cluster_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindist2.mosaic_clusters"
   
    lagtime = 50

    mpt_kernel = MPT.kernel.MPTKernel()
    #smpt_kernel = MPT.kernel.SMPTKernel(method="n", param=5)
    # smpt_kernel = MPT.kernel.SMPTKernel(method="p", param=1, c=0.0)
    # feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05)
    kl_kernel = MPT.kernel.KLKernel()

    mpt = MPT.MPT(traj, lagtime)
    mpt.mpt(mpt_kernel)
    mpt.add_feature(feature_traj)
    mpt.assign_macrostates(0.005, 0.5)

    smpt = MPT.MPT(traj, lagtime)
    smpt.mpt(kl_kernel)
    smpt.add_feature(feature_traj)
    smpt.assign_macrostates(0.02, 0.5)
    
    # smpt = MPT.MPT(traj, lagtime)
    # smpt.mpt(smpt_kernel, feature_kernel=feature_kernel, n=10)
    # #smpt.mpt(smpt_kernel, n=1000)
    # smpt.add_feature(feature_traj)
    # smpt.assign_macrostates(0.005, 0.5)

    MPT.plot.report_1v1(smpt, mpt, multi_feature_traj, cluster_file, out)

    return mpt, smpt

def run_(out):
    traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
    feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
    multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")

    lagtime = 50

    mpt_kernel = MPT.kernel.MPTKernel()
    kl_kernel = MPT.kernel.KLKernel()

    mpt = MPT.MPT(traj, lagtime)
    mpt.mpt(mpt_kernel)
    mpt.add_feature(feature_traj)
    mpt.assign_macrostates(0.02, 0.65)
    
    mpt_kl = MPT.MPT(traj, lagtime)
    mpt_kl.mpt(kl_kernel)
    mpt_kl.add_feature(feature_traj)
    mpt_kl.assign_macrostates(0.02, 0.65)

    mpt_kl.macrostate_assignment[0][7] += mpt_kl.macrostate_assignment[0][8]

    MPT.plot.evaluate_stochastic_clustering(mpt, mpt_kl, out)

    return mpt, mpt_kl

def run_(out):
    # Kullback-Leibler; Mosaic
    traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
    feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
    multi_feature_traj_raw = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")
    multi_feature_traj = multi_feature_traj_raw < 0.45
    out_base = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/"

    lagtime = 50

    states = np.unique(traj)
    state_feature = np.zeros((len(states, multi_feature_traj.shape[1])), dtype=np.float32)
    for s in states:
        state_feature[s-1] = multi_feature_traj[traj==s].mean(axis=0)
    
    mpt_kernel = MPT.kernel.MPTKernel()

    mpt = MPT.MPT(traj, lagtime)
    mpt.mpt(mpt_kernel)

    mpt.add_feature(feature_traj)
    mpt.assign_macrostates(0.005, 0.5)

def run_(out):
    traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
    feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
    multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")
    
    out_base = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/"

    lagtime = 50

    mpt_kernel = MPT.kernel.MPTKernel()
    kl_kernel = MPT.kernel.KLKernel()

    mpt = MPT.MPT(traj, lagtime)
    mpt.mpt(mpt_kernel)
    mpt.add_feature(feature_traj)
    mpt.assign_macrostates(0.005, 0.5)
    
    smpt = MPT.MPT(traj, lagtime)
    smpt.from_Z(out_base + "hp35_stoch_n2_1k.Z.npy")
    smpt.add_feature(feature_traj)
    smpt.assign_macrostates(0.005, 0.5)

    MPT.plot.evaluate_stochastic_clustering(mpt, smpt, out)

    return mpt, smpt

def run_(out):
    traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
    feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
    multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")

    lagtime = 50

    mpt_kernel = MPT.kernel.MPTKernel()
    smpt_kernel = MPT.kernel.SMPTKernel(method="n", param=2)
    feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.1)

    mpt = MPT.MPT(traj, lagtime)
    mpt.mpt(smpt_kernel, feature_kernel=feature_kernel, n=1000)
    mpt.add_feature(feature_traj)
    mpt_det.assign_macrostates(0.005, 0.5)
    mpt_det.calc_time_scales()

    mpt = MPT.MPT(traj, lagtime)
    mpt.mpt(smpt_kernel, n=1000, feature_kernel=feature_kernel)
    mpt.add_feature(feature_traj)
    mpt.assign_macrostates(0.005, 0.5)
    mpt.calc_time_scales()

    its = mpt.time_scales / mpt_det.time_scales

    fig, axs = plt.subplots(1, 4, figsize=(10, 4.5), sharey=True)
    for i, ax in enumerate(axs[:-1]):
        ax.hist(its[:, i], bins=20)
        ax.set_title(f'its {i+1}')
    axs[-1].hist(its.mean(axis=1), bins=20)
    axs[-1].set_title(f'Mean its {1}-{i+1}')

    fig.supxlabel(r"Implied Timescale Similarity $\left(\frac{t_\mathrm{stoch}}{t_\mathrm{det}}\right)$")
    fig.supylabel('Count of Clusterings')
    fig.suptitle(f'{mpt.n_runs} clusterings, n=2, FNC, sigma=0.1')
    plt.tight_layout()
    plt.savefig(out)

    return its, mpt, mpt_det

def run_(out):
    traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
    feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
    multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")

    lagtime = 50

    mpt_kernel = MPT.kernel.MPTKernel()
    smpt_kernel = MPT.kernel.SMPTKernel(method="n", param=2)
    feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.01)
    #feature_kernel = MPT.kernel.FeatureKernel(multi_feature_traj, traj, sigma=0.5)

    mpt = MPT.MPT(traj, lagtime)
    mpt.mpt(mpt_kernel, feature_kernel=feature_kernel)
    mpt.add_feature(feature_traj)
    mpt.assign_macrostates(0.005, 0.5)
    # mpt.plot(out)

    # print(scy.stats.pearsonr(np.log(np.diag(mpt.tmat)), mpt.feature))
    #
    # plt.scatter(mpt.feature, np.diag(mpt.tmat), s=1)
    # plt.title("Feature / Transition Probability Correlation")
    # plt.xlabel("Feature Distance")
    # plt.ylabel("Transition Probability")
    # # plt.yscale("log")
    # plt.savefig(out)

    feature = np.expand_dims(mpt.feature, -1)
    ft = scy.spatial.distance_matrix(feature, feature, p=1)
    mask = np.where((ft > 0) & (mpt.tmat > 0))
    # tmat = np.log(mpt.tmat[mask])
    tmat = mpt.tmat[mask]
    # print(len(mask[0]))
    # print(scy.stats.pearsonr(tmat, ft[mask]))

    # plt.hist(ft.flatten(), bins=30)
    plt.scatter(ft[mask].flatten(), tmat.flatten(), s=0.1, alpha=0.3)
    plt.title("Feature / Transition Probability Correlation")
    plt.xlabel("Feature Distance")
    plt.ylabel("Transition Probability")
    plt.yscale("log")
    plt.savefig(out)

def run_(out):
    traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
    feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
    multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")

    lagtime = 50

    mpt_kernel = MPT.kernel.MPTKernel()
    smpt_kernel = MPT.kernel.SMPTKernel(method="n", param=2)
    feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.1)
    #feature_kernel = MPT.kernel.FeatureKernel(multi_feature_traj, traj, sigma=0.5)

    mpt = MPT.MPT(traj, lagtime)
    mpt.mpt(mpt_kernel, feature_kernel=feature_kernel)
    mpt.add_feature(feature_traj)
    mpt.assign_macrostates(0.005, 0.5)
    mpt.plot(out)

def run_(out):
    traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
    feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
    multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")

    lagtime = 50

    mpt_kernel = MPT.kernel.MPTKernel()
    smpt_kernel = MPT.kernel.SMPTKernel(method="n", param=2)
    #smpt_kernel = MPT.kernel.SMPTKernel(method="p", param=0.5)
    feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.1)

    mpt = MPT.MPT(traj, lagtime)
    mpt.mpt(mpt_kernel)
    mpt.add_feature(feature_traj)
    mpt.assign_macrostates(0.005, 0.5)

    # Non-batched production
    # mpt1 = MPT.MPT(traj, lagtime)
    # mpt1.mpt(smpt_kernel, n=500)
    # mpt1.add_feature(feature_traj)
    # mpt1.assign_macrostates(0.005, 0.5)
    # mpt1.macro_to_micro_feature()
    # mpt1.plot_macro_feature(
    #     out,
    #     [(
    #         mpt.macrostate_assignment[0],
    #         mpt.macrostate_feature[0],
    #         "r",
    #         "Reference",
    #         mpt.full_pop[:, :mpt.n_states],
    #     )]
    # )

    # batched production
    batches = 1
    n = 100
    macro_features = np.zeros((mpt.n_states, batches * n))
    pop = np.zeros((mpt.n_states, batches * n))
    for i in range(batches):
        print(f'Batch {i+1} / { batches }')
        mpt1 = MPT.MPT(traj, lagtime)
        mpt1.mpt(smpt_kernel, n=n, feature_kernel=feature_kernel)
        mpt1.add_feature(feature_traj)
        mpt1.assign_macrostates(0.005, 0.5)
        mpt1.macro_to_micro_feature()
        macro_features[:, i * n:(i+1) * n] = mpt1.micro_feature
        pop[:, i * n:(i+1) * n] = mpt1.full_pop[:, :mpt1.n_states].T

    MPT.plot.plot_macro_feature(
        macro_features,
        out,
        [(
            mpt.macrostate_assignment[0],
            mpt.macrostate_feature[0],
            "r",
            "Reference",
            mpt.full_pop[:, :mpt.n_states],
        )],
        pop=pop,
    )

def main():
    out_base = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/"
    #out = out_base + "img/hp35_det_KL_thr_similarity_89_t.pdf"
    
    #out = out_base + "img/hp35_smpt_1k_n2_report"
    #out = out_base + "img/hp35_smpt_1k_n2_feature_s10_report"
    #out = out_base + "img/hp35_smpt_1k_n2_feature_s05_report"
    #out = out_base + "img/hp35_smpt_1k_p50_feature_s10_report"
    #out = out_base + "img/hp35_smpt_1k_n5_feature_s10_report"
    #out = out_base + "img/hp35_smpt_1k_p90_feature_s05_c15_report"
    out = out_base + "img/hp35_mpt_kl_report"
    start = time.time()
    ret = run(out)
    execution_time = time.time() - start
    source_code = inspect.getsource(run)
    with open(out + '.code', 'w') as f:
        # Write the file location as a comment
        f.write(f'# File location: {os.path.abspath(out)}\n')
        # Write the current timestamp as a comment
        f.write(f'# Timestamp: {time.ctime()}\n')
        # Write the execution time as a comment
        if execution_time > 7200:
            f.write(f'# Execution time: {execution_time/3600:.2f} h\n')
        if execution_time > 60:
            f.write(f'# Execution time: {execution_time/60:.2f} m\n')
        else:
            f.write(f'# Execution time: {execution_time:.2f} s\n')
        f.write('\n')
        # Write the unindented code to the text file
        f.write(source_code)
    return ret

ret = main()
