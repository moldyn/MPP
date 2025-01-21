#!/usr/bin/env python

import os
import time
import inspect

import numpy as np
import scipy as scy
import matplotlib.pyplot as plt

import MPT


def run_(out):
    traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
    feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
    multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")
    #multi_feature_traj_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2"
    cluster_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindist2.mosaic_clusters"
   
    lagtime = 50

    mpt_kernel = MPT.kernel.MPTKernel(similarity="P", a=1, b=0, c=80, term="+")
    # feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05)
    feature_kernel = MPT.kernel.MultiFeatureKernel(multi_feature_traj, traj, similarity="JS")

    mpt = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
    mpt.mpt(mpt_kernel, feature_kernel=feature_kernel)
    # mpt.mpt(mpt_kernel)
    
    mpt.plot(out + "dendrogram.pdf", scale=1)
    mpt.plot_implied_timescales(out + "timescales.pdf", scale=1)
    mpt.plot_sankey(out + "sankey.pdf", scale=1)
    mpt.plot_contact_rep(multi_feature_traj, cluster_file, out + "contact_rep.pdf", scale=1)
    
    np.savetxt(out + "linkage.txt", mpt.linkage)

    return mpt

def run_(out):
    traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
    feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
    multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")
    #multi_feature_traj_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2"
    cluster_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindist2.mosaic_clusters"
   
    lagtime = 50

    mpt_kernel = MPT.kernel.MPTKernel(similarity="KL", a=0, b=1, c=1, term="*")
    # feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05)
    feature_kernel = MPT.kernel.MultiFeatureKernel(multi_feature_traj, traj, similarity="JS")

    mpt = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
    mpt.mpt(mpt_kernel, feature_kernel=feature_kernel)
    # mpt.mpt(mpt_kernel)
    
    mpt.plot(out + "dendrogram.pdf", scale=1)
    mpt.plot_implied_timescales(out + "timescales.pdf", scale=1)
    mpt.plot_sankey(out + "sankey.pdf", scale=1)
    mpt.plot_contact_rep(multi_feature_traj, cluster_file, out + "contact_rep.pdf", scale=1)
    
    np.savetxt(out + "linkage.txt", mpt.linkage)

    return mpt






def run_(out):
    traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
    feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
    multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")
    multi_feature_traj_bool = multi_feature_traj < 0.45
    #multi_feature_traj_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2"
    cluster_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindist2.mosaic_clusters"
   
    lagtime = 50

    mpt_kernel = MPT.kernel.MPTKernel(similarity="P", a=0, b=0, c=1, term="*")
    # feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05)
    feature_kernel = MPT.kernel.MultiFeatureKernel(multi_feature_traj_bool, traj, similarity="JS")

    mpt = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
    mpt.mpt(mpt_kernel, feature_kernel=feature_kernel)
    # mpt.mpt(mpt_kernel)
    
    mpt.plot(out + "dendrogram.pdf", scale=1)
    mpt.plot_implied_timescales(out + "timescales.pdf", scale=1)
    mpt.plot_sankey(out + "sankey.pdf", scale=1)
    mpt.plot_contact_rep(multi_feature_traj, cluster_file, out + "contact_rep.pdf", scale=1)
    
    np.savetxt(out + "linkage.txt", mpt.linkage)

    return mpt

def run_(out):
    traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
    feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
    multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_source/hp35.mindist2.gaussian10f")
    multi_feature_traj_bool = multi_feature_traj < 0.45
    #multi_feature_traj_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2"
    cluster_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindist2.mosaic_clusters"
   
    lagtime = 50

    mpt_kernel = MPT.kernel.MPTKernel(similarity="P", a=0, b=0, c=1, term="*")
    # feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05)
    feature_kernel = MPT.kernel.MultiFeatureKernel(multi_feature_traj_bool, traj, similarity="JS")

    mpt = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
    mpt.mpt(mpt_kernel, feature_kernel=feature_kernel)
    # mpt.mpt(mpt_kernel)

    mpt.plot(out + "dendrogram.pdf", scale=1)
    mpt.plot_implied_timescales(out + "timescales.pdf", scale=1)
    mpt.plot_sankey(out + "sankey.pdf", scale=1)
    mpt.plot_contact_rep(multi_feature_traj, cluster_file, out + "contact_rep.pdf", scale=0.8)

    np.savetxt(out + "linkage.txt", mpt.linkage)

    return mpt

def run_(out):
    traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
    feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
    multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")
    #multi_feature_traj_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2"
    cluster_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindist2.mosaic_clusters"
   
    lagtime = 50

    mpt_kernel = MPT.kernel.MPTKernel(similarity="P", a=1, b=0, c=1, term="*")
    feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05)
    # feature_kernel = MPT.kernel.MultiFeatureKernel(multi_feature_traj, traj, similarity="JS")

    mpt = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
    mpt.mpt(mpt_kernel, feature_kernel=feature_kernel)
    # mpt.mpt(mpt_kernel)

    # mpt.plot(out + "dendrogram.pgf", scale=1)
    # mpt.plot_implied_timescales(out + "timescales.pgf", scale=1)
    # mpt.plot_sankey(out + "sankey.pgf", scale=1)
    # mpt.plot_contact_rep(multi_feature_traj, cluster_file, out + "contact_rep.pgf", scale=0.8)
    
    # mpt.n_i = np.argmax(mpt.timescales[:, 0])
    mpt.plot(out + "dendrogram.pdf", scale=1)
    mpt.plot_implied_timescales(out + "timescales.pdf", scale=0.9)
    mpt.plot_sankey(out + "sankey.pdf", scale=0.9)
    mpt.plot_contact_rep(multi_feature_traj, cluster_file, out + "contact_rep.pdf", scale=0.8)
    # mpt.plot_relative_implied_timescales(out + "relative_timescales.pdf")
    
    np.savetxt(out + "linkage.txt", mpt.linkage)

    return mpt

def run_(out):
    traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
    feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
    multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")
    #multi_feature_traj_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2"
    cluster_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindist2.mosaic_clusters"
   
    lagtime = 50

    mpt_kernel = MPT.kernel.MPTKernel(similarity="P", a=1, b=0, c=1, term="*")
    # feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05)
    feature_kernel = MPT.kernel.MultiFeatureKernel(multi_feature_traj, traj, similarity="JS")

    mpt = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
    mpt.mpt(mpt_kernel, feature_kernel=feature_kernel)
    # mpt.mpt(mpt_kernel)

    # mpt.plot(out + "dendrogram.pgf", scale=1)
    # mpt.plot_implied_timescales(out + "timescales.pgf", scale=1)
    # mpt.plot_sankey(out + "sankey.pgf", scale=1)
    # mpt.plot_contact_rep(multi_feature_traj, cluster_file, out + "contact_rep.pgf", scale=0.8)
    
    # mpt.n_i = np.argmax(mpt.timescales[:, 0])
    mpt.plot(out + "dendrogram.pdf", scale=1)
    mpt.plot_implied_timescales(out + "timescales.pdf", scale=1)
    mpt.plot_sankey(out + "sankey.pdf", scale=1)
    mpt.plot_contact_rep(multi_feature_traj, cluster_file, out + "contact_rep.pdf", scale=1)
    # mpt.plot_relative_implied_timescales(out + "relative_timescales.pdf")
    
    np.savetxt(out + "linkage.txt", mpt.linkage)

    return mpt

def run(out):
    traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
    feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
    # multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")
    multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_source/hp35.mindist2.gaussian10f")
    multi_feature_traj_bool = multi_feature_traj < 0.45
    #multi_feature_traj_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2"
    cluster_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindist2.mosaic_clusters"
   
    lagtime = 50

    mpt_kernel = MPT.kernel.MPTKernel(similarity="KL", a=0, b=1, c=1, term="*", param=2)
    # feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05)
    feature_kernel = MPT.kernel.MultiFeatureKernel(multi_feature_traj_bool, traj, similarity="JS")

    mpt = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
    # mpt.mpt(mpt_kernel, feature_kernel=feature_kernel)
    mpt.mpt(mpt_kernel, feature_kernel=feature_kernel, n=1000)

    # mpt.plot(out + "dendrogram.pgf", scale=1)
    # mpt.plot_implied_timescales(out + "timescales.pgf", scale=1)
    # mpt.plot_sankey(out + "sankey.pgf", scale=1)
    # mpt.plot_contact_rep(multi_feature_traj, cluster_file, out + "contact_rep.pgf", scale=0.8)
    
    mpt.n_i = np.argmax(mpt.timescales[:, 0])
    mpt.plot(out + "dendrogram.pdf", scale=1)
    mpt.plot_implied_timescales(out + "timescales.pdf", scale=1)
    mpt.plot_sankey(out + "sankey.pdf", scale=1)
    mpt.plot_contact_rep(multi_feature_traj, cluster_file, out + "contact_rep.pdf", scale=1)
    mpt.plot_relative_implied_timescales(out + "relative_timescales.pdf")
    
    np.savetxt(out + "linkage.txt", mpt.linkage)

    return mpt


def main():
    out_base = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/thesis/"
    # name = "hp35_det_JS_contacts_add_50/"
    # name = "hp35_det_KL_JS_contacts_mul/"
    # name = "hp35_det_only_JS_contacts/"
    # name = "hp35_det_KL_fnc_1_mul/"
    # name = "hp35_det_fnc_1/"
    # name = "hp35_det_JS_contacts_mul/"
    name = "hp35_stoch_n2_KL_JS_contacts/"
    #out = out_base + "img/hp35_det_KL_thr_similarity_89_t.pdf"
    
    out = out_base + name
    # out = out_base + "img/hp35_smpt_n2_s05_b2"
    # out = out_base + "img/hp35_smpt_c15_s05_b2"
    start = time.time()
    ret = run(out)
    execution_time = time.time() - start
    source_code = inspect.getsource(run)
    with open(out + f'/{os.path.basename(out)}.code', 'w') as f:
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
