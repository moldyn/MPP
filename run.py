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

    p_js_4875 = {
        "a": -0.31390263446885075,
        "b": 0.7685878238249264,
        "c": 0.13041883715594693,
        "e": 5.682928465388565,
        "f": 1.4272292032636014,
    }
    p_js_loss = {
        "a": 0.6061912866743351,
        "b": 0.5945954305784469,
        "c": 0.9815886167424418,
        "e": 14.666986771456534,
        "f": 5.681534627126421,
    }

    mpt_kernel = MPT.kernel.MPTKernel(similarity="JS", **p_js_loss)
    smpt_kernel = MPT.kernel.MPTKernel(method="n", param=2)
    # smpt_kernel = MPT.kernel.MPTKernel(method="p", param=1, c=0.15)
    feature_kernel = MPT.kernel.FeatureKernel(multi_feature_traj, traj, sigma=0.05)
    # kl_kernel = MPT.kernel.KLKernel()

    mpt = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
    mpt.mpt(mpt_kernel)

    # smpt = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
    # smpt.mpt(smpt_kernel, feature_kernel=feature_kernel, n=1000)
    
    #MPT.plot.report_1v1(smpt, mpt, multi_feature_traj, cluster_file, out)
    # MPT.plot.report_stochastic(smpt, mpt, multi_feature_traj, cluster_file, out)
    MPT.plot.report(mpt, multi_feature_traj, cluster_file, out)

    return mpt

def main():
    out_base = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/"
    #out = out_base + "img/hp35_det_KL_thr_similarity_89_t.pdf"
    
    out = out_base + "img_v2/hp35_det_js_loss"
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
