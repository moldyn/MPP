# Documentation of the MPT package
## Features
- Perform the most probable path (MPP) algorithm on a given microstate trajectory (several independent trajectories).
- Store the Z matrix of the lumping, which determines the relation between the microstates. The macrostate assignment is done separately.
- For the lumping, four different similarities are supported:
  - T
  - KL
  - T/JS
  - KL/JS


# Old file descriptions
## core.py

Contains fundamental functions and classes for the package:

- class BinaryTreeNode
- function cluster

## kernel.py

Contains the kernel classes:

- class MPTKernel
- class FeatureKernel -> fnc
- class MultiFeatureKernel -> Jensen-Shannon contacts

## MPT.py

This is the main file. It contains the definition of the MPT class, which holds a lumping:

- class MPT

## plot.py

Contains functions to produce various plots:

- plot_tree
- evaluate_stochastic_clustering
- plot_implied_timescales
- _plot_impl_times
- plot_relative_implied_timescales_
- plot_relative_implied_timescales
- plot_heatmap
- plot_tmat
- plot_trans_time
- plot_macro_feature
- add_ref
- contact_rep
- plot_sankey
- plot_rmsd
- plot_state_trajectory
- report_stochastic
- report_1v1
- report
- report_

## utils.py

Contains small helper functions:

- feature_mean
- get_micro
- translate_traj
- macro_traj
- macro_tmat
- get_grid_format
- gmrq
- sparse_to_matrix
- Z_to_linkage
- linkage_to_Z
- merge_states
- get_macrostate_tmat_from_assignment
- dim
- get_macrostate_assignment_from_tree
- similarity
- kullback_leibler
- dq_kl
- jensen_shannon_div
- jensen_shannon
- shannon_entropy
- load_traj
- load_mean_frames
- find_mean_frame
- estimate_rmsd
- align_trajectory_to_reference
- calc_var
- opt_num_batches
- calc_rmsd
- write_pdbs
- find_state_lengths
