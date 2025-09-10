# The Most Probable Path Algorithm
This package implements the most probable path (MPP) algorithm, which is used to coarse-grain the number of discrete states of a Markov process. Based on a microstate trajectory, a Markov state model is estimated utilising [msmhelper](https://github.com/moldyn/msmhelper). Based on the transition probabilities and optional other descriptors, states are combined so that the final macrostates exhibit a given minimum population and metastability.

## Features
- Perform the most probable path (MPP) algorithm on a given microstate trajectory.
- Multi trajectory support
- Three levels of user interface
- Extensions to the basic algorithm
  - Similarity by Kullback-Leibler divergence of transition probabilities
  - Incorporation Jenson-Shannon divergence of a feature (e.g. contact distances)
  - Stochastic lumping
- Variety of analysis plots
- Easy adaptable to your needs

# Usage
Dependent on your skills and needs, this module can be used in at three different levels:
- In a [Snakemake](https://snakemake.github.io/) workflow where you only need to provide the configuration of your system and you're ready to go.
- Use the high-level Python interface (MPP.run) via the command line or in your own scripts.
- Integrate the central MPP.Lumping object in your Python pipeline.

## The Snakemake Workflow
Snakemake is a workflow organization tool and used here to provide a high level user interface. In general, you only need to tell snakemake which file you would like to have, e.g.

```bash
snakemake --cores 'all' --sdm conda -p data/HP35/results/{t,t_js,kl,kl_js}/dendrogram.pdf --cache
```

Explanation of some flags:

- `--cores` Number of cores to utilize. 'all' for all cores.
- `--software-deployment-method, --sdm` Use conda to deploy software environment.
- `--snakefile, -s` Use a non-local snakefile. Use e.g. `-s /data/evaluation/MPP/stochastic_MPP_Felix/tools/MPP/workflow/Snakefile`
- `--dry-run, --dryrun, n` Do not execute anything, just print out the jobs that would be run.
- `--cache` So rules may be eligible for caching. Enable it with this option.
- `--force, -f` Force recreation of the given file(s).
- `--printshellcmds, -p` Print out the shell commands that will be executed.

More information can be found here: [Snakemake Documentation](https://snakemake.readthedocs.io/en/stable/executing/cli.html)

Note that bash parameter expansion (the use of `{` and `}`) is possible to create e.g. several diagrams at once for multiple systems and/or setups.

## Data Directory Structure

```bash
data/
├── System1
│   ├── input
│   │   ├── clusters
│   │   ├── config.yml
│   │   ├── contact_distances_trajectory
│   │   ├── contacts.ndx
│   │   ├── microstate_trajectory
│   │   ├── README.md
│   │   ├── topology.pdb
│   │   ├── trajectory.xtc
│   │   └── view
│   └── results
│       ├── t
│       │   ├── output_file1
│       │   ├── output_file2
│       │   └── ...
│       ├── kl
│
├── System2
```



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
