# Stochastic Clustering of HP35 and Analysis approaches
This folder contains all Scripts and datasets from the stochastic clustering approaches of HP35

##Scripts
1. `benchmark_its.py`: Benchmarking the implied timescales (its) and macrostate trajectories (`msmhelper.md.compare_discretization`) in a folder to a reference. The its are plotted in a histogram.
2. `benchmark_macrostates.py`: Compare the reference macrostates to stochastic macrostates individually with the similarity measures S_1, S_2 and S_3  from Report 9. Similarities are plotted in histograms for all macrostates. Code is not optimized in speed and readability!
3. `compmat.py`: Calculate the macrotrajectory similarity matrix of each macrotraj in folder. All trajs are compared to each other with `msmhelper.md.compare_discretization`.
4. `compmat_mosaic_clustering.py`: `MoSAIC` Clustering of the similarity matrix from stochastic clusterings.
5. `compmat_multiprocessing.py`: `compmat` script to calculate faster on multiple kernels.
6. `macrostates`: Calculation of the macrostates from a clustering with population and metastability cutoff values as in MPP+, without expensive plotting of dendrogram.
7. `MPT.py`: Deterministic Most-Probable-Transition algorithm as used in Bachelors-Thesis
8. `MPT_MCMC.py`: Stochastic MPT algorithm with MCMC step without any restrictions as used in Bachelors-Thesis
9. `MPT_MCMC_fnc.py`: Stochastic MPT algorithm with fraction of native contacts (FNC) scoring of probability distributions. Default values are set to produce same output as `MPT.py`. If exponent is set to 0, no FNC scoring happens. If probability cutoff is set to 1.0 the clustering is deterministic.
10. `MPT_MCMC_peaks.py`: Stochastic MPT algorithm which only considers a given amount of probability maxima. If only one maximum is considered, same output as in `MPT_MCMC.py`.
11. `process_mpp.py`: Code from HP35 Case Study to calculate macrostates and plot dendrogram of a clustering. Caution: macrotrajs get named differently. Therefore if this code and `macrostates.py`is run on the same datasets, macrotrajs can double.

> All files from Scripts are saved to the given folder or the folder of the relevant input files. 

> Stochastic Clusterings can have complex eigenvalues, resulting in non-markovian behaviour. If a stochastic clustering is considered, this needs to be checked first. 

##Datasets
1. HP35_reference_data: microstate-traj and FNC-traj of HP35 from case study, such as the clustering files from clustering with `MPT.py.
2. MCMC_test_HP35: Random clusterings to test and improve Scripts
3. MPT_MCMC_FNC: Stochastic clusterings for different parameters in FNC scoring. The name of the dataset folders name the used parameters. The variance (var) represents the variance of the FNC scoring function, the exponent (exp) represents the exponent in the scoring function, the cutoff (cut) represents the relative probability cutoff below which all probabilites are neglected. The datasets with 0.10 cutoff contain 1000 clusterings and feature `MoSAIC` clustering. All datasets with higher cutoffs have 4000 clusterings. All datasets have been benchmarked in respect to timescales and macrostates. A exemplary bash script to create and benchmark the datasets is also included.
4. MPT_MCMC_Peaks: Stochastic Clusterings with different number of considered peaks and cutoff probabilities. The dataset names represent the used parameters. All datasets contain 1000 clusterings. The 3Max clusterings were accidentially done twice and showed to be a stochastically insignificant sample. They feature the `MoSAIC` clustering. The 2Max clusterings have been used for more serious analysis. They only contain `MoSAIC` clustering for 0.10 and 0.20 cut parameters. All of these datasets are benchmarked in respect to timescales and macrostates.
5. Probability_analysis: Contains the plots of the state probabilities througout different methods of clustering. They are without restrictions, with only a cutoff, and with FNC scoring. The names show the respective parameters. Also contains the Scripts the distributions were plotted with. Scripts are not improved in readability!


> Each clustering consists of a linkage.dat file, and a .macrostates and .macrotraj file.

> Clusterings are named without their respective parameters. The Scripts for MPT_MCMC_fnc.py and MPT_MCMC_peaks.py have been improved in this regard.

> comparison_overview files have been overwritten with the max, min and mean of the number of macrostates. They can be reproduced easily by reading in the comparison_its and comparison_discretization files. 

