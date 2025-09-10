#!/usr/bin/bash
# snakemake --cores 'all' --sdm conda -p results/aSyn_{kmeans,rdc}{,_200ns,_800ns}_stoch/{t,t_js,kl,kl_js}/{dendrogram,sankey,contacts,timescales,macrotraj,ck_test,rmsd,delta_rmsd}.p{ng,df} results/PDZ3_7_stoch/{t,t_js,kl,kl_js}/{dendrogram,sankey,contacts,timescales,macrotraj,ck_test,rmsd,delta_rmsd}.p{ng,df}
# snakemake --cores 'all' --sdm conda -p results/aSyn_{kmeans,rdc}{,_200ns,_800ns}_stoch/{t,t_js,kl,kl_js}/Z.npy results/PDZ3_7_stoch/{t,t_js,kl,kl_js}/Z.npy
# snakemake --cores 'all' --sdm conda -p results/aSyn_{kmeans,rdc}{,_200ns,_800ns}/{t,t_js,kl,kl_js}/random_frames results/{HP35,PDZ3_7}/{t,t_js,kl,kl_js}/random_frames --cache -f
# snakemake --cores 'all' --sdm conda -p results/aSyn_{kmeans,rdc}{,_200ns,_800ns}/{t,t_js,kl,kl_js}/macrostates.txt results/{HP35,PDZ3_7}/{t,t_js,kl,kl_js}/macrostates.txt --cache -f

# snakemake --cores 'all' --sdm conda -p results/aSyn_{kmeans,rdc}{,_200ns,_400ns,_800ns}/gpcca/{dendrogram,sankey,contacts,timescales,macrotraj,ck_test}.p{ng,df} results/PDZ3_7/gpcca/{dendrogram,sankey,contacts,timescales,macrotraj,ck_test}.p{ng,df} --cache
# snakemake --cores 'all' --sdm conda -p results/aSyn_{kmeans,rdc}_400ns/{t,t_js,kl,kl_js}/Z.npy
# snakemake --cores 'all' --sdm conda -p results/aSyn_{kmeans,rdc}_400ns/{t,t_js,kl,kl_js}/{dendrogram,sankey,contacts,timescales,macrotraj,ck_test,rmsd,delta_rmsd}.p{ng,df} --cache

# snakemake --cores 'all' --sdm conda -p results/{HP35,PDZ3_7}/gpcca/rmsd.npy
# snakemake --cores 'all' --sdm conda -p results/aSyn_{kmeans,rdc}{,_200ns,_400ns,_800ns}/gpcca/rmsd.npy
# snakemake --cores 'all' --sdm conda -p results/aSyn_kmeans/kl_js/{,delta_}rmsd.p{df,ng} -f -n

# snakemake --cores 'all' --sdm conda -p results/PDZ3_7/t/rmsd.npy -f

# snakemake --cores 'all' --sdm conda -p results/aSyn_rdc_200ns/{t,t_js,kl,kl_js,gpcca}/random_frames --cache -n
snakemake --cores 'all' --sdm conda -p results/aSyn_rdc_200ns/gpcca/random_frames --cache -f
# snakemake --cores 'all' --sdm conda -p results/aSyn_rdc_200ns/{t,t_js,kl,kl_js,gpcca}/macrostates_pdb.txt --cache -n -f
# snakemake --cores 'all' --sdm conda -p results/aSyn_rdc_200ns/{t,t_js,kl,kl_js,gpcca}/macrostates_pdb.txt --cache
