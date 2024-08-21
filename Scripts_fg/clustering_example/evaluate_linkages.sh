for linkage in /data/MPP_MC/Lukas/Scripts/clustering_example/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153_*_linkage.dat
do
     echo $linkage
     /bin/python3 /data/MPP_MC/Lukas/Scripts/process_mpp.py --linkage $linkage --tlag 50 --cut-params 0.005 0.50 --state-traj /data/MPP_MC/Lukas/Scripts/clustering_example/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153 --fraction-of-native-contacts /data/MPP_MC/Lukas/Scripts/clustering_example/hp35.mindists2.gaussian10f.q --hide-labels
     
done
