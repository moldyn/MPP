# script to create all minimal distances

# Print usage instructions
usage() {
	echo "Usage: $0 PARAM1 PARAM2 PARAM3 PARAM4 PARAM5"
	echo "  PARAM1: xtc file"
	echo "  PARAM2: pdb file"
	echo "  PARAM3: threshold for contact formation in % frames"
	echo "  PARAM4: indices file"
	echo "  PARAM5: base name for system, for output files"
}

# Check if the number of arguments is correct
if [ "$#" -ne 5 ]; then
	usage
	exit 1
fi

dir=$(dirname $0)
XTC="$1"    # e.g. for HP35 'villin_360K.xtc'
PDB="$2"    # e.g. for HP35 'villin_360K.pdb'
THR="$3"    # e.g. for HP35 0.3
INDEX="$4"  # e.g. for HP35 'all_indices.ndx'
SYSTEM="$5" # e.g. for HP35 'hp35'

LTHR=$(printf "$THR" | cut -d "-" -f 1)
if printf "$THR" | grep "-" >/dev/null 2>&1; then
	UTHR=$(printf "$THR" | cut -d "-" -f 2)
else
	UTHR="1"
fi

if [ ! -f "$XTC" ] || [ ! -f "$PDB" ] || [ ! -f "$INDEX" ]; then
	echo "Error: One or all of the specified files do not exist"
	usage
	exit 1
fi

# created files
IS_MINDIST="${SYSTEM}.is_mindist"
ATOMDIST="${SYSTEM}.all_thr${THR}_selected_atom_distances"

# estimate all mindist
if [[ ! -e "${IS_MINDIST}" ]]; then
	time python $dir/estimate_contacts.py \
		--top $PDB \
		--traj $XTC \
		--index $INDEX \
		--output $IS_MINDIST
fi

# plot and select formed mindist
time python $dir/extract_indices.py \
	--is-contacts $IS_MINDIST \
	--lower-threshold $LTHR \
	--upper-threshold $UTHR

# extract all atom pairwise distances of selected residues
time python $dir/contacts.py -f $XTC \
	-s $PDB \
	-n ${IS_MINDIST}.thr${LTHR}-${UTHR}.ndx \
	-o $ATOMDIST

# extract minimal distances between all atom pairs forming a contact more often
# than the the given treshold
time python $dir/extract_contacts.py --contacts $ATOMDIST \
	--index ${ATOMDIST}.atom_indices \
	--lower-threshold ${LTHR} \
	--upper-threshold ${UTHR} \
	--output ${SYSTEM}.mindist2
