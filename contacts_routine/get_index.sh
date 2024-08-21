#!/bin/bash
#
# First parameter is length of the protein
# Second parameter is delta residues

nres=$1
dres=$2
for ((i = 1; i <= nres - dres; i++)); do
	for ((j = i + dres; j <= nres; j++)); do
		printf "$i $j\\n"
	done
done
