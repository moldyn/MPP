from pymol import cmd
import sys
import os


def load_residue_indices(file_name):
    residue_indices = []
    with open(file_name, "r") as f:
        for line in f:
            residue_indices.append(line.rstrip().split(" "))
    return residue_indices


random_frames_file = sys.argv[1]
out_file = sys.argv[2]
i_macrostate = int(os.path.basename(random_frames_file).split(".")[0]) - 1

cmd.load(random_frames_file, "frames")
cmd.split_states("frames")
cmd.delete("frames")

if sys.argv[3] == "none":
    residue_selection = ""
else:
    residue_indices = load_residue_indices(sys.argv[3])[i_macrostate]
    residue_selection = f" and ({' or '.join([f'resi {i}' for i in residue_indices])})"

objects = cmd.get_object_list(selection="(all)")
# ref = objects[0]
cmd.select("ref", f"name CA and {objects[0]}" + residue_selection)

for o in objects[1:]:
    cmd.select(f"ca_{o}", f"name CA and {o}" + residue_selection)
    cmd.align(f"ca_{o}", "ref")
cmd.spectrum("resi", "rainbow")
cmd.orient()
cmd.zoom(complete=1)
cmd.png(out_file, 800, 800, ray=1)
