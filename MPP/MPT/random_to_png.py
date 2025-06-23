from pymol import cmd
import sys
import os


def load_residue_indices(file_name):
    residue_indices = []
    with open(file_name, "r") as f:
        for line in f:
            residue_indices.append(line.rstrip().split(" "))
    return residue_indices


random_frames_dir = sys.argv[1]
files = os.listdir(random_frames_dir)

pdb_files = [f for f in files if f.endswith(".pdb")]
states = set([s.split("_")[0] for s in pdb_files])
state_files = [[s for s in pdb_files if s.startswith(state + "_")] for state in states]

cmd.load(os.path.join(random_frames_dir, state_files[0][0]), "ref")

if len(sys.argv) == 3:
    residue_indices = load_residue_indices(sys.argv[2])
    residue_selection = f" and ({' or '.join([f'resi {i}' for i in residue_indices])})"
else:
    residue_selection = [""] * len(state_files)

for s, name, res_sel in zip(state_files, states, residue_selection):
    del_obj = []
    for f in s:
        cmd.load(os.path.join(random_frames_dir, f))
        n = f.split(".")[0]
        cmd.select(f"ca_{n}", f"name CA and {n}" + res_sel)
        cmd.align(f"ca_{n}", "ref")
        del_obj.append(n)
        del_obj.append(f"ca_{n}")
    cmd.spectrum("resi", "rainbow")
    cmd.orient()
    cmd.zoom(complete=1)
    cmd.png(os.path.join(random_frames_dir, name + ".png"), 800, 800, ray=1)
    cmd.delete(" ".join(del_obj))
