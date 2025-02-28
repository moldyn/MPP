from pymol import cmd
import sys
import os

random_frames_dir = sys.argv[1]
files = os.listdir(random_frames_dir)

pdb_files = [f for f in files if f.endswith(".pdb")]
states = set([s.split("_")[0] for s in pdb_files])
state_files = [[s for s in pdb_files if s.startswith(state + "_")] for state in states]

for s, name in zip(state_files, states):
    for f in s:
        cmd.load(os.path.join(random_frames_dir, f))
    cmd.spectrum("resi", "rainbow")
    cmd.orient()
    cmd.zoom(complete=1)
    cmd.png(os.path.join(random_frames_dir, name + ".png"), 800, 800, ray=1)
    cmd.delete("all")
