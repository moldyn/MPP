# Tool to analyze states of a trajectory

Scripts starting with 'z' are unfinished or have severe issues.

## prepare_trajectory.py

The analysis requires the trejectory to be aligned. The alignement of an entire trajectory is quite intense, so, this script alignes a trajectory and writes the aligned trajectory for further use.

## state_rmsf.py

Analyze an aligned trajectory and a state trajectory and calculate the root mean square fluctuation per state (RMSF). The result is writte to a file (.rmsf) containing the state id, the number of frames in that state and the RMSF.

## lifetime.py

Determine the minimun, maximum and mean lifetime for each state.

## plot_lifetime.py

Plot the lifetime (.row file) created by lifetime.

## plot_lifetime_rmsf.py

Plot the lifetime against the RMSF.
