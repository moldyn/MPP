#!/usr/bin/env python3

"""
Plot a spring model graph of a state trajectory
"""

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot a state trajectory as spring model."
        )
    )
    parser.add_argument(
        "state_trajectory",
        type=str,
        help="Path to the state trajectory file"
    )
    parser.add_argument(
        "-t",
        "--lag-time",
        type=int,
        help=(
            "Lag time for the markov state model"
        )
    )
    parser.add_argument(
        "-s",
        "--save_plot",
        type=str,
        help="Save plot to file."
    )
    return parser.parse_args()

def analyze_trajectory(traj_file, tlag):
    # Load state trajectory to numpy array
    traj = np.loadtxt(traj_file, dtype=int, comments='#')

    # Calculate transnition matrix
    matrix, permutation = mh.msm.estimate_markov_model(traj, tlag)
    tmat = normalize(matrix)

    # Get list of states and populations
    states, pop = np.unique(traj, return_counts=True)
    pop = pop/np.sum(pop)

def main():
    # Parse arguments
    args = parse_args()

    # Calculate transnition matrix and populations
    tmat, states, pop = analyze_trajectory(args.state_trajectory, args.lag_time)


if __name__ == "__main__":
    main()
