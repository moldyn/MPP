#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Plot the states in the course of a trajectory.'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Input text file containing the sequence of states'
    )
    parser.add_argument(
        '-s',
        '--save',
        type=str,
        help='Optional: Save the plotted figure to a PNG file',
        default=None
    )
    parser.add_argument(
        '-g',
        '--histogram',
        action='store_true',
        help='Optional: Plot a histogram of the trajectory'
    )
    parser.add_argument(
        '-n',
        '--normalize',
        action='store_true',
        help='Optional: Normalize histogram'
    )
    return parser.parse_args()

def plot_states(states, save_path=None):
    plt.figure(figsize=(30, 4))
    #plt.plot(states, marker='o')
    plt.scatter(np.arange(len(states)), states, marker='o', s=3)
    plt.title('Trajectory of States')
    plt.xlabel('Step')
    plt.ylabel('State')
    
    if save_path:
        plt.savefig(save_path)
        print(f'Figure saved to {save_path}')
    else:
        plt.show()

def plot_histogram(states, save_path=None, norm=False):
    unique_states = sorted(set(states)) 
    #unique_states.append(unique_states[-1]+1)
    max_state = unique_states[-1]+1
    plt.figure()
    plt.hist(
        states,
        bins=np.arange(1, max_state)-0.5,
        density=norm,
        edgecolor='black',
        rwidth=0.8
    )
    plt.title('Histogram of States')
    plt.xlabel('State')
    plt.ylabel('Frequency')
    
    if save_path:
        plt.savefig(save_path)
        print(f'Histogram saved to {save_path}')
    else:
        plt.show()

def main():
    args = parse_arguments()
    states = np.loadtxt(args.input_file)

    if args.histogram:
        plot_histogram(states, args.save, args.normalize)
    else:
        plot_states(states, args.save)

if __name__ == "__main__":
    main()
