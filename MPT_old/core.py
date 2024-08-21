"""
core.py
=======

Core functions for MPT class
"""

__all__ = [
    "find_all_nodes",
]

from numba import njit, types
from numba.typed import List
import numpy as np
from typing import List as TypedList, Tuple

@njit
def build_tree(linkage_matrix: np.ndarray) -> List:
    """
    Builds a tree represented as an adjacency list from a linkage matrix.
    
    Args:
        linkage_matrix (np.ndarray): A 2D array where each row represents a
        (child, parent) pair.
        
    Returns:
        List: An adjacency list representation of the tree.
    """
    # Determine the size of the linkage matrix to allocate memory
    max_node = int(np.max(linkage_matrix))
    tree = List()
    for _ in range(max_node + 1):
        tree.append(List.empty_list(types.int64))
    
    # Populate the adjacency list
    for i in range(linkage_matrix.shape[0]):
        child, parent = linkage_matrix[i]
        tree[parent].append(child)
    
    return tree

@njit
def find_all_nodes(
        linkage_matrix: np.ndarray[int],
        root: int
    ) -> TypedList[int]:
    """
    Finds all nodes in the tree starting from the given root.
    
    Args:
        linkage_matrix (np.ndarray): A 2D array where each row represents a
                (child, parent) pair.
        root (int): The root node to start the traversal from.
        
    Returns:
        List[int]: A list of all nodes reachable from the root.
    """
    tree = build_tree(linkage_matrix)
    all_nodes = List.empty_list(types.int64)
    stack = List.empty_list(types.int64)
    stack.append(root)
    
    while len(stack) > 0:
        node = stack.pop()
        all_nodes.append(node)
        for child in tree[node]:
            stack.append(child)
    
    return all_nodes
