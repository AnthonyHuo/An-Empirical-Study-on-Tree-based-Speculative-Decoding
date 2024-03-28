import numpy as np
import random
import torch

class TreeNode:
    def __init__(self, logit=0, parent=None, depth=0):
        self.logit = logit
        self.parent = parent
        self.children = []
        self.depth = depth

    def compute_cumulative_logit(self):
        """Compute the cumulative logit from this node up to the root."""
        node = self
        cumulative_logit = 0
        while node:
            cumulative_logit += node.logit
            node = node.parent
        return cumulative_logit
    
    def remove_child(self, child):
        self.children.remove(child)

def get_all_nodes(root):
    """Get all nodes in the tree starting from root."""
    all_nodes = []
    stack = [root]
    while stack:
        current_node = stack.pop()
        all_nodes.append(current_node)
        stack.extend(current_node.children)
    return all_nodes

def prune_tree(root, keep=128):
    """Prune the tree to keep only the top 'keep' nodes based on cumulative logits."""
    all_nodes = get_all_nodes(root)
    all_nodes_with_logitss = [(node, node.compute_cumulative_logits()) for node in all_nodes]
    sorted_nodes = sorted(all_nodes_with_logitss, key=lambda x: x[1], reverse=True)[:keep]
    keep_nodes = set(node[0] for node in sorted_nodes)
    
    for node in all_nodes:
        node.children = [child for child in node.children if child in keep_nodes]
    
    return [node[0] for node in sorted_nodes]

def expand_and_prune(root, levels=1, expansion_factor=128, prune_to=128):
    last_level_grow_map = None  # Initialize as None and will be updated on the last level
    
    for level in range(levels):
        all_nodes = get_all_nodes(root)
        
        # Expand each node
        for node in all_nodes:
            for _ in range(expansion_factor):
                child_logits = random.random()  # Simulate a logits for the child node
                child_depth = node.depth + 1 
                child = TreeNode(logit=child_logit, parent=node, depth=child_depth)
                node.children.append(child)
        
        # Prune the tree at every level
        pruned_nodes = prune_tree(root, keep=prune_to)
    
    return pruned_nodes, last_level_grow_map

# Initialize the root node
root = TreeNode(logit=0)  # Assuming root starts with a logit of 0

# Perform the expansion and pruning process
top_nodes_after_pruning, grow_maps = expand_and_prune(root, levels=6, expansion_factor=5, prune_to=5)

# Printing some information about the top nodes
print(f"Top nodes after pruning: {len(top_nodes_after_pruning)}")
for i, node in enumerate(top_nodes_after_pruning[:10]):  # Print the first 10 for brevity
    print(f"Node {i+1}: logit = {node.logit}, Cumulative logit = {node.compute_cumulative_logit()}")
print(grow_maps)
