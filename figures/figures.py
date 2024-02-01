import argparse
from datetime import datetime
import json
import math
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def log(str_to_log, verbose=False, always_log=False):
  if always_log or verbose:
    print(f'{datetime_str()}: {str_to_log}')


#### HELPER FUNCTIONS ####
def datetime_str():
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def date_str():
  return datetime.now().strftime("%Y_%m_%d")


def log(str_to_log, verbose=False, always_log=False):
  if always_log or verbose:
    print(f'{datetime_str()}: {str_to_log}')


def is_power_of_two(number):
  # Check if the number is greater than 0 and has only one set bit
  return (number > 0) and ((number & (number - 1)) == 0)


def get_alphas(df, draft, metric, temp=0.6, top_p=1.0, target_only=True):
  df_p = df[(df['Draft'] == draft) & (df['Temp'] == temp) & (df['Top-p'] == top_p) & (df['Metric name'] == metric)]
  alphas = np.array([0] + df_p['Metric value'].tolist())
  return alphas


#### PLOTTING FUNCTIONS ####
def legend():
  plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


class Node:
  def __init__(self, children=None):
    self.children = children if children is not None else []
    self.num_nodes_in_tree = 1 + sum(c.num_nodes_in_tree for c in self.children)
    self.depth = 1 + max([0] + [c.depth for c in self.children])

  def __str__(self, depth=1):
    if self.children:
      delim = '\n' + ''.join(['    '] * depth)
      children_str = ': ' + delim + delim.join(f"[{c.__str__(depth=depth+1)}]" for c in self.children)
    else:
      children_str = ''
    return f'[{(self.num_nodes_in_tree, self.depth)}{children_str}]'


def best_tree_dp_limit_branch_depth(M, D, alphas, memo={}, max_branch_width=8):
  """Dynamic programming solution."""
  if (M, D) in memo:
    return memo[(M, D)]
  if M == -1:
    memo[(M, D)] = (0.0, [0])
  elif M == 0:
    memo[(M, D)] = (1.0, [0])
  elif D == 0:
    exp_length = (1-alphas[1] ** M) / (1 - alphas[1])
    memo[(M, D)] = (exp_length, [-M])
  else:
    best_len = 1
    best_ks = [0]
    for k in range(1, max_branch_width + 1):
      curr_len, curr_ks = best_tree_dp_limit_branch_depth(
          math.floor(M/k) - 1,
          D - 1,
          alphas,
          memo=memo, 
          max_branch_width=max_branch_width,
      )
      if 1 + alphas[k] * curr_len > best_len:
        best_len = 1 + alphas[k] * curr_len
        best_ks = [k] + curr_ks
    memo[(M, D)] = (best_len, best_ks)
  return memo[(M, D)]


def run_dp_limit_branch_depth(alphas, max_branch_width=32, max_budget=10**6, max_branch_depth=16, verbose=True):
  memo = {}
  for D in range(max_branch_depth + 1):
    for M in range(max_budget + 1):
      exp_length, ks = best_tree_dp_limit_branch_depth(
          M, D, alphas, memo=memo, max_branch_width=max_branch_width)
      if is_power_of_two(M) and verbose:
        log(f'The best tree of size <= {M}', verbose=verbose)
        log(f'Expected length = {exp_length}, ks={ks}\n', verbose=verbose)
  
  exp_lengths = []
  D = max_branch_depth
  for M in range(max_budget + 1):
    exp_length, _ = memo[(M, D)]
    exp_lengths.append(exp_length)
  return np.array(exp_lengths)


def baselines(alphas, max_budget, max_branch_width):
  n = np.arange(1, max_budget + 1)[np.newaxis, :]
  k = np.arange(1, max_branch_width + 1)[:, np.newaxis]
  p = alphas[1:max_branch_width + 1, np.newaxis]
  
  ### Expected accepted length of independent chains.
  p1 = p[0, 0]
  single_chain = ((1 - p1 ** (1 + n)) / (1 - p1)).squeeze()
  independent_chains = np.max(1 + p * (1 - p1 ** (np.floor(n / k))) / (1 - p1), axis=0)
  
  # # Single chain (Max-branch depth=0)
  # exp_lengths_md0 = run_dp_limit_branch_depth(
  #     alphas, max_budget=max_budget, max_branch_width=max_branch_width, max_branch_depth=0)
  # # Independent chains (Max-branch depth=1)
  # exp_lengths_md1 = run_dp_limit_branch_depth(
  #     alphas, max_budget=max_budget, max_branch_width=max_branch_width, max_branch_depth=1)
  # Max-branch depth=2
  # exp_lengths_md2 = run_dp_limit_branch_depth(
  #     alphas, max_budget=max_budget, max_branch_width=128, max_branch_depth=2)
  # # Max-branch depth=2
  # exp_lengths_md3 = run_dp_limit_branch_depth(
  #     alphas, max_budget=max_budget, max_branch_width=64, max_branch_depth=3)

  ### Expected accepted length of k-trees (and best k-tree).
  # We start indexing from a binary tree (k=2).
  k2 = k[1:]
  p2 = p[1:]
  # k_tree = (1 - p2 ** (1 + np.floor(np.log(n)/np.log(k2)))) / (1 - p2)
  # We delete the np.floor for it to be smoother.
  k_tree = (1 - p2 ** (1 + np.log(n)/np.log(k2))) / (1 - p2)

  # best_k_tree = np.max(k_tree, axis=0)
  # Note that 
  return single_chain, independent_chains, k_tree


# C_L[L, M] = expected # accepted tokens, if root node has EXACTLY L children,
# and the total number of nodes in the tree is M.
# C[M] = max(C_L[:, M])
def best_tree_unbalanced(alphas, max_branch_width=32, max_budget=10**8, verbose=True, cache_memo_path=None):
  p = np.hstack(([0], alphas[1:] - alphas[:-1]))
  c_L = np.zeros((max_branch_width + 1, max_budget + 1))
  c = np.zeros(max_budget + 1)  # expected number accepted tokens per budget.
  c_L[0, 1] = 1
  c[1] = 1
  # best_new_node[L, M] = A pointer to the best node (tree root node) to add
  #   as the L^th child of the tree root with budget M and L children.
  # best_tree[M] = A pointer to the best node (tree root node) with M nodes.
  # best_tree[M] can be constructed by picking L* maximimizing c_L[L, M], then
  #   taking N = best_new_node[L*, M] as the last child of a new root node. Then
  #   taking best_new_node[L* - 1, M - N.num_nodes_in_tree], etc.
  best_new_node = {(0, 1): None}
  best_tree = {1: Node()}
  for M in range(2, max_budget + 1):
    # Is there a way to vectorize this loop?
    for L in range(1, min(max_branch_width + 1, M)):
      c_L[L, M] = np.max(c_L[L - 1, L:M] + p[L] * c[M - L:0:-1])
      argmax = np.argmax(c_L[L - 1, L:M] + p[L] * c[M - L:0:-1])
      # `M - L - argmax` is the number of nodes to assign to the L'th child
      # of the optimal (L,M) tree.
      best_new_node[L, M] = best_tree[M - L - argmax]
    c[M] = np.max(c_L[:, M])

    # We now construct the best tree of budget M.
    best_L = np.argmax(c_L[:, M])
    best_M_budget_tree_children = []
    remaining_budget = M
    for L in range(best_L, 0, -1):
      next_child = best_new_node[L, remaining_budget]
      best_M_budget_tree_children.insert(0, next_child)
      remaining_budget -= next_child.num_nodes_in_tree
    assert remaining_budget == 1
    best_tree[M] = Node(children=best_M_budget_tree_children)

    # Log progress every power of two iterations.
    if is_power_of_two(M):
      log(f'The best tree of size <= {M}', verbose=verbose)
      log(f'Expected length = {c[M]}\n', verbose=verbose)  
      if cache_memo_path:
        np.savez(cache_memo_path, c_L=c_L[:, :M+1], c=c[:M+1], M=M)
  return c, best_tree
