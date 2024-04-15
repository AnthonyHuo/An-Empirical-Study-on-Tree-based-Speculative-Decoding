import torch
from transformers import AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, DataCollatorForLanguageModeling, OPTForCausalLM, AutoTokenizer
from Llama import LlamaModel_Attn, LlamaForCausalLM_Attn
from torch.nn.functional import softmax
from copy import deepcopy
from tqdm import tqdm
from Tree import Tree
import time
import deepspeed
from Engine import GraphInferenceEngine, GraphInferenceEngineTG
from torch.profiler import profile, record_function, ProfilerActivity
from utils import get_sampling_logits, make_tree_attention_mask, select_kv, ChildrenAccept, get_residual, cat_kv, _make_causal_mask
class TreeNode:
    def __init__(self, logit=0, value=None, parent=None, depth=0, updated=False):
        self.logit = logit  # Logit value for this node
        self.value = value  # The actual token value this node represents
        self.parent = parent
        self.children = []
        self.depth = depth
        self.updated = updated 

    def compute_cumulative_logit(self):
        """Compute the cumulative logit from this node up to the root."""
        # if self.updated:
        #     # For nodes not updated in the current iteration, return only the node's logit
        #     return self.logit
        # else: 
        node = self
        cumulative_logit = 0
        weight = self.depth*(-0.09)+1
        while node:
            cumulative_logit += (node.logit*weight)
            # cumulative_logit += node.logit
            node = node.parent
            weight+=0.09
        return cumulative_logit

    def add_child(self, child):
        self.children.append(child)
def bfs_prune_tree(root, keep_nodes):
    """Prune the tree using a BFS approach, keeping only nodes in 'keep_nodes'.
    If a parent is not in 'keep_nodes', it and all its children are automatically pruned."""
    queue = [root]
    while queue:
        current_node = queue.pop(0)
        # Proceed only if the current node is in 'keep_nodes'
        if current_node in keep_nodes:
            # Filter children to keep only those in 'keep_nodes'
            filtered_children = [child for child in current_node.children if child in keep_nodes]
            current_node.children = filtered_children
            # Add filtered children to the queue
            queue.extend(filtered_children)
def flatten_tree(root):
    """Flatten the tree into a list using BFS and return the list."""
    all_nodes = []
    queue = [root]
    while queue:
        current_node = queue.pop(0)
        all_nodes.append(current_node)
        queue.extend(current_node.children)
    return all_nodes

def generate_successors_list(root):
    """Generate the 2D list of successors for all nodes."""
    all_nodes = flatten_tree(root)
    node_to_index = {node: i for i, node in enumerate(all_nodes)}  # Create a node to index mapping

    successors = [[] for _ in range(len(all_nodes))]  # Preparing the 2D list for all nodes

    # Populate the successor list for each node
    for node in all_nodes:
        node_index = node_to_index[node]
        for child in node.children:
            child_index = node_to_index[child]
            successors[node_index].append(child_index)

    return successors

def get_all_nodes(root):
    """Get all nodes in the tree starting from root using BFS."""
    all_nodes = []
    queue = [root]
    while queue:
        current_node = queue.pop(0)  # Change to queue operation
        all_nodes.append(current_node)
        queue.extend(current_node.children)  # Children are added to the end of the queue
    return all_nodes

def get_all_nodes_value(root):
    """Get all nodes in the tree starting from root using BFS."""
    all_nodes = []
    queue = [root]
    while queue:
        current_node = queue.pop(0)  # Change to queue operation
        all_nodes.append(current_node.value)
        queue.extend(current_node.children)  # Children are added to the end of the queue
    return all_nodes

def get_all_nodes_depth(root):
    """Get all nodes in the tree starting from root using BFS."""
    all_nodes = []
    queue = [root]
    while queue:
        current_node = queue.pop(0)  # Change to queue operation
        all_nodes.append(current_node.depth)
        queue.extend(current_node.children)  # Children are added to the end of the queue
    return all_nodes

def prune_tree(root, keep=127):
    """Prune the tree to keep up to 'keep' nodes based on unique cumulative logits, prioritizing uniqueness."""
    all_nodes = get_all_nodes(root)
    # Compute cumulative logits excluding the root
    all_nodes_with_logits = [(node, node.compute_cumulative_logit()) for node in all_nodes if node.parent is not None]

    # Sort nodes based on cumulative logits to prioritize higher values
    sorted_nodes = sorted(all_nodes_with_logits, key=lambda x: x[1], reverse=True)[:keep]
    keep_nodes = set(node[0] for node in sorted_nodes)
    # unique_values_and_parents = set()
    # keep_nodes = set()
    # for node, _ in sorted_nodes:
    #     value_parent_pair = (node.value, node.parent)
    #     if value_parent_pair not in unique_values_and_parents:
    #         unique_values_and_parents.add(value_parent_pair)
    #         keep_nodes.add(node)
    #     if len(keep_nodes) >= keep:
    #         break

    # Ensure the root is always kept
    keep_nodes.add(root)

    bfs_prune_tree(root, keep_nodes)
# def generate_mask_for_pruned_tree(root):
#     # Step 1: Collect all nodes using BFS, assuming the tree is already pruned to 128 nodes.
#     all_nodes = get_all_nodes(root)
#     all_nodes = all_nodes[1:]
#     node_to_index = {node: index for index, node in enumerate(all_nodes)}
#     # Step 2: Create a mapping of node to index.
#     # node_to_index = {node: index for index, node in enumerate(all_nodes)}
    
#     # Step 3: Initialize the mask with zeros.
#     mask = [[0 for _ in range(127)] for _ in range(127)]
    
#     # Step 4: Populate the mask.
#     for node in all_nodes:
#         current_index = node_to_index[node]
#         # Set the node's own position in the mask to 1.
#         mask[current_index][current_index] = 1
        
#         # Traverse up to the root to set ancestor relations.
#         current_node = node.parent
#         while current_node.value is not None:
#             parent_index = node_to_index[current_node]
#             mask[current_index][parent_index] = 1  # Mark the ancestor's relation to this node.
#             current_node = current_node.parent
    
#     # Convert mask to a tensor
#     mask_tensor = torch.tensor(mask, dtype=torch.int)
    
#     return mask_tensor
class GreedySTree(Tree):
    def __init__(self, 
                 #draft_model :LlamaForCausalLM_Attn, 
                 draft_model_engine :GraphInferenceEngine,
                 target_model_engine :GraphInferenceEngineTG,
                 prefix :torch.LongTensor,
                 temperature :float = 0.6,
                 top_p: float = 0.9,
                 draft_kv_len = 0,
                 target_kv_len = 0,
                 max_length = 256,
                 device :str = 'cpu',
                 max_target_seq = 256,
                 vocab_size = 32000,
                 grow_map = None,
                 attn_mask = None, 
                 sequence = None, 
                 new_tokens_buffer = None, 
                 parents_buffer = None, 
                 position_ids = None,
                 residual_graph = None,
                 sampling_callables = None,
                 sample_gather_indices = None) -> None:
        super().__init__(device=device, max_length=max_length)
        assert self.max_length == draft_model_engine.engine.max_length
        self.max_target_seq = max_target_seq
        #self.draft_model = draft_model.to(self.device).eval()
        # p 0.9 t 0.01
        self.draft_model_engine = draft_model_engine
        self.target_model_engine = target_model_engine
        self.temperature = temperature
        self.top_p = top_p
        self.residual_graph = residual_graph
        self.grow_map = grow_map
        self.sampling_callables = sampling_callables
        self.sample_gather_indices = sample_gather_indices
        self.draft_step = 7
        self.grow_map_roots_gpu = []
        for x in self.grow_map["roots"]:
             self.grow_map_roots_gpu.append(torch.Tensor(x).to(self.device).long())
        self.Successors = self.grow_map["Successors"]
        tree_mask :torch.Tensor = self.grow_map["mask"].to(self.device)
        tree_mask = (tree_mask == 0).type(self.dtype)
        
        tree_mask.masked_fill_(tree_mask > 0, torch.finfo(self.dtype).min)
        self.initialize(attn_mask, sequence, new_tokens_buffer, parents_buffer, position_ids, None)
        self.set_prefix(prefix=prefix)
        self.tree_size = self.grow_map["size"]
        self.tree_mask = tree_mask

        self.full_attn_mask[self.max_length - self.tree_size + 1: self.max_length, self.max_length - self.tree_size + 1: self.max_length] = tree_mask[1:, 1:]
        self.root = TreeNode()

        total_nodes = len(prefix) + self.tree_size - 1
        self.attn_mask = self.full_attn_mask[self.max_length - total_nodes: 2 * self.max_length - total_nodes, self.max_length - total_nodes: 2 * self.max_length - total_nodes]
        self.ground_truth_len = len(prefix)
        
        
        self.position_ids[len(prefix) : len(prefix) + self.tree_size - 1] = (self.grow_map["depth"][1:].to(self.device) + len(prefix) - 1)
        self.storage_ids = torch.arange(self.max_length).to(self.device)
        self.depth = self.grow_map["depth"][1:].to(self.device)
        
        self.draft_logits = torch.zeros((self.max_length, vocab_size), dtype=self.dtype).to(self.device)
        if draft_kv_len == 0:
            draft_model_outputs = self.draft_model_engine.inference(input_ids=self.tokens[:self.num_nodes].unsqueeze(0), 
                                storage_ids=self.storage_ids[:self.num_nodes], 
                                position_ids=self.position_ids[:self.num_nodes].unsqueeze(0),
                                attn_mask=self.attn_mask[:self.num_nodes][None, None, :, :])
            self.draft_logits[0] :torch.FloatTensor= draft_model_outputs[...,-1,:][0]
        
        else:
            draft_model_outputs = self.draft_model_engine.inference(input_ids = self.tokens[draft_kv_len: self.num_nodes].unsqueeze(0), 
                                                    storage_ids=self.storage_ids[draft_kv_len: self.num_nodes],
                                                    position_ids=self.position_ids[draft_kv_len: self.num_nodes].unsqueeze(0),
                                                    attn_mask=self.attn_mask[draft_kv_len: self.num_nodes][None, None, :, :])
            self.draft_logits[0] :torch.FloatTensor = draft_model_outputs[...,-1,:][0]
        self.draft_kv_len = self.num_nodes
        
        self.target_kv_len = target_kv_len
        self.seq_to_use = list(range(self.max_length))
    def generate_mask_for_pruned_tree(self,root):
        # Step 1: Collect all nodes using BFS, assuming the tree is already pruned to 128 nodes.
        all_nodes = get_all_nodes(root)
        all_nodes = all_nodes[1:]
        node_to_index = {node: index for index, node in enumerate(all_nodes)}
        # Step 2: Create a mapping of node to index.
        # node_to_index = {node: index for index, node in enumerate(all_nodes)}
        
        # Step 3: Initialize the mask with zeros.
        mask = [[0 for _ in range(127)] for _ in range(127)]
        
        # Step 4: Populate the mask.
        for node in all_nodes:
            current_index = node_to_index[node]
            # Set the node's own position in the mask to 1.
            mask[current_index][current_index] = 1
            
            # Traverse up to the root to set ancestor relations.
            current_node = node.parent
            while current_node.value is not None:
                parent_index = node_to_index[current_node]
                mask[current_index][parent_index] = 1  # Mark the ancestor's relation to this node.
                current_node = current_node.parent
        
        # Convert mask to a tensor
        mask_tensor = torch.tensor(mask, dtype=torch.int)
        
        return mask_tensor
    def update_tree_with_logits(self, logits, parent_nodes, grow_step):
        """Update the tree dynamically with new tokens and logits."""
        if grow_step !=0:
            new_tokens_values, new_tokens_set = logits.topk(k=31)
        else:
            new_tokens_values, new_tokens_set = logits.topk(k=127)
        i=0
        if grow_step !=0:
            parent_nodes = parent_nodes[1:]
        for parent_node in parent_nodes:
            new_token_value = new_tokens_values[i]
            new_token_set = new_tokens_set[i]
            if parent_node.updated == False:
                for value, logit in zip(new_token_set, new_token_value):
                    child = TreeNode(logit=logit.item(), value=value.item(), parent=parent_node, depth=parent_node.depth + 1, updated=False)
                    parent_node.add_child(child)
                parent_node.updated = True
            i+=1
        prune_tree(self.root, keep=127)  # Assuming root is defined and accessible
    
    @torch.inference_mode()
    def collective_grow_dynamic(self, benchmark=False, grow_step = None):
        
        if benchmark:
            x1 = 0.0
            x2 = 0.0
        
        if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
        if grow_step == 0:
           sampling_logits = self.draft_logits[0].unsqueeze(0)
        else:
           sampling_logits = self.draft_logits[1:128]
        nodes = [node for node in get_all_nodes(self.root)]
        self.update_tree_with_logits(sampling_logits, nodes, grow_step)
        self.tokens[self.num_nodes: self.num_nodes + 127] = torch.tensor(get_all_nodes_value(self.root)[1:])
        if benchmark:
                    torch.cuda.synchronize()
                    t2 = time.time()
                    x1 += (t2 - t1)
        # self.num_nodes = self.num_nodes + 128
        # nodes = [node for node in get_all_nodes(self.root)]
        
        position_ids = torch.tensor(get_all_nodes_depth(self.root)[1:])+self.num_nodes-1
        attn_mask = self.attn_mask[self.num_nodes: self.num_nodes+127]
        tree_mask = self.generate_mask_for_pruned_tree(self.root)
        tree_mask = (tree_mask == 0).type(self.dtype)
        tree_mask.masked_fill_(tree_mask > 0, torch.finfo(self.dtype).min)
        attn_mask[0:127 , self.num_nodes:self.num_nodes+127] = tree_mask
        attn_mask = attn_mask[None, None, :, :]

        draft_model_outputs = self.draft_model_engine.inference(
            input_ids = self.tokens[self.num_nodes: self.num_nodes+127].unsqueeze(0),
            position_ids = position_ids.unsqueeze(0),
            attn_mask = attn_mask,
            storage_ids=self.storage_ids[self.num_nodes: self.num_nodes+127]
            
        )
        self.draft_kv_len = self.num_nodes+127
        self.draft_logits[1:128] = draft_model_outputs[0][-127:]
        if benchmark:
                    torch.cuda.synchronize()
                    t3 = time.time()
                    x2 += (t3 - t2)
        if benchmark:
            return x1, x2
        return position_ids, tree_mask
    @torch.inference_mode()
    def accept_step(self, parent_id :int) ->ChildrenAccept:
        logits_id = parent_id - (self.ground_truth_len - 1)
        
        target_token = self.target_token[logits_id]
        children = self.Successors[logits_id]
        if len(children) == 0:
            return -1
        
        for pos in children:

            token = self.tokens[pos + (self.ground_truth_len - 1)]
            if token == target_token:
                return pos + (self.ground_truth_len - 1)
        
        return -1


        
    @torch.inference_mode()
    def verify(self, benchmark = False):
        new_node_num = (self.num_nodes - self.ground_truth_len + 1)
        if self.target_kv_len == 0:
            start_pos = 0
            end_pos = self.num_nodes
            attn_mask = self.attn_mask[start_pos: end_pos, :end_pos]
            attn_mask[end_pos-127:end_pos,end_pos-127:end_pos] = self.draft_tree_mask
            attn_mask = attn_mask[None, None, :, :]
            if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
            self.position_ids[end_pos-127:end_pos] = self.draft_postion_ids
            target_model_outputs = self.target_model_engine.inference(input_ids = self.tokens[start_pos : end_pos].unsqueeze(0), 
                                    position_ids = self.position_ids[start_pos : end_pos].unsqueeze(0), attn_mask = attn_mask, 
                                    storage_ids=self.storage_ids[start_pos : end_pos])
            if benchmark:
                torch.cuda.synchronize()
                t2 = time.time()
            self.target_logits :torch.FloatTensor= target_model_outputs[0][self.ground_truth_len - 1:]

        else:
            start_pos = self.target_kv_len
            end_pos = self.num_nodes
            attn_mask = self.attn_mask[start_pos: end_pos, :end_pos]
            attn_mask[1:,end_pos-127:end_pos] = self.draft_tree_mask
            attn_mask = attn_mask[None, None, :, :]
            if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
            self.position_ids[start_pos+1 : end_pos] = self.draft_postion_ids
            target_model_outputs = self.target_model_engine.inference(input_ids = self.tokens[start_pos : end_pos].unsqueeze(0), 
                                        position_ids =self.position_ids[start_pos : end_pos].unsqueeze(0), attn_mask = attn_mask,
                                        storage_ids=self.storage_ids[start_pos : end_pos])
            if benchmark:
                torch.cuda.synchronize()
                t2 = time.time()
            self.target_logits :torch.FloatTensor = target_model_outputs[0][-(new_node_num):]
        
        self.target_logits = get_sampling_logits(logits=self.target_logits, top_p=self.top_p, T=self.temperature, replicate=False)
        self.target_logits = softmax(self.target_logits / self.temperature, dim=-1)
        #self.target_token = self.target_logits.argmax(dim=-1)
        self.target_token = self.target_logits.multinomial(num_samples=1)
        accept_list = self.seq_to_use[:self.ground_truth_len]
        
        terminal = False
        while True:
            parent_id = accept_list[-1]
            pos = self.accept_step(parent_id=parent_id)
            if pos != -1:
                accept_list.append(pos)
                if self.tokens[pos] == 0 or self.tokens[pos] == 2:
                     terminal = True
                     break
            else:
                break
        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
        accept_length = len(accept_list)
        self.tokens[:accept_length] = self.tokens[accept_list]
        if not terminal:
            self.tokens[accept_length] = self.target_token[accept_list[-1] - self.ground_truth_len + 1].reshape(1)
            self.draft_model_engine.engine.kv_cache.gather_kv_incremental(accept_list[self.ground_truth_len:], self.ground_truth_len)
            self.target_model_engine.engine.kv_cache.gather_kv_incremental(accept_list[self.ground_truth_len:], self.ground_truth_len)
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                self.prepare_for_next_iter(accept_list, self.tokens[:accept_length+1])
                return self.tokens[:accept_length+1], accept_length, accept_length, t2 - t1, t3-t2, t4 - t3, terminal
            self.prepare_for_next_iter(accept_list, self.tokens[:accept_length+1])
            return self.tokens[:accept_length+1], accept_length, accept_length, terminal
            
        else:
             if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                return self.tokens[:accept_length], accept_length, accept_length, t2 - t1, t3-t2, t4 - t3, terminal
             return self.tokens[:accept_length], accept_length, accept_length, terminal
    def verbose(self):
        super().verbose()
    def construct_dynamic_tree(self, benchmark = False):
        if benchmark:
            sample_time = 0
            compute_time = 0
        for i in range(self.draft_step - 1):
                if benchmark:
                        _, t1, t2 = self.collective_grow_static(self.grow_map_roots_gpu[i], self.grow_map['branches'][i], benchmark=benchmark, grow_step=i)
                        sample_time += t1
                        compute_time += t2   
                else:
                        position_ids, tree_mask = self.collective_grow_dynamic(grow_step=i)
        self.draft_postion_ids = position_ids
        self.draft_tree_mask = tree_mask
        self.num_nodes = self.num_nodes+127
        self.Successors = generate_successors_list(self.root)
        if benchmark:
            return sample_time, compute_time
        else:
            return None
    def prepare_for_next_iter(self, accept_list: list[int], valid_tokens :torch.LongTensor):
        if len(accept_list) + 1 > self.max_target_seq:
              return 
        self.tokens[:len(valid_tokens)] = valid_tokens
        self.position_ids[:len(accept_list)] =  self.position_ids[accept_list]
        self.position_ids[len(accept_list)] = len(accept_list) 
        self.position_ids[len(valid_tokens) : len(valid_tokens) + self.tree_size - 1] = (self.depth + len(valid_tokens) - 1)
        self.ground_truth_len = len(valid_tokens)
        self.num_nodes = len(valid_tokens)

        total_nodes = len(valid_tokens) + self.tree_size - 1
        # self.attn_mask.fill_(torch.finfo(self.dtype).min)
        # self.attn_mask[:self.num_nodes, :self.num_nodes] = _make_causal_mask((1, self.num_nodes),dtype=self.dtype, device=self.device)
        # self.attn_mask[len(valid_tokens) : len(valid_tokens) + self.tree_size - 1, : len(valid_tokens)] = 0.0
        # self.attn_mask[len(valid_tokens) : len(valid_tokens) + self.tree_size - 1, len(valid_tokens) : len(valid_tokens) + self.tree_size - 1] = self.tree_mask[1:, 1:]

        self.attn_mask = self.full_attn_mask[self.max_length - total_nodes: 2 * self.max_length - total_nodes, self.max_length - total_nodes: 2 * self.max_length - total_nodes]

        
        draft_model_outputs = self.draft_model_engine.graph_inference(input_ids = self.tokens[len(accept_list): self.num_nodes].unsqueeze(0), 
                                                    storage_ids=self.storage_ids[len(accept_list): self.num_nodes],
                                                    position_ids=self.position_ids[len(accept_list): self.num_nodes].unsqueeze(0),
                                                    attn_mask=self.attn_mask[len(accept_list): self.num_nodes][None, None, :, :])
        
        self.draft_logits[0] :torch.FloatTensor = draft_model_outputs[...,-1,:][0]

        self.draft_kv_len = self.num_nodes
        self.target_kv_len = len(accept_list)
        self.root = TreeNode()
        


        


