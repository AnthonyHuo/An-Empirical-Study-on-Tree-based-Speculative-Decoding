import torch
from transformers import AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, DataCollatorForLanguageModeling, OPTForCausalLM, AutoTokenizer
from Llama import LlamaModel_Attn, LlamaForCausalLM_Attn
from torch.nn.functional import softmax
from copy import deepcopy
from tqdm import tqdm
from Tree import Tree
import time
from transformers import AutoProcessor
import deepspeed
from Engine import GraphInferenceEngine, GraphInferenceEngineTG
from torch.profiler import profile, record_function, ProfilerActivity
from utils import get_sampling_logits, make_tree_attention_mask, select_kv, ChildrenAccept, get_residual, cat_kv, _make_causal_mask
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
                 probs = None,
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
        self.draft_model_engine = draft_model_engine
        self.target_model_engine = target_model_engine
        self.temperature = temperature
        self.top_p = top_p
        self.residual_graph = residual_graph
        self.grow_map = grow_map
        self.sampling_callables = sampling_callables
        self.sample_gather_indices = sample_gather_indices
        self.draft_step = len(self.grow_map["roots"])
        self.grow_map_roots_gpu = []
        for x in self.grow_map["roots"]:
             self.grow_map_roots_gpu.append(torch.Tensor(x).to(self.device).long())
        self.Successors = self.grow_map["Successors"]
        tree_mask :torch.Tensor = self.grow_map["mask"].to(self.device)
        tree_mask = (tree_mask == 0).type(self.dtype)
        
        tree_mask.masked_fill_(tree_mask > 0, torch.finfo(self.dtype).min)
        self.initialize(attn_mask, sequence, new_tokens_buffer, parents_buffer, position_ids, None, None)
        self.set_prefix(prefix=prefix)
        self.tree_size = self.grow_map["size"]
        self.tree_mask = tree_mask
        self.full_attn_mask[self.max_length - self.tree_size + 1: self.max_length, self.max_length - self.tree_size + 1: self.max_length] = tree_mask[1:, 1:]


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

        # with open('sorted_followers_all1.txt', 'r') as file:
        #     self.indices = [int(line.strip()) for line in file.readlines()]
            #   self.probs = [float(line.strip()) for line in file.readlines()]
        # # print(self.indices)
        # # self.probs = torch.tensor(self.probs, device=self.device)
        # # self.probs =probs
        # self.indices_set = set(self.indices)
        # self.indices = torch.tensor(self.indices, device=self.device)
        with open('sorted_followers_top12.txt', 'r') as file:
            lines = file.readlines()

        # Convert the lines to a list of tuples of integers
        indices = [list(map(int, line.strip().split())) for line in lines]

        # # Convert the list to a torch tensor with shape (lines, 3)
        self.indices = torch.tensor(indices, device=self.device)
        # self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    @torch.inference_mode()
    def collective_grow_static(self, idx_list, n_branch_list :list[int], benchmark=False, grow_step = None,old_tokens= None):
        
        if benchmark:
            x1 = 0.0
            x2 = 0.0
        
        
        
        
        total_branch = sum(n_branch_list)
        max_branch = max(n_branch_list)

        if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
        sampling_logits = self.draft_logits[idx_list]
        # print(sampling_logits.shape)
        # self.probs = self.probs.unsqueeze(0)
        # sampling_logits += self.probs
        # sampling_logits += self.probs
        new_tokens_set = sampling_logits.topk(k=max_branch).indices.flatten()
        # if grow_step <= 1:
        #     new_tokens_set = sampling_logits.topk(k=max_branch).indices.flatten()
        # else:
        #     seq_tokens = self.indices[old_tokens]
        #     org_tokens_set = sampling_logits.topk(k=max_branch).indices
        #     new_tokens_set = torch.empty((sampling_logits.size(0), max_branch), dtype=org_tokens_set.dtype,device=org_tokens_set.device)

        #     for i in range(seq_tokens.shape[0]):
        #         if seq_tokens[i] != -1:
        #             # If seq_tokens in this dimension is not -1, sample max_branch - 1 tokens
        #             # First token is seq_tokens[i], rest are the top k-1 from new_tokens_set[i]
        #             new_tokens_set[i, 1] = seq_tokens[i]
        #             new_tokens_set[i, 0] = org_tokens_set[i, 0]
        #             new_tokens_set[i, 2:] = org_tokens_set[i, :max_branch-2]
        #         else:
        #             # If seq_tokens in this dimension is -1, sample all max_branch tokens
        #             new_tokens_set[i] = org_tokens_set[i]
            # for i in range(seq_tokens.shape[0]):
            #     for j in range(3):
            #         if seq_tokens[i, j] != -1:
            #             # If seq_tokens in this position is not -1, use it
            #             new_tokens_set[i, j] = seq_tokens[i, j]
            #             new_tokens_set[i, j:] = org_tokens_set[i, :max_branch-j]
            #         else:
            #             # If seq_tokens in this position is -1, use the corresponding value from org_tokens_set
            #             # new_tokens_set[i, j] = org_tokens_set[i, :, j]
            #             new_tokens_set[i, j:] = org_tokens_set[i, :max_branch-j]
            # new_tokens_set = new_tokens_set.flatten()
        # new_tokens_set = sampling_logits.topk(k=max_branch).indices.flatten()
        # print(sampling_logits)
        # print(new_tokens_set)
        # sampled_size = max_branch*3
        # sampled_tokens = sampling_logits.topk(k=sampled_size).indices
        # # # # # # print(len(sampled_tokens))
        # # # # # Step 2: Filter to get tokens that belong to self.indices
        # if sampling_logits.size(0)==1:
        #     sampled_tokens = sampled_tokens.flatten()
        #     valid_tokens = [token for token in sampled_tokens if token.item() in self.indices_set]
        #     if len(valid_tokens) < max_branch:
        #             remaining_tokens = [token for token in sampled_tokens if token.item() not in self.indices_set]
        #             valid_tokens += remaining_tokens[:max_branch - len(valid_tokens)]
        # else:
        #     valid_tokens = []
        #     for batch_idx in range(sampling_logits.size(0)):
        #          valid_token = [token for token in sampled_tokens[batch_idx] if token.item() in self.indices_set]
        #          if len(valid_token) < max_branch:
        #             remaining_tokens = [token for token in sampled_tokens[batch_idx] if token.item() not in self.indices_set]
        #             valid_token += remaining_tokens[:max_branch - len(valid_token)]

        #          valid_tokens.append(valid_token[:max_branch])
                
        # # # # # # # Keep only the top max_branch tokens
        # # # # # # # new_tokens_set = torch.tensor(valid_tokens[:max_branch],device=sampled_tokens.device)
        # new_tokens_set = torch.tensor(valid_tokens,device=sampled_tokens.device).flatten()
        # new_tokens_set = new_tokens_set[:max_branch*sampling_logits.shape[0]]
        # print(new_tokens_set_1 == new_tokens_set)
        # print(len(valid_tokens))
        # if len(valid_tokens) < max_branch:
        #     remaining_tokens = [token for token in sampled_tokens_cpu if token.item() not in self.indices_set]
        #     valid_tokens += remaining_tokens[:max_branch - len(valid_tokens)]
        # new_tokens_set = torch.tensor(valid_tokens[:max_branch],device=sampled_tokens.device)
        new_logits_set = sampling_logits.topk(k=max_branch).values.flatten()
        # new_tokens_set = self.sampling_callables[grow_step](self.draft_logits[idx_list])
        # finished_tokens = 0
            
        # for i in range(len(idx_list)):
        #         n_branch = n_branch_list[i]
        #         self.tokens[self.num_nodes + finished_tokens: self.num_nodes + finished_tokens + n_branch]  = new_tokens_set[i][:n_branch]
        #         finished_tokens += n_branch
        self.tokens[self.num_nodes: self.num_nodes + total_branch] = new_tokens_set[self.sample_gather_indices[grow_step]]
        self.logits[self.num_nodes: self.num_nodes + total_branch] = new_logits_set[self.sample_gather_indices[grow_step]]
        if benchmark:
                    torch.cuda.synchronize()
                    t2 = time.time()
                    x1 += (t2 - t1)
        self.num_nodes = self.num_nodes + total_branch
        
        start_pos = self.num_nodes - total_branch
        end_pos = self.num_nodes
        attn_mask = self.attn_mask[self.num_nodes - total_branch: self.num_nodes]
        attn_mask = attn_mask[None, None, :, :]
        
        draft_model_outputs = self.draft_model_engine.graph_inference(
            input_ids = self.tokens[self.draft_kv_len: self.num_nodes].unsqueeze(0),
            position_ids = self.position_ids[start_pos : end_pos].unsqueeze(0),
            attn_mask = attn_mask,
            storage_ids=self.storage_ids[self.draft_kv_len: self.num_nodes]
            
        )
        self.draft_kv_len = self.num_nodes
        self.draft_logits[start_pos - self.ground_truth_len + 1:end_pos - self.ground_truth_len + 1] = draft_model_outputs[0][-total_branch:]
        if benchmark:
                    torch.cuda.synchronize()
                    t3 = time.time()
                    x2 += (t3 - t2)
        if benchmark:
            return n_branch_list, x1, x2
        return n_branch_list,new_tokens_set[self.sample_gather_indices[grow_step]]

    @torch.inference_mode()
    def accept_step(self, parent_id :int, file) ->ChildrenAccept:
        logits_id = parent_id - (self.ground_truth_len - 1)
        
        target_token = self.target_token[logits_id]
        parent_token = self.tokens[parent_id].unsqueeze(0)
        # target = self.processor.batch_decode(target_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # parent = self.processor.batch_decode(parent_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print(parent)
        # print(target)
        draft_logits = self.draft_logits[logits_id]
        children = self.Successors[logits_id]
        if len(children) == 0:
            return -1
        exist = False
        for pos in children:
             
            token = self.tokens[pos + (self.ground_truth_len - 1)]
            draft_logits = self.logits[pos + (self.ground_truth_len - 1)]
            draft_logits =  torch.exp(draft_logits)
            if token == target_token:
                file.write(f"1 {draft_logits.tolist()}\n")
                # file.write(f"1 {token}\n")
                # file.write(f"{token}\n")
                exist = True
                return pos + (self.ground_truth_len - 1)
            # else:
            #     for token in self.indices[parent_token].squeeze(0):
            #         if token == target_token:
            #             self.tokens[pos+ (self.ground_truth_len - 1)] = token
            #             return pos + (self.ground_truth_len - 1)

        # if exist == False: 
        #     for token in self.indices[parent_token].squeeze(0):
        #         if token == target_token:
        #             self.tokens[children[0]+ (self.ground_truth_len - 1)] = token
        #             return children[0] + (self.ground_truth_len - 1)

            # file.write(f"0 {token}\n")
            file.write(f"0 {draft_logits.tolist()}\n")
        
        return -1


        
    @torch.inference_mode()
    def verify(self, file, benchmark = False):
        new_node_num = (self.num_nodes - self.ground_truth_len + 1)
        if self.target_kv_len == 0:
            start_pos = 0
            end_pos = self.num_nodes
            attn_mask = self.attn_mask[start_pos: end_pos, :end_pos]
            attn_mask = attn_mask[None, None, :, :]
            if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
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
            attn_mask = attn_mask[None, None, :, :]
            if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
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
        more = False
        while True:
            parent_id = accept_list[-1]
            pos = self.accept_step(parent_id=parent_id,file=file)
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
        # self.logits[:accept_length] = self.logits[accept_list]
        if not terminal:
            self.tokens[accept_length] = self.target_token[accept_list[-1] - self.ground_truth_len + 1].reshape(1)
            # self.logits[accept_length] = 
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
    def construct_grow_map(self, benchmark = False):
        if benchmark:
            sample_time = 0
            compute_time = 0
        old_tokens = None
        for i in range(self.draft_step - 1):
                if benchmark:
                        _, t1, t2 = self.collective_grow_static(self.grow_map_roots_gpu[i], self.grow_map['branches'][i], benchmark=benchmark, grow_step=i)
                        sample_time += t1
                        compute_time += t2   
                else:
                        n_branch_list,tokens = self.collective_grow_static(self.grow_map_roots_gpu[i], self.grow_map['branches'][i], grow_step=i,old_tokens= old_tokens)
                old_tokens = tokens
        if benchmark:
            return sample_time, compute_time
        else:
            return None
    def prepare_for_next_iter(self, accept_list: list[int], valid_tokens :torch.LongTensor):
        if len(accept_list) + 1 > self.max_target_seq:
              return 
       
        # self.tokens[:len(valid_tokens)] = valid_tokens
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


        


        


class GreedySTreeTest(Tree):
    def __init__(self, 
                 draft_model_engine :GraphInferenceEngine,
                 target_model_engine :GraphInferenceEngineTG,
                 prefix :torch.LongTensor,
                 temperature :float = 0.6,
                 top_p: float = 0.9,
                 draft_kv_len = 0,
                 target_kv_len = 0,
                 max_length = 256,
                 max_width = 32,
                 device :str = 'cpu',
                 attn_mask = None, 
                 sequence = None, 
                 new_tokens_buffer = None, 
                 parents_buffer = None, 
                 position_ids = None) -> None:
        
        super().__init__(device=device, max_length=max_length)
        assert self.max_length == draft_model_engine.engine.max_length
        self.max_width = max_width
        self.draft_model_engine = draft_model_engine
        self.target_model_engine = target_model_engine
        self.temperature = temperature
        self.top_p = top_p
        
        self.initialize(attn_mask, sequence, new_tokens_buffer, parents_buffer, position_ids, None)
        self.set_prefix(prefix=prefix)
        self.Successors = [list(range(1, self.max_width + 1))]
        self.Successors.extend([[] for _ in range(self.max_width)])

        self.attn_mask = self.full_attn_mask[:self.max_length, :self.max_length]
        for idx in range(self.max_width):
             self.attn_mask[idx + self.num_nodes] = self.attn_mask[self.num_nodes - 1]
             self.attn_mask[idx + self.num_nodes][idx + self.num_nodes] = 0.0
        
        self.position_ids[self.num_nodes : self.num_nodes + self.max_width] = self.position_ids[self.num_nodes - 1] + 1
        self.ground_truth_len = len(prefix)
        self.r = torch.rand(len(position_ids)).to(self.device)
        self.storage_ids = torch.arange(self.max_length).to(self.device)
        
        
        if draft_kv_len == 0:
            draft_model_outputs = self.draft_model_engine.inference(input_ids=self.tokens[:self.num_nodes].unsqueeze(0), 
                                storage_ids=self.storage_ids[:self.num_nodes], 
                                position_ids=self.position_ids[:self.num_nodes].unsqueeze(0),
                                attn_mask=self.attn_mask[:self.num_nodes][None, None, :, :])
            self.draft_logits :torch.FloatTensor= draft_model_outputs[...,-1,:]
        
        else:
            draft_model_outputs = self.draft_model_engine.inference(input_ids = self.tokens[draft_kv_len: self.num_nodes].unsqueeze(0), 
                                                    storage_ids=self.storage_ids[draft_kv_len: self.num_nodes],
                                                    position_ids=self.position_ids[draft_kv_len: self.num_nodes].unsqueeze(0),
                                                    attn_mask=self.attn_mask[draft_kv_len: self.num_nodes][None, None, :, :])
            self.draft_logits :torch.FloatTensor = draft_model_outputs[...,-1,:]
        self.draft_kv_len = self.num_nodes
        
        self.target_kv_len = target_kv_len
        self.rand = torch.empty((self.max_width + 1, self.draft_logits.shape[1])).uniform_().to(self.device)
        self.collective_grow_static([0], [self.max_width])
    
    @torch.inference_mode()
    def collective_grow_static(self, idx_list :torch.LongTensor, n_branch_list :list[int], benchmark=False):
        
        
        assert len(set(idx_list)) == len(idx_list)
        assert len(self.draft_logits) == (self.num_nodes - self.ground_truth_len + 1)
        
        total_branch = sum(n_branch_list)
        max_branch = max(n_branch_list)
        sampling_logits = self.draft_logits[idx_list]
        
        sampling_q = softmax(sampling_logits / self.temperature, dim=-1)
        
            
            
        new_tokens_set  = (self.rand[idx_list].log()/sampling_q).topk(k=max_branch).indices
        
            
        
        finished_tokens = 0
            
        for i, idx in enumerate(idx_list):
                n_branch = n_branch_list[i]
                self.tokens[self.num_nodes + finished_tokens: self.num_nodes + finished_tokens + n_branch]  = new_tokens_set[i][:n_branch]
                finished_tokens += n_branch
            
        
        self.num_nodes = self.num_nodes + total_branch
        

        
        start_pos = self.num_nodes - total_branch
        end_pos = self.num_nodes
        attn_mask = self.attn_mask[self.num_nodes - total_branch: self.num_nodes]
        attn_mask = attn_mask[None, None, :, :]
        
        draft_model_outputs = self.draft_model_engine.graph_inference(
            input_ids = self.tokens[self.draft_kv_len: self.num_nodes].unsqueeze(0),
            position_ids = self.position_ids[start_pos : end_pos].unsqueeze(0),
            attn_mask = attn_mask,
            storage_ids=self.storage_ids[self.draft_kv_len: self.num_nodes]
            
        )
        self.draft_kv_len = self.num_nodes
        self.draft_logits = torch.cat([self.draft_logits, draft_model_outputs[0][-total_branch:]], dim=0)
        assert len(self.draft_logits) == (self.num_nodes - self.ground_truth_len + 1)
        
        return n_branch_list
    @torch.inference_mode()
    def accept_step(self, parent_id :int) ->ChildrenAccept:
        logits_id = parent_id - (self.ground_truth_len - 1)
        p = self.target_logits[logits_id]
        
        draft_logits = self.draft_logits[logits_id]
        children = self.Successors[logits_id]
        if len(children) == 0:
            return ChildrenAccept(accept_mark=2, residual=p)
        
        for idx, pos in enumerate(children):

            token = self.tokens[pos + (self.ground_truth_len - 1)]
            q = softmax(draft_logits / self.temperature, dim=-1)
            r = self.r[pos + (self.ground_truth_len - 1)]
            if p[token] >= r * q[token]:
                return ChildrenAccept(accept_mark=0, token=token, position=pos + (self.ground_truth_len - 1), successor_order=idx)
            else:
                p = get_residual(p, q)
                draft_logits[token] = -torch.inf
        
        return ChildrenAccept(accept_mark=1, residual=p)


        
    @torch.inference_mode()
    def verify(self, benchmark = False):
        new_node_num = (self.num_nodes - self.ground_truth_len + 1)
        if self.target_kv_len == 0:
            start_pos = 0
            end_pos = self.num_nodes
            attn_mask = self.attn_mask[start_pos: end_pos, :end_pos]
            attn_mask = attn_mask[None, None, :, :].type(self.target_model_engine.dtype)
            target_model_outputs = self.target_model_engine.inference(input_ids = self.tokens[start_pos : end_pos].unsqueeze(0), 
                                    position_ids = self.position_ids[start_pos : end_pos].unsqueeze(0), attn_mask = attn_mask, 
                                    storage_ids=self.storage_ids[start_pos : end_pos])
            self.target_logits :torch.FloatTensor= target_model_outputs[0][self.ground_truth_len - 1:]
            
        else:
            start_pos = self.target_kv_len
            end_pos = self.num_nodes
            attn_mask = self.attn_mask[start_pos: end_pos, :end_pos]
            attn_mask = attn_mask[None, None, :, :].type(self.target_model_engine.dtype)
            target_model_outputs = self.target_model_engine.inference(input_ids = self.tokens[start_pos : end_pos].unsqueeze(0), 
                                        position_ids =self.position_ids[start_pos : end_pos].unsqueeze(0), attn_mask = attn_mask,
                                        storage_ids=self.storage_ids[start_pos : end_pos])
            
            self.target_logits :torch.FloatTensor = target_model_outputs[0][-(new_node_num):]
        
        assert len(self.draft_logits) == (self.num_nodes - self.ground_truth_len + 1)
        assert len(self.target_logits) == (self.num_nodes - self.ground_truth_len + 1)
        self.target_logits = get_sampling_logits(logits=self.target_logits, top_p=self.top_p, T=self.temperature, replicate=False)
        self.target_logits = softmax(self.target_logits / self.temperature, dim=-1)
        accept_list = list(range(self.ground_truth_len))
        b = -1
        terminal = False
        while True:
            parent_id = accept_list[-1]
            children_accept = self.accept_step(parent_id=parent_id)
            if children_accept.accept_mark == 0:
                accept_list.append(children_accept.position)
                b = children_accept.successor_order
                if self.tokens[children_accept.position] == 2 or self.tokens[children_accept.position] == 0:
                     terminal = True
                     break
            else:
                residual = children_accept.residual
                break
        if not terminal:
            if torch.isnan(residual).any():
                 terminal = True
            else:
                last_token = residual.multinomial(num_samples=1, replacement=True)

        
        accept_tokens = self.tokens[accept_list]
        if not terminal:
            valid_tokens = torch.cat([accept_tokens, last_token], dim=-1)
            
            self.draft_model_engine.gather_kv(accept_list)
            self.target_model_engine.gather_kv(accept_list)

            return valid_tokens, len(accept_list), len(accept_list), b, terminal
        else:
            return accept_tokens, len(accept_list), len(accept_list), b, terminal
    
    def verbose(self):
        super().verbose()

    
    

                