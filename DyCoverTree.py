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
class DyCoverTree():
    def __init__(self,  
                 draft_model_engine :GraphInferenceEngine,
                 target_model_engine :GraphInferenceEngineTG,
                 prefix :torch.LongTensor,
                 temperature :float = 0.6,
                 top_p: float = 0.9,
                 draft_kv_len = 0,
                 target_kv_len = 0,
                 max_length = 256,
                 pool_size = 32,
                 span = 8,
                 device :str = 'cpu',
                 attn_mask = torch.Tensor, 
                 position_ids = torch.Tensor,
                 dtype = torch.float16,
                 eps = 1e-9,
                 alpha = 0.9999,
                 iterations = 8) -> None:
        


        
        self.draft_model_engine = draft_model_engine
        self.target_model_engine = target_model_engine
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self.max_length = max_length
        self.dtype = dtype
        self.tokens = torch.zeros(max_length, device=device).long()
        self.parents = torch.zeros(max_length, device=device).long()
        self.initialize(attn_mask, position_ids)
        self.set_prefix(prefix=prefix)
        self.ground_truth_len = len(prefix)
        self.storage_ids = torch.arange(self.max_length).to(self.device)

        assert self.max_length == draft_model_engine.engine.max_length
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
        self.target_token = None
        self.pool_size = pool_size
        self.span = span
        self.eps = eps
        self.alpha = alpha
        self.iterations = iterations
        self.scores = torch.zeros(pool_size + 1, device=device)
        self.draft_logits = softmax(self.draft_logits / self.temperature, dim=-1) * self.alpha
        self.draft_logits = torch.log(self.draft_logits + self.eps)
    @torch.inference_mode()
    def initial_grow(self):
        assert len(self.draft_logits) == 1
        topk = self.draft_logits[0].topk(k=self.span)
        self.tokens[self.num_nodes: self.num_nodes + self.span] = topk.indices
        self.scores[1:1+self.span] = topk.values
        self.parents[self.num_nodes: self.num_nodes + self.span] = self.ground_truth_len - 1
        self.attn_mask[self.num_nodes: self.num_nodes + self.span] = self.attn_mask[self.parents[self.num_nodes: self.num_nodes + self.span]]
        self.attn_mask[self.num_nodes: self.num_nodes + self.span].scatter_(dim=1, index=self.storage_ids[self.num_nodes: self.num_nodes + self.span].unsqueeze(-1), value=0.0)
        self.position_ids[self.num_nodes: self.num_nodes + self.span] = self.position_ids[self.parents[self.num_nodes: self.num_nodes + self.span]] + 1

        
        self.draft_logits = self.draft_model_engine.graph_inference(
            input_ids = self.tokens[self.num_nodes-1: self.num_nodes + self.span].unsqueeze(0),
            storage_ids=self.storage_ids[self.num_nodes-1: self.num_nodes + self.span],
            position_ids = self.position_ids[self.num_nodes-1: self.num_nodes + self.span].unsqueeze(0),
            attn_mask=self.attn_mask[self.num_nodes-1: self.num_nodes + self.span][None, None, :, :]
        )[0]
        self.draft_logits = softmax(self.draft_logits / self.temperature, dim=-1) * self.alpha
        self.draft_logits = torch.log(self.draft_logits + self.eps)
        assert len(self.draft_logits) == (self.span + 1)
        self.num_nodes = self.num_nodes + self.span
    
    @torch.inference_mode()
    def boost_iteration(self):
        
        span_nodes = self.draft_logits.topk(k=self.span)
        span_nodes_tokens = span_nodes.indices.view(-1)

        
        span_nodes_scores = (span_nodes.values + self.scores[:len(self.draft_logits)].unsqueeze(-1)).view(-1)

        
        topk = span_nodes_scores.topk(k=self.pool_size)

        tokens = span_nodes_tokens[topk.indices]
        parents = (topk.indices // self.span)  + self.ground_truth_len - 1
        scores = span_nodes_scores[topk.indices]

        
        self.num_nodes = self.ground_truth_len + self.pool_size
        self.tokens[self.ground_truth_len : self.num_nodes] = tokens
        self.scores[1:] = scores
        self.parents[self.ground_truth_len : self.num_nodes] = parents

        self.attn_mask[self.ground_truth_len : self.num_nodes] = self.attn_mask[self.parents[self.ground_truth_len : self.num_nodes]]
        self.attn_mask[self.ground_truth_len : self.num_nodes].scatter_(dim=1, index=self.storage_ids[self.ground_truth_len : self.num_nodes].unsqueeze(-1), value=0.0)
        self.position_ids[self.ground_truth_len : self.num_nodes] = self.position_ids[self.parents[self.ground_truth_len : self.num_nodes]] + 1

        self.draft_logits = self.draft_model_engine.graph_inference(
            input_ids = self.tokens[self.ground_truth_len - 1: self.num_nodes].unsqueeze(0),
            storage_ids=self.storage_ids[self.ground_truth_len - 1: self.num_nodes],
            position_ids = self.position_ids[self.ground_truth_len - 1: self.num_nodes].unsqueeze(0),
            attn_mask=self.attn_mask[self.ground_truth_len - 1: self.num_nodes][None, None, :, :]
        )[0]

        self.draft_logits = softmax(self.draft_logits / self.temperature, dim=-1) * self.alpha
        self.draft_logits = torch.log(self.draft_logits + self.eps)
        assert len(self.draft_logits) == (self.pool_size + 1)

    @torch.inference_mode()
    def grow(self):
            self.initial_grow()
            for _ in range(self.iterations):
                self.boost_iteration()
    @torch.inference_mode()
    def accept_step(self, parent_id :int) ->ChildrenAccept:
        logits_id = parent_id - (self.ground_truth_len - 1)
        
        target_token = self.target_token[logits_id]
        children = torch.nonzero(self.parents == parent_id)
        if len(children) == 0:
            return ChildrenAccept(accept_mark=2)
        
        for idx, pos in enumerate(children):

            token = self.tokens[pos]
            if token == target_token:
                return ChildrenAccept(accept_mark=0, token=token, position=pos.item(), successor_order=idx)
        
        return ChildrenAccept(accept_mark=1)


        
    @torch.inference_mode()
    def verify(self, benchmark = False):
        new_node_num = (self.num_nodes - self.ground_truth_len + 1)
        if self.target_kv_len == 0:
            start_pos = 0
            end_pos = self.num_nodes
            attn_mask = self.attn_mask[start_pos: end_pos, :end_pos]
            attn_mask = attn_mask[None, None, :, :].type(self.target_model_engine.dtype)
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
            attn_mask = attn_mask[None, None, :, :].type(self.target_model_engine.dtype)
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
        
        assert len(self.draft_logits) == (self.num_nodes - self.ground_truth_len + 1)
        assert len(self.target_logits) == (self.num_nodes - self.ground_truth_len + 1)
        self.target_logits = get_sampling_logits(logits=self.target_logits, top_p=self.top_p, T=self.temperature, replicate=False)
        self.target_logits = softmax(self.target_logits / self.temperature, dim=-1)
        
        self.target_token = self.target_logits.multinomial(num_samples=1, replacement=True)
        
        accept_list = list(range(self.ground_truth_len))
        if benchmark:
                torch.cuda.synchronize()
                t2 = time.time()
        while True:
            parent_id = accept_list[-1]
            children_accept = self.accept_step(parent_id=parent_id)
            if children_accept.accept_mark == 0:
                accept_list.append(children_accept.position)
            else:
                break
        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
        last_token = self.target_token[accept_list[-1] - self.ground_truth_len + 1]

        accept_tokens = self.tokens[accept_list]

        
        valid_tokens = torch.cat([accept_tokens, last_token], dim=-1)
        
        self.draft_model_engine.gather_kv(accept_list)
        self.target_model_engine.gather_kv(accept_list)
        if benchmark:
            torch.cuda.synchronize()
            t4 = time.time()
            return valid_tokens, len(accept_list), len(accept_list), t2 - t1, t3-t2, t4-t3
        return valid_tokens, len(accept_list), len(accept_list)
    
    def initialize(self, attn_mask, position_ids):
        self.attn_mask :torch.Tensor = attn_mask
        self.position_ids :torch.Tensor = position_ids
        assert self.attn_mask.shape[1] == self.max_length

    def set_prefix(self, prefix: torch.LongTensor):
        self.tokens[:len(prefix)] = prefix.to(self.device)
        self.position_ids[:len(prefix)] = torch.arange(len(prefix))
        
        self.num_nodes = len(prefix)
        self.attn_mask[:self.num_nodes, :self.num_nodes] = _make_causal_mask((1, self.num_nodes),dtype=self.dtype, device=self.device)


    
    

                