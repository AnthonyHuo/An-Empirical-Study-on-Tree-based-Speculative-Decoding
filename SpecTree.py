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
from Engine import GraphInferenceEngine
from torch.profiler import profile, record_function, ProfilerActivity
from utils import get_sampling_logits, make_tree_attention_mask, select_kv, ChildrenAccept, get_residual, cat_kv
class SpecTree(Tree):
    def __init__(self, 
                 #draft_model :LlamaForCausalLM_Attn, 
                 draft_model_engine :GraphInferenceEngine,
                 target_model :LlamaForCausalLM_Attn,
                 prefix :torch.LongTensor,
                 temperature :float = 0.6,
                 top_p: float = 0.9,
                 target_kv = None,
                 draft_kv_len = 0,
                 target_kv_len = 0,
                 max_length = 256,
                 device :str = 'cpu',
                 grow_map = None,
                 attn_mask = None, 
                 sequence = None, 
                 new_tokens_buffer = None, 
                 parents_buffer = None, 
                 position_ids = None, 
                 active_mark = None) -> None:
        super().__init__(device=device, max_length=max_length)
        assert self.max_length == draft_model_engine.engine.max_length
        #self.draft_model = draft_model.to(self.device).eval()
        self.draft_model_engine = draft_model_engine
        self.target_model = target_model.eval()
        self.temperature = temperature
        self.top_p = top_p
        self.grow_map = grow_map
        self.draft_step = len(self.grow_map["roots"])
        self.Successors = self.grow_map["Successors"]
        tree_mask :torch.Tensor = self.grow_map["mask"].to(self.device)
        tree_mask = (tree_mask == 0).type(self.dtype)
        
        tree_mask.masked_fill_(tree_mask > 0, torch.finfo(self.dtype).min)
        self.initialize(attn_mask, sequence, new_tokens_buffer, parents_buffer, position_ids, active_mark)
        self.set_prefix(prefix=prefix)
        self.tree_size = self.grow_map["size"]
        self.attn_mask[len(prefix) : len(prefix) + self.tree_size, : len(prefix)] = 0.0
        self.attn_mask[len(prefix) : len(prefix) + self.tree_size - 1, len(prefix) : len(prefix) + self.tree_size - 1] = tree_mask[1:, 1:]
        self.ground_truth_len = len(prefix)
        self.r = torch.rand(len(position_ids)).to(self.device)
        
        self.position_ids[len(prefix) : len(prefix) + self.tree_size - 1] = (self.grow_map["depth"][1:].to(self.device) + len(prefix) - 1)
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
        self.target_kv = target_kv
        if self.target_kv is not None:
            assert self.target_kv[0][0].shape[-2] == self.target_kv_len
            assert self.target_kv_len == (self.num_nodes - 1)
        else:
            assert self.target_kv_len == 0
        self.rand = torch.empty((self.tree_size, self.draft_logits.shape[1])).uniform_().to(self.device)
    
    @torch.inference_mode()
    def collective_grow_static(self, idx_list :list[int], n_branch_list :list[int], benchmark=False):
        
        if benchmark:
            x1 = 0.0
            x2 = 0.0
        
        
        assert len(set(idx_list)) == len(idx_list)
        assert len(self.draft_logits) == (self.num_nodes - self.ground_truth_len + 1)
        
        if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
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
        self.draft_logits = torch.cat([self.draft_logits, draft_model_outputs[0][-total_branch:]], dim=0)
        assert len(self.draft_logits) == (self.num_nodes - self.ground_truth_len + 1)
        if benchmark:
                    torch.cuda.synchronize()
                    t3 = time.time()
                    x2 += (t3 - t2)
        if benchmark:
            return n_branch_list, x1, x2
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
            attn_mask = attn_mask[None, None, :, :].type(self.target_model.dtype)
            if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
            target_model_outputs = self.target_model(input_ids = self.tokens[start_pos : end_pos].unsqueeze(0), 
                                    use_cache=True, position_ids = self.position_ids[start_pos : end_pos].unsqueeze(0), attention_mask = attn_mask)
            if benchmark:
                torch.cuda.synchronize()
                t2 = time.time()
            self.target_kv = target_model_outputs.past_key_values
            self.target_logits :torch.FloatTensor= target_model_outputs.logits[0][self.ground_truth_len - 1:]
            
        else:
            assert self.target_kv[0][0].shape[-2] == self.target_kv_len
            start_pos = self.target_kv_len
            end_pos = self.num_nodes
            attn_mask = self.attn_mask[start_pos: end_pos, :end_pos]
            attn_mask = attn_mask[None, None, :, :].type(self.target_model.dtype)
            if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
            target_model_outputs = self.target_model(input_ids = self.tokens[self.target_kv_len: self.num_nodes].unsqueeze(0), 
                                        use_cache=True,past_key_values=self.target_kv, 
                                        position_ids =self.position_ids[start_pos : end_pos].unsqueeze(0), attention_mask = attn_mask)
            
            if benchmark:
                torch.cuda.synchronize()
                t2 = time.time()
            self.target_kv = target_model_outputs.past_key_values
            self.target_logits :torch.FloatTensor = target_model_outputs.logits[0][-(new_node_num):]
        self.target_kv_len = len(self.tokens)
        assert len(self.draft_logits) == (self.num_nodes - self.ground_truth_len + 1)
        assert len(self.target_logits) == (self.num_nodes - self.ground_truth_len + 1)
        self.target_logits = get_sampling_logits(logits=self.target_logits, top_p=self.top_p, T=self.temperature, replicate=False)
        self.target_logits = softmax(self.target_logits / self.temperature, dim=-1)
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
                residual = children_accept.residual
                break
        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
        last_token = residual.multinomial(num_samples=1)

        accept_tokens = self.tokens[accept_list]
        valid_tokens = torch.cat([accept_tokens, last_token], dim=-1)
        self.draft_model_engine.gather_kv(accept_list)
        
        target_kv = select_kv(self.target_kv, accept_list)
        if benchmark:
            torch.cuda.synchronize()
            t4 = time.time()
            return valid_tokens, len(accept_list), target_kv, t2 - t1, t3-t2, t4-t3
        return valid_tokens, len(accept_list), target_kv
    def verbose(self):
        super().verbose()
    

    def construct_grow_map(self, benchmark = False):
        if benchmark:
            sample_time = 0
            compute_time = 0
        for i in range(self.draft_step - 1):
                if benchmark:
                        _, t1, t2 = self.collective_grow_static(self.grow_map['roots'][i], self.grow_map['branches'][i], benchmark=benchmark)
                        sample_time += t1
                        compute_time += t2   
                else:
                        self.collective_grow_static(self.grow_map['roots'][i], self.grow_map['branches'][i])
        if benchmark:
            return sample_time, compute_time
        else:
            return None
    
    

                