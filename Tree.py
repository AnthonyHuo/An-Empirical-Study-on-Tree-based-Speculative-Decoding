import torch
from transformers import AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, DataCollatorForLanguageModeling, OPTForCausalLM, AutoTokenizer
from torch.nn.functional import softmax
from copy import deepcopy
from tqdm import tqdm
import time
from utils import _make_causal_mask
class Tree:
    def __init__(self, device :str = 'cpu', max_length = 512, dtype = torch.float16) -> None:
        self.tokens :torch.LongTensor = None
        self.Successors :list[list[int]] = []
        self.num_nodes = 0
        self.device = device
        self.max_length = max_length
        self.dtype = dtype


    def initialize(self, attn_mask, sequence, new_tokens_buffer, parents_buffer, position_ids, active_mark):
        self.attn_mask = attn_mask
        self.sequence = sequence
        self.new_tokens_buffer = new_tokens_buffer
        self.parents_buffer = parents_buffer
        self.position_ids = position_ids
        self.active_mark = active_mark

        assert self.attn_mask.shape[1] == self.max_length

    def set_prefix(self, prefix: torch.LongTensor):
        self.tokens = prefix.to(self.device)
        self.position_ids[:len(prefix)] = torch.arange(len(prefix))
        
        self.num_nodes = len(prefix)
        self.attn_mask[:self.num_nodes, :self.num_nodes] = _make_causal_mask((1, self.num_nodes),dtype=self.dtype, device=self.device)

        
        

    def collective_expand_position(self, parents :torch.LongTensor, expand_tokens :torch.LongTensor):
        self.tokens = torch.cat([self.tokens, expand_tokens], dim=-1)
        

    def verbose(self):
        print(self.tokens)
        print(self.Successors)






        



