import torch
from transformers import LlamaConfig

class KV_Cache:

    def __init__(self, 
        config :LlamaConfig,
        batch_size :int = 1,
        max_length :int = 256, 
        device :str = 'cuda:0',
        dtype = torch.float16) -> None:
        self.config = config
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.k_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_attention_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_attention_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

    def initialize_kv(self,
            k_cache :torch.Tensor,
            v_cache :torch.Tensor,
            kv_len :int):
        
        self.k_cache[...,:kv_len,:] = k_cache[...,:kv_len,:]
        self.v_cache[...,:kv_len,:] = v_cache[...,:kv_len,:]
        
        
    
    def gather_kv(self, indices: list[int]):

        self.k_cache[..., :len(indices), :] = self.k_cache[..., indices, :]
        self.v_cache[..., :len(indices), :] = self.v_cache[..., indices, :]

        self.k_cache[..., len(indices):, :] = 0.0
        self.v_cache[..., len(indices):, :] = 0.0


    
    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            storage_ids :torch.LongTensor,
            debug :bool = False):
        
        input_length = len(storage_ids)
        if debug:
            assert input_length == new_k_cache.shape[-2]
            assert input_length == new_v_cache.shape[-2]
        
        self.k_cache[layer_idx].index_copy_(dim=-2, index=storage_ids, source=new_k_cache)
        self.v_cache[layer_idx].index_copy_(dim=-2, index=storage_ids, source=new_v_cache)

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        