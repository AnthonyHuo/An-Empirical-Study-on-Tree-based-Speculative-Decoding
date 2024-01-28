import torch
import dataclasses
import math
from copy import deepcopy
from torch.nn.functional import relu
def get_residual(p: torch.Tensor, q:torch.Tensor):
    residual = p - q
    residual.relu_()
    residual = residual / (residual.sum(dim=-1).unsqueeze(-1) + 1e-9)
    return residual

def expand_kv(kv_cache, k):
    kv_shape = kv_cache[0][0].shape
    new_kv_cache = ()
    for kv in kv_cache:
        new_kv_cache = new_kv_cache + ([kv[0].expand(k, kv_shape[1], kv_shape[2], kv_shape[3]), 
                kv[1].expand(k, kv_shape[1], kv_shape[2], kv_shape[3])],)
    return new_kv_cache

def cat_kv(old_kv, delta_kv, cut_len :int):
    new_kv_cache = ()
    for i in range(len(old_kv)):
          k = torch.cat([old_kv[i][0], delta_kv[i][0][..., -cut_len:, :]], dim=-2)
          v = torch.cat([old_kv[i][1], delta_kv[i][1][..., -cut_len:, :]], dim=-2)
          new_kv_cache += ([k,v],)
    return new_kv_cache
    
    
def make_tree_attention_mask(
        prefix_len :int,
        gen_len :int,
        ancestors :list[list[int]],
        device ="cpu",
        dtype = torch.float32
    ) -> torch.FloatTensor:
    tree_mask = torch.full((gen_len, gen_len + prefix_len), torch.finfo(dtype).min, dtype=dtype).to(device=device)
    for idx, ancestor in enumerate(ancestors):
        if len(ancestor) > 0:
            tree_mask[idx][ancestor] = 0.0
    return tree_mask[None, None, :, :]

def get_sampling_logits(logits :torch.Tensor, top_p:float, T: float, replicate = False):
    if replicate:
        logits = logits.clone()
    if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
                filter = cumulative_probs > top_p
                filter[..., 1:] = filter[..., :-1].clone()
                filter[..., 0] = 0
                indices_to_remove = filter.scatter(-1, sorted_indices, filter)
                logits[indices_to_remove] = float('-inf')
    return logits

def select_kv(kv_cache: tuple[list[torch.FloatTensor]], indices: list[int]):
        new_kv_cache = ()
        for k,v in kv_cache:
             k = k[..., indices, :]
             v = v[..., indices, :]
             new_kv_cache += ([k,v],)
        return new_kv_cache

@dataclasses.dataclass
class ChildrenAccept:
    accept_mark :int = None
    token :int = None
    position :int = None
    successor_order :int = -1
    residual :torch.FloatTensor = None

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask
