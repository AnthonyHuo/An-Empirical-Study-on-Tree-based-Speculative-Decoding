import torch
from Engine import InferenceEngine
from Llama_utils import _make_causal_mask
DTYPE = torch.float32
DEVICE = "cpu"
MAX_LENGTH = 24

MODEL_NAME = "JackFram/llama-68m"
input_ids = torch.Tensor(
[
    [    1, 21429, 29899,  6451, 22545,  1078,   505, 1063],
]
).long()

position_ids = torch.Tensor(
[
    [    1, 2, 2,  3, 3,  3,   4, 5],
]
).long()

storage_ids = torch.Tensor(
[
    0, 1, 2,  3, 4,  5,  6, 7
]
).long()
attn_mask = torch.full((input_ids.shape[1], MAX_LENGTH), torch.tensor(torch.finfo(DTYPE).min, device=DEVICE), device=DEVICE)
attn_mask[:,:input_ids.shape[1]] = _make_causal_mask(input_ids_shape=input_ids.shape, dtype=DTYPE, device=DEVICE)

attn_mask[7][6] = torch.finfo(DTYPE).min
attn_mask[5][4] = torch.finfo(DTYPE).min

engine = InferenceEngine(max_length=MAX_LENGTH, model_name_or_path=MODEL_NAME, dtype=DTYPE, device=DEVICE)

engine.model_run(input_ids=input_ids, storage_ids=storage_ids, position_ids=position_ids, attention_mask=attn_mask[None, None,:,:])


extra_input_ids = torch.Tensor(
[
    [    1472,21429, 29899, 6451, 22545,  1078,   505, 1063],
]
).long()

extra_position_ids = torch.Tensor(
[
    [    4, 2, 7, 3, 4, 6, 8, 8],
]
).long()

extra_storage_ids = torch.Tensor(
[
    8, 9, 10,  11, 12,  13,  14, 15
]
).long()

extra_attn_mask = torch.full((extra_input_ids.shape[1], MAX_LENGTH), torch.tensor(torch.finfo(DTYPE).min, device=DEVICE), device=DEVICE)

extra_attn_mask[..., 0] = 0.0
extra_attn_mask[..., 1] = 0.0
extra_attn_mask[..., 4] = 0.0
extra_attn_mask[..., 7] = 0.0
extra_attn_mask[..., 8] = 0.0

k, v = engine.get_kv_cache()
engine.clear_kv()
engine.initialize_kv(k, v, input_ids.shape[1])
logits = engine.model_run(input_ids=extra_input_ids, storage_ids=extra_storage_ids, position_ids=extra_position_ids, attention_mask=extra_attn_mask[None, None,:,:])


print(logits)

















