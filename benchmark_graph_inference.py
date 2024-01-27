from Engine import GraphInferenceEngine
import torch
from Llama_utils import _make_causal_mask
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="JackFram/llama-68m", help='model')
parser.add_argument('--T', type=int, default=100, help='time')
parser.add_argument('--M', type=int, default=256, help='max length')
parser.add_argument('--P', type=int, default=128, help='prefix length')
parser.add_argument('--D', type=int, default=8, help='dec length')
args = parser.parse_args()
PREFIX_LEN = args.P
MAX_LEN = args.M
DEC_LEN = args.D
MODEL_NAME = args.model
DTYPE = torch.float16
DEVICE = "cuda:0"
T = args.T
WARM_UP = 10
prefix = torch.randint(low=3, high=30000, size=(1, PREFIX_LEN), device=DEVICE)
prefix_storage_ids = torch.arange(PREFIX_LEN, device=DEVICE)
attn_mask = _make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
attn_mask = attn_mask[None, None, :, :]
prefix_position_ids = torch.arange(PREFIX_LEN, device=DEVICE).unsqueeze(0)

graph_engine = GraphInferenceEngine(max_length=MAX_LEN, model_name_or_path=MODEL_NAME, dtype=DTYPE, device=DEVICE)
graph_engine.initialize_cuda_graph([DEC_LEN])

graph_engine.inference(input_ids=prefix, storage_ids=prefix_storage_ids, position_ids=prefix_position_ids, attn_mask=attn_mask[..., :PREFIX_LEN,:])

input_ids = torch.randint(low=3, high=30000, size=(1, DEC_LEN), device=DEVICE)
storage_ids = torch.arange(DEC_LEN, device=DEVICE) + PREFIX_LEN
position_ids = storage_ids.clone().unsqueeze(0)

attn_mask=attn_mask[..., PREFIX_LEN: PREFIX_LEN + DEC_LEN,:].clone()

for _ in range(WARM_UP):
    graph_engine.graph_inference(input_ids=input_ids, storage_ids=storage_ids, position_ids=position_ids, attn_mask=attn_mask)


torch.cuda.synchronize()
t1 = time.time()

for _ in range(T):
    graph_engine.graph_inference(input_ids=input_ids, storage_ids=storage_ids, position_ids=position_ids, attn_mask=attn_mask)

torch.cuda.synchronize()
t2 = time.time()


print("Max Length :{}, Decode Length :{}, Prefix Length :{}, inference time:{}s".format(MAX_LEN, DEC_LEN, PREFIX_LEN, (t2 - t1)/ T))
