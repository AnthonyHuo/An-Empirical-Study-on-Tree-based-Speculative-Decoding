import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity
N = 8
M = 16
T = 10000

def sample(x:torch.Tensor, y: torch.LongTensor, z: torch.Tensor, m :int):
    return (x[y].log()/z).topk(k=m, sorted=False).indices

sample_compiled = torch.compile(sample, mode="reduce-overhead")

rand = torch.empty((8, 32000), dtype=torch.float16).uniform_().to("cuda:0")
sampling_q = torch.rand((N, 32000), dtype=torch.float16).to("cuda:0")
idx_list = list(range(N))
idx_list = torch.Tensor(idx_list).to("cuda:0").long()
for _ in range(100):
        sample_compiled(rand, idx_list, sampling_q, M)
torch.cuda.synchronize()
with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    for _ in range(T):
        sample_compiled(rand, idx_list, sampling_q, M)
print(prof.key_averages().table(sort_by="cpu_time_total"))