import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity
T = 10000
n_branch_list = [2, 4, 1, 5, 6, 1, 3, 7]
tokens = torch.zeros(256).long().cuda()
new_tokens_set = torch.randint(high = 32000, size=(8, 16)).cuda()

torch.cuda.synchronize()
with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    for _ in range(T):
        finished_tokens = 0
        for i in range(8):
                        n_branch = n_branch_list[i]
                        tokens[finished_tokens: finished_tokens + n_branch] = new_tokens_set[i][:n_branch]
                        finished_tokens += n_branch

print(prof.key_averages().table(sort_by="cpu_time_total"))