import torch
torch.set_printoptions(profile="full")
import json
from copy import deepcopy
p = torch.tensor([0.0, 8.1058e-01, 8.5745e-02, 3.2890e-02, 1.7190e-02, 1.1283e-02, 7.3646e-03,
        5.5534e-03, 3.7437e-03, 3.1577e-03, 2.6452e-03, 1.9326e-03, 1.4449e-03,
        1.1956e-03, 1.0326e-03, 1.1496e-03, 1.0294e-03, 5.6361e-04, 5.4559e-04,
        5.0096e-04, 6.0773e-04, 2.8959e-04, 4.0065e-04, 3.3581e-04, 3.6632e-04,
        4.1801e-04, 3.9557e-04, 3.6161e-04, 2.9983e-04, 2.3403e-04, 1.7872e-04,
        2.7581e-04, 1.0436e-04])

max_branch = p.shape[0] - 1

max_depth = 25

max_budget = 1024

T = torch.zeros((max_budget + 1, max_depth + 1, max_branch + 1)).fill_(-torch.inf)
T_max = torch.zeros((max_budget + 1, max_depth + 1))
branch_map = {}
for l in range(1, max_depth + 1):
    for b in range(0, max_branch + 1):
        if b == 0:
            T[1][l][b] = 1.0
            branch_map[(1,l,b)] = []


for m in range(2, max_budget+1):
    for l in range(2, max_depth + 1):
        T[m][l][1] = 1 + p[1] * T[m-1][l-1].max()
        if T[m][l][1] > 0:
            branch_map[(m,l,1)] = [(m-1, l-1, T[m-1][l-1].argmax(dim=0).item())]
        for b in range(2, max_branch + 1):
            max_value = -torch.inf
            #new_y = -1
            for y in range(1, m):
                new_value = T[y][l][b-1] + p[b] * T[m-y][l-1].max()
                if new_value > max_value:
                    max_value = new_value
                    new_y = y
                max_value = max(max_value, new_value)
            T[m][l][b] = max_value
            if max_value >= 0:
                new_branch = T[m-new_y][l-1].argmax(dim=0).item()
                new_list :list = deepcopy(branch_map[(new_y, l, b-1)])
                new_list.append((m-new_y, l-1, new_branch))
                branch_map[(m,l,b)] = new_list

 
    

results = T.max(dim=2).values
print(results)
exit(0)
draft_inference_time = 0.0004
target_verify_time = [
                    0.031,
                    0.03143482923507691,
                    0.03153644561767578,
                    0.03149235248565674,
                    0.031020984649658204,
                    0.03269104957580567,
                    0.031229987144470214,
                    0.031333377361297605,
                    0.03102855443954468,
                    0.03154854774475098,
                    0.03275298833847046,
                    0.03128926038742066,
                    0.031237010955810548,
                    0.03158270597457886,
                    0.03199977397918701,
                    0.03207526922225952]


valid_budget = [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256]

dec_time = torch.inf
pairs = None
for i, b in enumerate(valid_budget):
    target_time = target_verify_time[i]
    for d, ac_len in enumerate(results[b]):
        if ac_len < 0:
            continue
        x = ((d-1) * draft_inference_time + target_time) / ac_len
        if x < dec_time:
            dec_time = x
            pairs = (b,d)

print(dec_time, target_verify_time[0] / dec_time, pairs)

exit(0)
(m, l) = pairs
b = T[m][l].argmax(dim=0).item()

positions = [0]
states = [(m,l,b)]
active = [True]
depth = [0]
Successors = [[]]
attention_mask = torch.zeros(m,m).long()
parents = [-1]
expand_lists = []
expand_branches = []
num_nodes = 1
while True:

    expand = []
    expand_branch = []
    for i, act in enumerate(active):
        if act: 
            if parents[i] != -1:
                attention_mask[i] = attention_mask[parents[i]]
            attention_mask[i][i] = 1
            expand.append(i)
            active[i] = False
            (x,y,z) = states[i]
            expand_branch.append(z)
            positions.extend(list(range(num_nodes, num_nodes + z)))
            Successors[i].extend(list(range(num_nodes, num_nodes + z)))
            Successors.extend([[] for _ in range(z)])
            parents.extend([i for _ in range(z)])
            depth.extend([depth[i] + 1 for _ in range(z)])
            states.extend(branch_map[(x,y,z)])
            assert len(branch_map[(x,y,z)]) == z
            num_nodes = num_nodes + z
    if len(expand) == 0:
        break
    expand_lists.append(expand)
    expand_branches.append(expand_branch)
    active.extend([True for _ in range(sum(expand_branch))])


assert num_nodes == m
assert len(positions) == m
assert len(depth) == m
grow_map = {
    "roots": expand_lists,
    "branches": expand_branches,
    "Successors":Successors,
    "mask": attention_mask,
    "depth": torch.LongTensor(depth),
    "size": num_nodes
}

path = "./growmaps/68m_7b.pt"

torch.save(grow_map, path)




     

















