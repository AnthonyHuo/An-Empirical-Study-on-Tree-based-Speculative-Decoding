import torch
from transformers import LlamaForCausalLM
import argparse
import time
import accelerate
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Llama-2-70b-hf",help='model')
parser.add_argument('--T', type=int, default=10, help='repeat times')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--P', type=int, default=128, help='prefix length')

parser.add_argument('--M', type=int, default=256, help='max length')
parser.add_argument('--D', type=int, default=8, help='dec length')
args = parser.parse_args()
print(args)

#target_model = LlamaForCausalLM_Attn.from_pretrained(args.target, torch_dtype=torch.float16, device_map="auto")
draft_model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
draft_model = accelerate.cpu_offload(draft_model, execution_device="cuda:0")


# draft_model = deepspeed.init_inference(draft_model,  
#                 dtype=torch.float16, enable_cuda_graph=True)
with torch.no_grad():
    T = args.T
    B = args.B
    P = args.P
    LEN = [1, 64, 128, 256, 512, 768, 1024, 1536, 2048, 2560, 3120, 4096]
    prefix = torch.randint(low=3, high=30000, size=(B, P)).cuda()
    past_key_values = draft_model(input_ids = prefix, use_cache=True).past_key_values

    for l in LEN:

        sentence = torch.randint(low=3, high=30000, size=(B,  l)).cuda()
        total_time = 0.0
        for _ in range(3):
            output = draft_model(input_ids = sentence, use_cache=True, past_key_values=past_key_values)
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(T):
        
            output = draft_model(input_ids = sentence, use_cache=True, past_key_values=past_key_values)
            
        
        torch.cuda.synchronize()
        t2 = time.time()
        total_time += (t2 - t1)
        print("Length :{}, inference time:{}".format(l, total_time / T))

        









