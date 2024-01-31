from transformers import LlamaForCausalLM, LlamaTokenizer, DataCollatorForLanguageModeling, OPTForCausalLM, AutoTokenizer
import torch
import numpy as np 
from datasets import load_from_disk, Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.nn.functional import softmax
import accelerate
from accelerate import Accelerator
import argparse
from data_converter import convert_dataset
import argparse
from CoverTree import CoverTree
from Llama import LlamaForCausalLM_Attn
import time
from time import sleep
from utils import get_sampling_logits
import json
from Engine import GraphInferenceEngine, GraphInferenceEngineTG
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model')
parser.add_argument('--target', type=str, help='target model')
parser.add_argument('--dataset', type=str, default="./dataset/c4_small.json", help='dataset path')
parser.add_argument('--start', type=int, default=0, help='start')
parser.add_argument('--end', type=int, default=200, help='end')
parser.add_argument('--T', type=float, default=0.6, help='temperature')
parser.add_argument('--P', type=float, default=0.9, help='top_p')
parser.add_argument('--DP', type=float, default=1.1, help='draft_top_p')
parser.add_argument('--D', type=int, default=1, help='depth')
parser.add_argument('--B', type=int, default=16, help='budget')
parser.add_argument('--W', type=int, default=16, help='max width')
parser.add_argument('--M', type=int, default=256, help='max length')
parser.add_argument('--Mode', type=str, default="greedy", help='tree mode')
parser.add_argument('--decay', type=float, default=0.85, help='decay')
parser.add_argument('--negative', action='store_true')
parser.add_argument('--static', action='store_true')
parser.add_argument('--offloading', action='store_true')
args = parser.parse_args()
print(args)

def simulation_baseline(target_model : LlamaForCausalLM_Attn, dataloader: DataLoader, T=0.6, top_p=0.9):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    total_time = 0.0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if labels[0][-1] == -100: terminate = True
            target_kv = None
            torch.cuda.synchronize()
            t1 = time.time()
            inner_decoding_step = 0
            while inner_decoding_step < 128 and terminate == False:
               
                output = target_model(input_ids = input_ids, use_cache=True, past_key_values=target_kv)
                
                target_kv = output.past_key_values
                logits :torch.Tensor = output.logits[0][-1]
                logits = get_sampling_logits(logits=logits, top_p=top_p, T=T)
                p = softmax(logits / T, dim=-1)
                new_token = p.multinomial(num_samples=1).unsqueeze(0)
                input_ids = new_token
                
                num_decoding_steps += 1
                inner_decoding_step += 1
                if input_ids[0][-1] == 2: terminate = True
            torch.cuda.synchronize()
            t2 = time.time()
            total_time += (t2 - t1)
            
    print("total time :{:.5f}s, latency :{:.5f}s, decoding step: {}".format(total_time, total_time / num_decoding_steps, num_decoding_steps))
    return num_decoding_steps

def simulation_greedy_with_tree_fast(target_model : GraphInferenceEngineTG, draft_model: GraphInferenceEngine, dataloader: DataLoader, T=0.6, top_p=0.9, draft_top_p=1.1, budget=32, w=4, decay=0.85, negative=False, static=False, max_length=512):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    total_time = 0.0
    dtype = torch.float16
    attn_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, dtype=dtype, device='cuda:0')
    sequence = torch.tensor(list(range(max_length)), device='cuda:0').long().unsqueeze(-1)
    new_tokens_buffer =  torch.zeros(max_length).long().to('cuda:0')
    parents_buffer =  torch.zeros(max_length).long().to('cuda:0')
    position_ids = torch.zeros(max_length).long().to('cuda:0')
    active_mark = torch.zeros(max_length).bool().to('cuda:0')
    path = "./growmaps/68m_13b.pt"

    grow_map = torch.load(path)
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if labels[0][-1] == -100: terminate = True
            draft_kv = None
            target_kv = None
            draft_kv_len = 0
            target_kv_len = 0
            attn_mask.fill_(torch.finfo(dtype).min)
            spectree = CoverTree(prefix=input_ids.squeeze(0), device='cuda:0', temperature=T,
                                    top_p=top_p, 
                                    draft_kv_len=draft_kv_len, target_kv_len=target_kv_len,
                                    draft_model_engine=draft_model, target_model_engine=target_model, 
                                    max_length=max_length, grow_map=grow_map,
                                    attn_mask = attn_mask, sequence = sequence, new_tokens_buffer = new_tokens_buffer, 
                                    parents_buffer = parents_buffer, 
                                    position_ids = position_ids)
            torch.cuda.synchronize()
            t1 = time.time()
            while input_ids.shape[1] < 256 and terminate == False:
                spectree.construct_grow_map()
                valid_tokens, draft_kv_len, target_kv_len= spectree.verify()
                num_decoding_steps += (valid_tokens.shape[0] - input_ids.shape[1])
                num_large_model_steps += 1
                input_ids = valid_tokens.unsqueeze(0)
                if input_ids[0][-1] == 2: terminate = True
            
            torch.cuda.synchronize()
            t2 = time.time()
            total_time += (t2 - t1)
            draft_model.clear_kv()
            target_model.clear_kv()
    print("total time :{:.5f}s, latency :{:.5f}s, decoding step: {}, large model step: {}".format(total_time, total_time / num_decoding_steps, num_decoding_steps, num_large_model_steps))
    return num_decoding_steps / num_large_model_steps

def simulation_greedy_with_tree_fast_benchmark(target_model : GraphInferenceEngineTG, draft_model: GraphInferenceEngine, dataloader: DataLoader, T=0.6, top_p=0.9, draft_top_p=1.1, budget=32, w=4, decay=0.85, negative=False, static=False, max_length=512):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    initialize_time = 0.0
    speculate_time = 0.0
    verify_time = 0.0
    large_model_run = 0.0
    accept_loop = 0.0
    kv_select = 0.0
    sample_time = 0.0
    small_model_compute = 0.0
    dtype = torch.float16
    attn_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, dtype=dtype, device='cuda:0')
    sequence = torch.tensor(list(range(max_length)), device='cuda:0').long().unsqueeze(-1)
    new_tokens_buffer =  torch.zeros(max_length).long().to('cuda:0')
    parents_buffer =  torch.zeros(max_length).long().to('cuda:0')
    position_ids = torch.zeros(max_length).long().to('cuda:0')
    active_mark = torch.zeros(max_length).bool().to('cuda:0')
    path = "growmaps/68m_7b-64.pt"

    grow_map = torch.load(path)
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if labels[0][-1] == -100: terminate = True
            draft_kv_len = 0
            target_kv_len = 0
            attn_mask.fill_(torch.finfo(dtype).min)
            spectree = CoverTree(prefix=input_ids.squeeze(0), device='cuda:0', temperature=T,
                                    top_p=top_p, 
                                    draft_kv_len=draft_kv_len, target_kv_len=target_kv_len,
                                    draft_model_engine=draft_model, target_model_engine=target_model, 
                                    max_length=max_length, grow_map=grow_map,
                                    attn_mask = attn_mask, sequence = sequence, new_tokens_buffer = new_tokens_buffer, 
                                    parents_buffer = parents_buffer, 
                                    position_ids = position_ids)
            while input_ids.shape[1] < 256 and terminate == False:
                torch.cuda.synchronize()
                t1 = time.time()
                
                torch.cuda.synchronize()
                t2 = time.time()
                a, b = spectree.construct_grow_map(benchmark=True)
                sample_time += a
                small_model_compute += b
                torch.cuda.synchronize()
                t3 = time.time()
                valid_tokens, draft_kv_len, target_kv_len, x, y, z, terminate = spectree.verify(benchmark=True)
                large_model_run += x
                accept_loop += y
                kv_select += z
                torch.cuda.synchronize()
                t4 = time.time()
                num_decoding_steps += (valid_tokens.shape[0] - input_ids.shape[1])
                num_large_model_steps += 1
                input_ids = valid_tokens.unsqueeze(0)
                if (input_ids[0] == 2)._is_any_true() or (input_ids[0] == 0)._is_any_true() or input_ids.shape[1] >= 256:
                    terminate = True
                initialize_time += (t2 - t1)
                speculate_time += (t3 - t2)
                verify_time += (t4 - t3)
            draft_model.clear_kv()
            target_model.clear_kv()
            if num_large_model_steps > 0:
                print(num_decoding_steps / num_large_model_steps)
    print("total decoding steps: {}".format(num_decoding_steps), "large model steps: {}".format(num_large_model_steps), "avg decoding step: {}".format(num_decoding_steps / num_large_model_steps))
    print("initialization time:{}".format(initialize_time / num_large_model_steps), "speculate time: {}".format(speculate_time / num_large_model_steps),  "verify time: {}".format(verify_time / num_large_model_steps))
    print("large model run: {}".format(large_model_run / num_large_model_steps) , "accept loop: {}".format(accept_loop / num_large_model_steps), "kv select: {}".format(kv_select / num_large_model_steps))
    print("small model run: {}".format(small_model_compute / num_large_model_steps) , "sample time: {}".format(sample_time / num_large_model_steps))
    return num_decoding_steps / num_large_model_steps



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset_eval = convert_dataset(tokenizer=tokenizer,file_path=args.dataset).select(list(range(args.start, args.end)))
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset_eval, batch_size=1, collate_fn=data_collator)

if args.offloading:
    target_model = LlamaForCausalLM_Attn.from_pretrained(args.target, torch_dtype=torch.float16)
    target_model = accelerate.cpu_offload(target_model, execution_device="cuda:0")
elif args.Mode == 'baseline':
    target_model = LlamaForCausalLM_Attn.from_pretrained(args.target, torch_dtype=torch.float16).cuda()
else:
    draft_model = GraphInferenceEngine(max_length=args.M, model_name_or_path = args.model, dtype = torch.float16, device="cuda:0")
    target_model = GraphInferenceEngineTG(max_length=args.M, model_name_or_path = args.target, dtype = torch.float16, device="cuda:0")
    graph_capture_list = list(range(1, 129))
    draft_model.initialize_cuda_graph(graph_capture_list)

accelerator = Accelerator()
dataloader = accelerator.prepare(dataloader)

if args.Mode == 'benchmark':
    simulation_greedy_with_tree_fast_benchmark(target_model=target_model, draft_model=draft_model, dataloader=dataloader, T=args.T, top_p=args.P, budget=args.B, draft_top_p=args.DP, w=args.W, negative=args.negative, decay=args.decay, static=args.static, max_length=args.M)
elif args.Mode == 'baseline':
    simulation_baseline(target_model=target_model, dataloader=dataloader, T=args.T, top_p=args.P)
elif args.Mode == 'greedy':
    simulation_greedy_with_tree_fast(target_model=target_model, draft_model=draft_model, dataloader=dataloader, T=args.T, top_p=args.P, budget=args.B, draft_top_p=args.DP, w=args.W, negative=args.negative, decay=args.decay, static=args.static, max_length=args.M)

