import json
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor
from llava_model import LlavaForConditionalGeneration,GraphInferenceEngineTG
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
from GreedySTree_llava_dy import GreedySTree
from Llama import LlamaForCausalLM_Attn
import time
from time import sleep
from utils import get_sampling_logits, _make_causal_mask, get_residual, cuda_graph_for_residual, cuda_graph_for_sampling_without_replacement, cuda_graph_for_sampling_with_replacement,cuda_graph_for_sampling_argmax 
import json
from Engine import GraphInferenceEngine
# , GraphInferenceEngineTG

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model')
parser.add_argument('--target', type=str, help='target model')
parser.add_argument('--dataset', type=str, default="dataset/c4_small.json", help='dataset path')
parser.add_argument('--growmap', type=str, default="growmaps/68m_7b-64.pt", help='dataset path')
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
def simulation_greedy_with_tree_fast(target_model : GraphInferenceEngineTG, draft_model: GraphInferenceEngine, T=0.6, top_p=0.9, 
            draft_top_p=1.1, budget=32, w=4, decay=0.85, negative=False, static=False, 
            max_length=512, residual_graph=None, grow_map=None, sampling_callables = None,
            sample_gather_indices = None):
    num_eval_steps = 200
    num_decoding_steps = 0
    num_large_model_steps = 0
    total_time = 0.0
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    # Load the JSON file
    with open('/home/zhuominc/Sequoia_mingxiaohuo/instruct150k_subset.json', 'r') as file:
        data = json.load(file)

    with torch.no_grad():
        for entry in tqdm(data[:200], desc="Processing labels"):
            # Construct the prompt
            prompt = f"USER: {entry['conversations'][0]['value']} ASSISTANT:"
            image_id = entry['id']  # Extract image ID
            image_path = f"/home/zhuominc/Sequoia_mingxiaohuo/train2014/COCO_train2014_{image_id.zfill(12)}.jpg"
            image = Image.open(image_path)
            # Process the prompt and image
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            input_ids = inputs['input_ids']
            terminate = False
            if input_ids[0][-1] == -100: terminate = True
            inputs.update( {
                "pixel_values": inputs['pixel_values'].to('cuda:0'),
                "input_ids": inputs['input_ids'].to('cuda:0')
            })
            torch.cuda.synchronize()
            t1 = time.time()
            dtype = torch.float16
            attn_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, dtype=dtype, device='cuda:0')
            sequence = torch.tensor(list(range(max_length)), device='cuda:0').long().unsqueeze(-1)
            new_tokens_buffer =  torch.zeros(max_length).long().to('cuda:0')
            parents_buffer =  torch.zeros(max_length).long().to('cuda:0')
            position_ids = torch.zeros(max_length).long().to('cuda:0')
            draft_kv_len = 0
            target_kv_len = 0
            attn_mask.fill_(torch.finfo(dtype).min)
            spectree = GreedySTree(prefix=input_ids.squeeze(0), device='cuda:0', temperature=T,
                                    top_p=top_p,
                                    draft_kv_len=draft_kv_len, target_kv_len=target_kv_len,
                                    draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_length, grow_map=grow_map,
                                    attn_mask = attn_mask, sequence = sequence, new_tokens_buffer = new_tokens_buffer, 
                                    parents_buffer = parents_buffer, 
                                    position_ids = position_ids,
                                    residual_graph = residual_graph,
                                    sampling_callables=sampling_callables,
                                    sample_gather_indices = sample_gather_indices)
            torch.cuda.synchronize()
            t1 = time.time()
            while input_ids.shape[1] < 256 and terminate == False:
                # spectree.construct_grow_map()
                spectree.construct_dynamic_tree()
                valid_tokens, draft_kv_len, target_kv_len, terminate = spectree.verify()
                
                num_decoding_steps += (valid_tokens.shape[0] - input_ids.shape[1])
                num_large_model_steps += 1
                input_ids = valid_tokens.unsqueeze(0)
                if (input_ids[0][-1] == 2) or (input_ids[0][-1] == 0): terminate = True
            
            torch.cuda.synchronize()
            t2 = time.time()
            total_time += (t2 - t1)
            draft_model.clear_kv()
            target_model.clear_kv()
    print("total time :{:.5f}s, latency :{:.5f}s, decoding step: {}, large model step: {}".format(total_time, total_time / num_decoding_steps, num_decoding_steps, num_large_model_steps))
    return num_decoding_steps / num_large_model_steps

def simulation_greedy_with_tree_fast_benchmark(target_model : GraphInferenceEngineTG, draft_model: GraphInferenceEngine, T=0.6, top_p=0.9, 
                draft_top_p=1.1, budget=32, w=4, decay=0.85, negative=False, static=False, 
                max_length=512, residual_graph=None, grow_map=None, sampling_callables = None,
                sample_gather_indices = None):
    return None
# Load the model and processor
def simulation_baseline(target_model : GraphInferenceEngineTG, T=0.6, top_p=0.9, max_length=1024):
    num_eval_steps = 200
    num_decoding_steps = 0
    total_time = 0.0
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    # Load the JSON file
    with open('/home/zhuominc/Sequoia_mingxiaohuo/instruct150k_subset.json', 'r') as file:
        data = json.load(file)

    with torch.no_grad():
        for entry in tqdm(data[:200], desc="Processing labels"):
            # Construct the prompt
            prompt = f"USER: {entry['conversations'][0]['value']} ASSISTANT:"
            image_id = entry['id']  # Extract image ID
            image_path = f"/home/zhuominc/Sequoia_mingxiaohuo/train2014/COCO_train2014_{image_id.zfill(12)}.jpg"
            image = Image.open(image_path)

            # Process the prompt and image
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            input_ids = inputs['input_ids']
            terminate = False
            if input_ids[0][-1] == -100: terminate = True
            inputs.update( {
                "pixel_values": inputs['pixel_values'].to('cuda:0'),
                "input_ids": inputs['input_ids'].to('cuda:0')
            })
            torch.cuda.synchronize()
            t1 = time.time()
            position_ids = torch.arange(max_length).to('cuda:0').unsqueeze(0)
            storage_ids = torch.arange(max_length).to('cuda:0')
            attn_mask = _make_causal_mask((max_length, max_length), target_model.dtype, target_model.device)
            inner_decoding_step = 0
            start_length = 0
            while inner_decoding_step < 128 and terminate == False:
                if inner_decoding_step == 0:
                    inputs_embeds, attention_mask, position_id,storage_id = target_model.inference_image(input_ids = inputs['input_ids'], storage_ids=storage_ids[:start_length], pixel_values=inputs['pixel_values'],
                                                    position_ids = position_ids[..., :start_length], 
                                                    attn_mask=attn_mask[:start_length, :start_length][None, None, :, :])
                    start_length = inputs_embeds.shape[1]
                    logits = target_model.inference(input_ids = None, inputs_embeds = inputs_embeds, storage_ids=storage_ids[:start_length], pixel_values=inputs['pixel_values'],
                                                    position_ids = position_ids[..., :start_length], 
                                                    attn_mask=attn_mask[:start_length, :start_length][None, None, :, :])[0][-1]
                else:
                    logits = target_model.inference(input_ids = inputs['input_ids'], inputs_embeds = None, storage_ids=storage_ids[start_length + inner_decoding_step-1 : start_length + inner_decoding_step],
                                                    position_ids = position_ids[..., start_length + inner_decoding_step-1 : start_length + inner_decoding_step], 
                                                    attn_mask=attn_mask[start_length + inner_decoding_step-1 : start_length + inner_decoding_step, :start_length + inner_decoding_step][None, None, :, :])[0][-1]
                
                logits = get_sampling_logits(logits=logits, top_p=top_p, T=T)
                
                p = softmax(logits / T, dim=-1)
                new_token = p.multinomial(num_samples=1).unsqueeze(0)
                inputs['input_ids'] = new_token
                outputs = processor.batch_decode(new_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                print(outputs)
                num_decoding_steps += 1
                inner_decoding_step += 1
                if input_ids[0][-1] == 2: 
                    terminate = True
            torch.cuda.synchronize()
            t2 = time.time()
            total_time += (t2 - t1)
            target_model.clear_kv()
            
    print("total time :{:.5f}s, latency :{:.5f}s, decoding step: {}".format(total_time, total_time / num_decoding_steps, num_decoding_steps))
    return num_decoding_steps
# processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
# prompt = "how are"
# tokens = tokenizer(prompt)
# print(tokens)

# image_path = f"/home/zhuominc/Sequoia_mingxiaohuo/train2014/COCO_train2014_000000311112.jpg"
# image = Image.open(image_path)

# # Process the prompt and image
# inputs = processor(text=prompt, images=image, return_tensors="pt")
# # Tokenizing the text
# print(inputs['input_ids'])
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# if args.dataset == 'openwebtext':
#     tokenized_dataset_eval = load_from_disk("dataset/openwebtext_eval").select(list(range(args.start, args.end)))
# else:
#     tokenized_dataset_eval = convert_dataset(tokenizer=tokenizer,file_path=args.dataset).select(list(range(args.start, args.end)))

# data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
# dataloader = DataLoader(tokenized_dataset_eval, batch_size=1, collate_fn=data_collator, shuffle=False)


if args.Mode == 'baseline':
    # target_model =  LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    target_model =  GraphInferenceEngineTG(max_length=args.M, model_name_or_path = args.target, dtype = torch.float16, device="cuda:0")
else:
    draft_model = GraphInferenceEngine(max_length=args.M, model_name_or_path = args.model, dtype = torch.float16, device="cuda:0")
    # target_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    target_model =  GraphInferenceEngineTG(max_length=args.M, model_name_or_path = args.target, dtype = torch.float16, device="cuda:0")
    graph_capture_list = list(range(1, 129))
    draft_model.initialize_cuda_graph(graph_capture_list)
    residual_graph = cuda_graph_for_residual()
    path = args.growmap
    grow_map = torch.load(path)

    tree_size = grow_map["size"]
    idx_lists = grow_map["roots"]
    branch_lists = grow_map['branches']
    draft_step = len(grow_map["roots"])
    sampling_callables = {}
    sample_gather_indices = {}
    for i in range(draft_step - 1):
        idx_len = len(idx_lists[i])
        num_samples = max(branch_lists[i])
        sampling_callables[i] = cuda_graph_for_sampling_argmax(
            max_length=args.M, idx_len=idx_len, num_samples=num_samples,
            temperature=args.T, tree_size=tree_size)  
    for i in range(draft_step - 1):
        ith_gather_list = []
        max_num_samples = max(branch_lists[i])
        for j, branch in enumerate(branch_lists[i]):
            branch_index = torch.arange(branch, device="cuda:0", dtype=torch.long)
            branch_index = branch_index + j * max_num_samples
            ith_gather_list.append(branch_index)
        ith_gather_list = torch.cat(ith_gather_list)
        sample_gather_indices[i] = ith_gather_list
    

    






# accelerator = Accelerator()
# dataloader = accelerator.prepare(dataloader)

#warm up functions:

if args.Mode == 'benchmark':
    simulation_greedy_with_tree_fast_benchmark(target_model=target_model, draft_model=draft_model, T=args.T, top_p=args.P, budget=args.B, draft_top_p=args.DP, w=args.W, negative=args.negative, decay=args.decay, static=args.static, 
                                               max_length=args.M, residual_graph = residual_graph, grow_map = grow_map, sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices)
elif args.Mode == 'baseline':
    simulation_baseline(target_model=target_model, T=args.T, top_p=args.P)
elif args.Mode == 'greedy':
    simulation_greedy_with_tree_fast(target_model=target_model, draft_model=draft_model, T=args.T, top_p=args.P, budget=args.B, draft_top_p=args.DP, w=args.W, negative=args.negative, decay=args.decay, static=args.static, 
                                     max_length=args.M, residual_graph = residual_graph, grow_map = grow_map, sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices)

