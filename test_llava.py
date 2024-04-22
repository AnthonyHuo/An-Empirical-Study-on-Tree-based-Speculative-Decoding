import json
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor
from llava_model import LlavaForConditionalGeneration
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
from GreedySTree_dy import GreedySTree
from Llama import LlamaForCausalLM_Attn
import time
from time import sleep
from utils import get_sampling_logits, _make_causal_mask, get_residual, cuda_graph_for_residual, cuda_graph_for_sampling_without_replacement, cuda_graph_for_sampling_with_replacement,cuda_graph_for_sampling_argmax 
import json
from Engine import GraphInferenceEngine, GraphInferenceEngineTG

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
    return None
def simulation_greedy_with_tree_fast_benchmark(target_model : GraphInferenceEngineTG, draft_model: GraphInferenceEngine, T=0.6, top_p=0.9, 
                draft_top_p=1.1, budget=32, w=4, decay=0.85, negative=False, static=False, 
                max_length=512, residual_graph=None, grow_map=None, sampling_callables = None,
                sample_gather_indices = None):
    return None
# Load the model and processor
def simulation_baseline(target_model : GraphInferenceEngineTG, T=0.6, top_p=0.9, max_length=256):
    num_eval_steps = 200
    num_decoding_steps = 0
    total_time = 0.0
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    # Load the JSON file
    with open('/home/zhuominc/Sequoia_mingxiaohuo/instruct150k_subset.json', 'r') as file:
        data = json.load(file)

    with torch.no_grad():
        responses = []
        for entry in tqdm(data[:200], desc="Processing labels"):
                # Construct the prompt
                prompt = f"USER: {entry['conversations'][0]['value']} ASSISTANT:"
                image_id = entry['id']  # Extract image ID
                image_path = f"/home/zhuominc/Sequoia_mingxiaohuo/train2014/COCO_train2014_{image_id.zfill(12)}.jpg"
                image = Image.open(image_path)

                # Process the prompt and image
                inputs = processor(text=prompt, images=image, return_tensors="pt")

                # Generate response
                generate_ids = target_model.generate(**inputs, max_new_tokens=35)
                output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                print(output)
                # Store or display the output
                responses.append(output)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
prompt = "USER: What could be the reason for the woman raising her cell phone in the air? ASSISTANT:"

# Tokenizing the text
tokens = tokenizer(prompt)
print(tokens)
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# if args.dataset == 'openwebtext':
#     tokenized_dataset_eval = load_from_disk("dataset/openwebtext_eval").select(list(range(args.start, args.end)))
# else:
#     tokenized_dataset_eval = convert_dataset(tokenizer=tokenizer,file_path=args.dataset).select(list(range(args.start, args.end)))

# data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
# dataloader = DataLoader(tokenized_dataset_eval, batch_size=1, collate_fn=data_collator, shuffle=False)


if args.Mode == 'baseline':
    target_model =  LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
else:
    draft_model = GraphInferenceEngine(max_length=args.M, model_name_or_path = args.model, dtype = torch.float16, device="cuda:0")
    target_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
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
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
