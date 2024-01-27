import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import time

from tqdm.auto import tqdm

import torch
import transformers
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer, LlamaForCausalLM, LlamaModel, LlamaTokenizer
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    MBartTokenizer,
    MBartTokenizerFast,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import pandas as pd

from datasets import load_dataset, concatenate_datasets

from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.28.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")


def convert_yahoo_dataset(tokenizer, seq_len = 256):
    dataset = load_dataset("yahoo_answers_topics", split="train")
    def tokenize_function(examples):
            tk = tokenizer(examples["best_answer"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
            ret = {
                'input_ids': tk['input_ids'],
                'attention_mask': tk['attention_mask'],
                'labels': examples['topic']
            }
            return ret
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['id', 'topic', 'question_title', 'question_content', 'best_answer'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset
def convert_yahoo_dataset_eval(tokenizer, seq_len = 256):
    dataset = load_dataset("yahoo_answers_topics", split="test")
    def tokenize_function(examples):
            tk = tokenizer(examples["best_answer"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
            ret = {
                'input_ids': tk['input_ids'],
                'attention_mask': tk['attention_mask'],
                'labels': examples['topic']
            }
            return ret
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['id', 'topic', 'question_title', 'question_content', 'best_answer'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset
def convert_ag_dataset(tokenizer, seq_len = 256):
    dataset = load_dataset("ag_news", split="train")
    def tokenize_function(examples):
            tk = tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
            ret = {
                'input_ids': tk['input_ids'],
                'attention_mask': tk['attention_mask'],
                'labels': examples['label']
            }
            return ret
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text', 'label'])

    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset

def convert_ag_dataset_eval(tokenizer, seq_len = 256):
    dataset = load_dataset("ag_news", split="test")

    def tokenize_function(examples):
            tk = tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
            ret = {
                'input_ids': tk['input_ids'],
                'attention_mask': tk['attention_mask'],
                'labels': examples['label']
            }
            return ret
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset
def convert_openweb_dataset_eval(tokenizer, seq_len = 256):
    dataset = load_dataset("openwebtext", split="train[8010000:]")
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataset.save_to_disk("/home/zhuominc/SpeculativeDecoding/data/openwebtext_eval_long")
    return dataset

def convert_c4_dataset(tokenizer, seq_len = 256):
    dataset = load_dataset("c4", "realnewslike", split="train")
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text', 'timestamp', 'url'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    #dataset.save_to_disk("/home/zhuominc/SpeculativeDecoding/data/c4_train")
    return dataset

def convert_c4_dataset_eval(tokenizer, seq_len = 256):
    dataset = load_dataset("c4", "realnewslike", split="validation[0:100]")
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text', 'timestamp', 'url'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    #dataset.save_to_disk("/home/zhuominc/SpeculativeDecoding/data/validation")
    return dataset
def convert_dataset(tokenizer, file_path):
    dataset = load_dataset("json", data_files=file_path, split="train")
    def tokenize_function(examples):
            input_ids = torch.Tensor(examples['input_ids'])
            labels = input_ids.clone()
            if tokenizer.pad_token_id is not None:
                 labels[labels == tokenizer.pad_token_id] = -100
            ret = {
                "input_ids": input_ids,
                "labels": labels
            }
            return ret
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['input_tokens'])
    dataset.set_format(type='torch', columns=['input_ids', "labels"])
    return dataset

def convert_dataset_truncate(tokenizer, file_path, truncate_length = 100):
    dataset = load_dataset("json", data_files=file_path, split="train")
    def tokenize_function(examples):
            src = tokenizer(examples['input'],return_tensors='pt',max_length=128,padding=True,truncation=True)
            dst = tokenizer(examples['output'],return_tensors='pt',max_length=129,padding=True,truncation=True)
            src_input_ids = src['input_ids']
            dst_input_ids = dst['input_ids']
            src_attn_mask = src['attention_mask']
            dst_attn_mask = dst['attention_mask']
            input_ids = torch.cat([src_input_ids, dst_input_ids[...,1:]], dim=1).long()
            attn_mask = torch.cat([src_attn_mask, dst_attn_mask[...,1:]], dim=1).long()
            labels = input_ids.clone()
            if tokenizer.pad_token_id is not None:
                 labels[labels == tokenizer.pad_token_id] = -100
            ret = {
                "input_ids": input_ids[..., truncate_length:],
                "attention_mask": attn_mask[..., truncate_length:],
                "labels": labels[..., truncate_length:]
            }
            return ret
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['input', 'output'])
    dataset.set_format(type='torch', columns=['input_ids', "attention_mask","labels"])
    return dataset
def generate_dataset(tokenizer, seq_len = 256):
    
    dataset1 = load_dataset("c4", "realnewslike", split="validation")
    dataset2 = load_dataset("c4", "en", split="validation[:20000]")
    dataset3 = load_dataset("openwebtext", split="train[8000000:]")
    dataset1.remove_columns(['timestamp', 'url'])
    dataset2.remove_columns(['timestamp', 'url'])
    dataset_large = concatenate_datasets([dataset1, dataset2, dataset3])
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset_large = dataset_large.map(tokenize_function, batched=True, remove_columns=['text', 'timestamp', 'url'])
    dataset_large.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataset_large.save_to_disk("/home/zhuominc/SpeculativeDecoding/data/dataset_large_valid")
    return dataset_large
def main():
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    generate_dataset(tokenizer=tokenizer)

if __name__ == '__main__':
    main()