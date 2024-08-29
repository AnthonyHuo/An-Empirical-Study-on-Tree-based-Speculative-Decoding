import os
os.environ["CUDA_VISIBLE_DEVICES"]="7,8,9"
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaConfig
import transformers
from datasets import load_dataset,load_from_disk
from data_converter import convert_dataset,convert_openweb_dataset_eval
import wandb

wandb.init(project="llm_speculative_decoding")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
# data = load_from_disk("/home/zhuominc/Sequoia_mingxiaohuo/dataset/openwebtext_eval").select(list(range(0, 3500)))
# dataset = load_dataset("openwebtext", split="train[:8010000]") 
dataset = load_dataset("cnn_dailymail", "1.0.0", split="train")
# print(data.shape)
# def tokenize_function(examples):
#         return tokenizer(examples["text"], return_tensors='pt',max_length=256,padding=True,truncation=True)
def tokenize_function(examples):
        return tokenizer(examples["article"], return_tensors='pt',max_length=256,padding=True,truncation=True)
# dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'],batch_size=2000, num_proc=16,load_from_cache_file=True)
dataset = dataset.map(tokenize_function, batched=True, remove_columns=['article'],batch_size=2000, num_proc=16,load_from_cache_file=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
# model_config = LlamaConfig(
#         vocab_size=tokenizer.vocab_size,
#         max_position_embeddings=2048, #2*tokenizer.model_max_length
#         hidden_size=768,
#         intermediate_size=3072,
#         num_hidden_layers=2,
#         num_attention_heads=12,
#         tie_word_embeddings=False,
#         pad_token_id=1,
# )
# model = LlamaForCausalLM(model_config)
# model = LlamaForCausalLM.from_pretrained('/home/zhuominc/Sequoia_mingxiaohuo/outputs/checkpoint-1000/')
model = LlamaForCausalLM.from_pretrained('JackFram/llama-68m')
trainer = transformers.Trainer(
    model=model, 
    train_dataset=dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=64, 
        gradient_accumulation_steps=64,
        warmup_steps=25, 
        max_steps=100, 
        learning_rate=5e-5, 
        fp16=True,
        logging_steps=1, 
        save_steps=50,
        report_to="wandb",
        output_dir='/home/zhuominc/Sequoia_mingxiaohuo/outputs2/'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference! 5e-5 for 68m
trainer.train()

