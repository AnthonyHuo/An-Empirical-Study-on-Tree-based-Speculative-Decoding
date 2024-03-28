import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaConfig
import transformers
from datasets import load_dataset,load_from_disk
from data_converter import convert_dataset

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
data = load_from_disk("/home/zhuominc/data/c4_train")

model_config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=2048, #2*tokenizer.model_max_length
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=2,
        num_attention_heads=12,
        tie_word_embeddings=False,
        pad_token_id=1,
)
model = LlamaForCausalLM(model_config)

trainer = transformers.Trainer(
    model=model, 
    train_dataset=data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=100, 
        max_steps=200, 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        output_dir='/home/zhuominc/Sequoia_mingxiaohuo/outputs/'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

