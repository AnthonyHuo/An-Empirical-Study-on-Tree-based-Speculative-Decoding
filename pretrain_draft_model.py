import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaConfig
import transformers
from datasets import load_dataset,load_from_disk
from data_converter import convert_dataset
import wandb

wandb.init(project="llm_speculative_decoding")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
data = load_from_disk("/home/zhuominc/data/c4_train")
data_eval = load_from_disk("/home/zhuominc/data/c4_validation")

model_config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=2048, #2*tokenizer.model_max_length
        hidden_size=1024,
        intermediate_size=5120,
        num_hidden_layers=4,
        num_attention_heads=16,
        tie_word_embeddings=False,
        pad_token_id=1,
)
model = LlamaForCausalLM(model_config)

trainer = transformers.Trainer(
    model=model, 
    train_dataset=data,
    eval_dataset=data_eval,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=128, 
        gradient_accumulation_steps=4,
        warmup_steps=1000, 
        num_train_epochs=1.2,
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        eval_steps=2000,
        save_steps=2000,
        save_total_limit=3,
        report_to="wandb",
        output_dir='/home/zhuominc/Sequoia_mingxiaohuo/outputs/'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
wandb.finish()