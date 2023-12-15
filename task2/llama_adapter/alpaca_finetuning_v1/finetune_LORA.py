import torch
from transformers import (
    TrainingArguments,
    LlamaForCausalLM,
    LlamaTokenizer,
)
# import bitsandbytes as bnb
from peft import LoraConfig, PeftModel,get_peft_model
from trl import SFTTrainer
from llama import ModelArgs, Tokenizer, Transformer
import json
from datasets import Dataset
import pandas as pd

ann = json.load(open("finetuning.json"))
model = LlamaForCausalLM.from_pretrained("llama/weights_7b/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16,load_in_8bit=True, device_map="cuda:0")
tokenizer = LlamaTokenizer.from_pretrained("llama/weights_7b/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    target_modules = ["q_proj","v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()
def formatting_func(example):
    output_text =[]
    for i in range(len(example)):
        text = 'abcd'
        # text = f"###Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n ###Instruction:\n {example[i]['instruction']} \n\n   Input:\n  {example[i]['input']}\n\n### Response: {example[i]['output']}"
        output_text.append(text)
    return output_text
column = ["text"]
output_text = formatting_func(ann)
dataset = Dataset.from_pandas(pd.DataFrame(output_text, columns=column))

training_arguments = TrainingArguments(
    output_dir="finetuned_llama",
    num_train_epochs=3,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=1,
    save_steps=50,
    logging_steps=50,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant"
)

fine_tuning = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    dataset_text_field = "text",
    args=training_arguments,
)

fine_tuning.train()
fine_tuning.model.save_pretrained(model)