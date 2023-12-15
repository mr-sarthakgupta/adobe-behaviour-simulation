# !pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 sentencepiece

import os
import json
import torch
from datasets import load_dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)
from datasets import Dataset
import pandas as pd
import time

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16,load_in_4bit=True, device_map="cuda:0", token="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")
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token="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")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

data = json.load(open("output-content-company.json"))

adapter_checkpoint = torch.load("adapter_adapter_len10_layer28_epoch5.pth")

model.load_state_dict(adapter_checkpoint, strict=False)

prompts = [PROMPT_DICT["prompt_input"].format_map({"instruction": x["instruction"], "input": x["input"]}) for x in data][:4]

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=350)

# print(len(prompts))

start_time = time.time()

outputs = pipe(prompts, batch_size=30)

end_time = time.time()

print(end_time - start_time)
print(outputs)
print('the end')