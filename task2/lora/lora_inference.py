import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
)
import bitsandbytes as bnb
import json
import pandas as pd
import logging

# Load the fine-tuned model
fine_tuned_model = LlamaForCausalLM.from_pretrained(
    "../../data/Llama_finetuned",
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    device_map="cuda:0",
)

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def inference(input_text):
    # Tokenize the input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # Generate output using the fine-tuned model
    output_ids = fine_tuned_model.generate(input_ids)

    # Decode the output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text


def json_to_df(path_to_json):
    with open(path_to_json, "r") as file:
        data = json.load(file)

    with open("temp.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["input", "instruction", "output"])
        for r in data:
            csvwriter.writerow([r["input"], r["instruction"], r["output"]])
    return pd.read_csv("temp.csv")


df_train = json_to_df("input.json")
df_eval = json_to_df("eval.json")


input_text = df_train["input"].apply(lambda x: x[0:250]).tolist()
output_text = df_train["output"].apply(lambda x: "\n tweet text: " + x).tolist()

pipe = pipeline(
    task="text-generation", model=fine_tuned_model, tokenizer=tokenizer, max_length=1024
)

output = [
    pipe(generate_prompt(input_text, output_text, i))[0]["generated_text"]
    for i in indices
]

final_output = [o.split("[/INST]")[-1].split("tweet text: ")[-1] for o in output]

json_file_path = "output.json"
with open(json_file_path, "w") as json_file:
    json.dump(final_output, json_file)

logger.info(f"The list has been saved to {json_file_path}")
