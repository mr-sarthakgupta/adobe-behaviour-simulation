# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple
from nltk.translate.bleu_score import sentence_bleu
import evaluate
import pandas as pd

import fire
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import LLaMA, ModelArgs, Tokenizer, Transformer

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


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    quantizer: bool=False,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    print(checkpoints)
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load("alpaca_finetuning_v1/llama/weights_7b/7B/consolidated.00.pth", map_location="cpu")
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
    model_args.adapter_layer = int(adapter_checkpoint["adapter_query.weight"].shape[0] / model_args.adapter_len)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    print(model)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    model.load_state_dict(adapter_checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def score(predictions,references):  
  rouge = evaluate.load('rouge')
  rouge_score = rouge.compute(predictions=[predictions], references=[references])
  reference = [references.split()]
  prediction = predictions.split()
  bleu_score = sentence_bleu(reference, prediction)
  return [bleu_score,rouge_score]

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    temperature: float = 0.9,
    top_p: float = 0,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    quantizer: bool = False,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    pre_prompts = json.load(open("output_content_time.json"))
    print(len(pre_prompts))
    generator = load(ckpt_dir, tokenizer_path, adapter_path, local_rank, world_size, max_seq_len, max_batch_size, quantizer)
    prompts = [PROMPT_DICT["prompt_input"].format_map({"instruction": x["instruction"], "input": x["input"]}) for x in pre_prompts]

    start_time = time.time()
    results = []
    c = 0
    for i in range(0, len(prompts), max_batch_size):
        results.extend(generator.generate(prompts[i : i + max_batch_size], max_gen_len=340, temperature=temperature, top_p=top_p))
        c += max_batch_size
        print(f"num_samples done: {c}, time elapsed: {time.time() - start_time:.2f} seconds")
    stop_time = time.time()
    print(f"Generated {len(results)} in {stop_time - start_time:.2f} seconds")
    df = pd.DataFrame({'content': results})
    df.to_csv('company_time.csv')
    print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
