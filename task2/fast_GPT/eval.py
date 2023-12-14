## Part of Code used from here:https://github.com/pytorch-labs/gpt-fast/blob/main/eval.py

import sys
import time
from pathlib import Path
from typing import Optional
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline

import torch
import torch._dynamo.config
import torch._inductor.config

torch._dynamo.config.automatic_dynamic_shapes = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.triton.cudagraphs = True
torch._dynamo.config.cache_size_limit = 100000

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# hacky path setup for lm-evaluation-harness
import os
import sys

from sentencepiece import SentencePieceProcessor

from gpt_fast.model import LLaMA

lm_evaluation_harness_path = "/".join(
    os.getcwd().split("/")[:-1] + ["lm-evaluation-harness"]
)
sys.path.insert(0, lm_evaluation_harness_path)
import lm_eval
import main as lm_evaluation_harness_main

from gpt_fast.generate import _load_model, encode_tokens, model_forward


if __name__ == "__main__":
    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), tokenizer_path

    device = "cuda"
    precision = torch.bfloat16

    print("Loading model ...")
    model = _load_model(checkpoint_path, device, precision, False)

    torch.cuda.synchronize()

    model.eval()

    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))

    torch.manual_seed(1234)

    pipe = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer, max_length=1024
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
