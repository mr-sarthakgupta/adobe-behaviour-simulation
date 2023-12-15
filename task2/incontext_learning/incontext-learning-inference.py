import gc
import ctypes
import json
import csv
from time import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging

import faiss
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def json_to_df(path_to_json):
    with open(path_to_json, 'r') as file:
        data = json.load(file)

    with open('temp.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['input', 'instruction', 'output'])
        for r in data:
            csvwriter.writerow([r['input'], r['instruction'], r['output']])
    return pd.read_csv('temp.csv')

def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

def load_sbert_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

def encode_sentences(model, sentences):
    return model.encode(sentences)

def build_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_index(query, index, model, k=5):
    query_embedding = model.encode([query])[0]
    _, result_indices = index.search(query_embedding.reshape(1, -1), k)
    return result_indices[0]

def load_llama_model():
    return LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16, load_in_8bit=True, device_map="cuda:0")

def load_llama_tokenizer():
    return LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

def generate_prompt(input_text, output_text, i):
    prompt = f'''<s>[INST] <<SYS>>
        You are a content generator, you have to generate the tweet text based on the expected likes and other additional information provided to you. Write the tweet text and nothing else.
        <</SYS>>

        {input_text[i[0]]} [/INST] {output_text[i[0]]} </s><s>{input_text[i[1]]} [/INST] {output_text[i[1]]} </s><s>{input_text[i[2]]} [/INST] {output_text[i[2]]} </s><s>[INST] {input_text[i[3]]} [/INST]'''
    return prompt

def main(input_path, eval_path):
    df_train = json_to_df(input_path)
    df_eval = json_to_df(eval_path)

    model_sbert = load_sbert_model()
    embeddings = encode_sentences(model_sbert, df_train['input'].tolist())
    index = build_faiss_index(embeddings)

    indices = [search_index(df_eval.loc[i, 'input'], index, model_sbert) for i in range(len(df_eval))]

    clean_memory()
    del index, model_sbert, embeddings

    input_text = df_train['input'].apply(lambda x: x[0:250]).tolist()
    output_text = df_train['output'].apply(lambda x: '\n tweet text: ' + x).tolist()

    model = load_llama_model()
    tokenizer = load_llama_tokenizer()
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1024)

    output = [pipe(generate_prompt(input_text, output_text, i))[0]['generated_text'] for i in indices]

    final_output = [o.split('[/INST]')[-1].split('tweet text: ')[-1] for o in output]

    json_file_path = 'output.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(final_output, json_file)

    logger.info(f'The list has been saved to {json_file_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and evaluation files.")
    parser.add_argument("--input", help="Path to the input JSON file", required=True)
    parser.add_argument("--eval", help="Path to the evaluation JSON file", required=True)

    args = parser.parse_args()
    main(args.input, args.eval)
