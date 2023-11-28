# 1. generates and stores embeddings

from PIL import Image
import PIL
import requests
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
import torch
import pandas as pd
from io import BytesIO
import numpy as np
from tqdm import tqdm

tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TRANSFORMERS_CACHE'] = '/scratch/apoorva_v.iitr/huggingface/hub'
os.environ['PYTORCH_TRANSFORMERS_CACHE'] = '/scratch/apoorva_v.iitr/huggingface/hub'
os.environ['CURL_CA_BUNDLE'] = ''

data = pd.read_csv("../content_simulation_train.csv")

image_data = data[data['media'].str.contains("Photo")]
video_data = data[data['media'].str.contains("Video")]
gif_data = data[data['media'].str.contains("Gif")]

print(f'len image data: {len(image_data)}, len video data: {len(video_data)}, len gif data: {len(gif_data)}')

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

image_embeddings = []
text_embeddings = []
likes = []

# loop through images in ../media/images and generate embeddings for those whose name matches with index in image_data
for index, row in tqdm(image_data.iterrows(), total=len(image_data)):
    image_path = f"../media/images/{index}.jpg"
    if os.path.exists(image_path):
        try:
            image = Image.open(image_path)
            image_inputs = processor(images=image, return_tensors="pt")
            image_features = model.get_image_features(**image_inputs)

            text = row['content']
            text_inputs = tokenizer(text, padding=True, return_tensors="pt")
            text_features = model.get_text_features(**text_inputs)
            
            image_embeddings.append(image_features.cpu().detach().numpy())
            text_embeddings.append(text_features.cpu().detach().numpy())
            likes.append(row['likes'])
        except:
            continue

    if len(image_embeddings) == 200:
        print('saving progress')

        data = {
            'image_embeddings': np.array(image_embeddings),
            'text_embeddings': np.array(text_embeddings),
            'likes': np.array(likes)
        }

        np.save(f'clip_embeddings/image_embeddings_{index}.npy', data)

        image_embeddings = []
        text_embeddings = []
        likes = []