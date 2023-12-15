import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from PIL import Image
import PIL
import requests
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
import torch
import pandas as pd
from io import BytesIO
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import pickle
from clip_video_encode import clip_video_encode
import urllib.request

tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')

import os

def generate_image_tweet_embeddings(image_data):
    image_model = SentenceTransformer('clip-ViT-B-32')
    text_model = SentenceTransformer('all-mpnet-base-v2-10-1M')

    image_tweet_likes = image_data['likes'].values.tolist()
    image_tweet_ids = image_data['id'].values.tolist()
    image_tweet_content = image_data['content'].values.tolist()
    image_tweet_companies = image_data['inferred company'].values.tolist()

    image_tweet_content_new = []
    for x, y in zip(image_tweet_companies, image_tweet_content):
        image_tweet_content_new.append(x + ' : ' + y)

    image_tweet_text_embeddings = text_model.encode(image_tweet_content_new)

    with open('image_tweet_embed/text.pkl', "wb") as fOut:
        pickle.dump({'ids': image_tweet_ids, 'embeddings': image_tweet_text_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    with open('image_tweet_embed/likes.pkl', "wb") as fOut:
        pickle.dump({'ids': image_tweet_ids, 'embeddings': image_tweet_likes}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    image_embeddings = []
    image_ids = []

    batch_images = []
    batch_image_ids = []

    for index, row in tqdm(image_data.iterrows(), total=len(image_data)):
        image_url = row['media'].split("'")[3]
        response = requests.get(image_url).content

        try:
            image = Image.open(BytesIO(response))
            batch_images.append(image)
            batch_image_ids.append(row['id'])
        except:
            continue

        if len(batch_image_ids) == 64:
            print('encoding')

            batch_image_emb = image_model.encode(batch_images)
            image_embeddings.extend(batch_image_emb)
            image_ids.extend(batch_image_ids)

            batch_images = []
            batch_image_ids = []

        if len(image_ids) == 64 * 10:
            print('saving')
        
            with open(f'image_tweet_embed/image_{index+1}.pkl', "wb") as fOut:
                pickle.dump({'ids': image_ids, 'embeddings': image_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

            image_embeddings = []
            image_ids = []

def generate_video_tweet_embeddings(video_data):
    text_model = SentenceTransformer('all-mpnet-base-v2-10-1M')

    video_tweet_likes = video_data['likes'].values.tolist()
    video_tweet_ids = video_data['id'].values.tolist()
    video_tweet_content = video_data['content'].values.tolist()
    video_tweet_companies = video_data['inferred company'].values.tolist()

    video_tweet_content_new = []
    for x, y in zip(video_tweet_companies, video_tweet_content):
        video_tweet_content_new.append(x + ' : ' + y)

    video_tweet_text_embeddings = text_model.encode(video_tweet_content_new)

    with open('video_tweet_embed/text.pkl', "wb") as fOut:
        pickle.dump({'ids': video_tweet_ids, 'embeddings': video_tweet_text_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    with open('video_tweet_embed/likes.pkl', "wb") as fOut:
        pickle.dump({'ids': video_tweet_ids, 'embeddings': video_tweet_likes}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


    for index, row in tqdm(video_data.iterrows(), total=len(video_data)):
        video_url = row['media'].split("'")[5]
        urllib.request.urlretrieve(video_url, 'video.mp4') 
        clip_video_encode('video.mp4', f'video_tweet_embed/video_{index+1}.pkl', 10)

def generate_gif_tweet_embeddings(gif_data):
    text_model = SentenceTransformer('all-mpnet-base-v2-10-1M')

    gif_tweet_likes = gif_data['likes'].values.tolist()
    gif_tweet_ids = gif_data['id'].values.tolist()
    gif_tweet_content = gif_data['content'].values.tolist()
    gif_tweet_companies = gif_data['inferred company'].values.tolist()

    gif_tweet_content_new = []
    for x, y in zip(gif_tweet_companies, gif_tweet_content):
        gif_tweet_content_new.append(x + ' : ' + y)

    gif_tweet_text_embeddings = text_model.encode(gif_tweet_content_new)

    with open('gif_tweet_embed/text.pkl', "wb") as fOut:
        pickle.dump({'ids': gif_tweet_ids, 'embeddings': gif_tweet_text_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    with open('gif_tweet_embed/likes.pkl', "wb") as fOut:
        pickle.dump({'ids': gif_tweet_ids, 'embeddings': gif_tweet_likes}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


    for index, row in tqdm(gif_data.iterrows(), total=len(gif_data)):
        gif_url = row['media'].split("'")[5]
        urllib.request.urlretrieve(gif_url, 'gif.mp4') 
        clip_video_encode('gif.mp4', f'gif_tweet_embed/gif_{index+1}.pkl', 10)

if __name__ == '__main__':
    data = pd.read_csv("../data/content_simulation_train.csv")

    image_data = data[data['media'].str.contains("Photo")]
    video_data = data[data['media'].str.contains("Video")]
    gif_data = data[data['media'].str.contains("Gif")]

    print(f'len image data: {len(image_data)}, len video data: {len(video_data)}, len gif data: {len(gif_data)}')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    generate_image_tweet_embeddings(image_data)
    generate_video_tweet_embeddings(video_data)
    generate_gif_tweet_embeddings(gif_data)
