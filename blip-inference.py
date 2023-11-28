from PIL import Image
import PIL
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Model
import torch
import pandas as pd
from io import BytesIO
from tqdm import tqdm

tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TRANSFORMERS_CACHE'] = '/scratch/apoorva_v.iitr/huggingface/hub'
os.environ['PYTORCH_TRANSFORMERS_CACHE'] = '/scratch/apoorva_v.iitr/huggingface/hub'
os.environ['CURL_CA_BUNDLE'] = ''

data = pd.read_csv("content_simulation_train.csv")

image_data = data[data['media'].str.contains("Photo")]
video_data = data[data['media'].str.contains("Video")]
gif_data = data[data['media'].str.contains("Gif")]

print(f'len image data: {len(image_data)}, len video data: {len(video_data)}, len gif data: {len(gif_data)}')

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)

def image_inference(image_url):
    response = requests.get(image_url)

    try:
        image = Image.open(BytesIO(response.content))
    except PIL.UnidentifiedImageError:
        return None

    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return generated_text

image_data["Generated Text"] = image_data["media"].progress_apply(lambda x: image_inference(x.split("'")[3]))
image_data.to_csv("image_data.csv", index=False)
