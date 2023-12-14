import os
import sys
import re
import cv2
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from io import BytesIO
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    Blip2Model,
    Blip2VisionModel,
    Blip2QFormerModel,
    Blip2QFormerConfig,
    OPTConfig,
    Blip2Config,
    BitsAndBytesConfig,
)

sys.path.insert(1, "./task2/blip-inference/BLIP")
from models.blip import blip_decoder
import torch
import argparse

tqdm.pandas()

import warnings

warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_CACHE"] = "./huggingface/hub"
os.environ["PYTORCH_TRANSFORMERS_CACHE"] = "./huggingface/hub"
os.environ["CURL_CA_BUNDLE"] = ""


parser = argparse.ArgumentParser(prog="blip-inference")

data = pd.read_excel(args.data_path)

image_data = data[data["media"].str.contains("Photo")]
video_data = data[data["media"].str.contains("Video")]
gif_data = data[data["media"].str.contains("Gif")]

parser.add_argument(
    "data-path", type=str, default="./data/content_simulation_train.xlsx"
)
parser.add_argument("username-data", type=str, default="./data/username_data.csv")
parser.add_argument("start_index", type=int, default=0)
parser.add_argument("end_index", type=int, default=len(data))

args = parser.parse_args()

username_data = pd.read_excel(args.username_data)


print(
    f"len image data: {len(image_data)}, len video data: {len(video_data)}, len gif data: {len(gif_data)}"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    load_in_8bit=True,
    device_map="auto",
    max_memory={0: "10GB"},
    torch_dtype=torch.float16,
    use_safetensors=True,
)


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
)

## We are using the parts of Blip2ForConditionalGeneration for inference i.e. vision_model
## and qformer.

blip2_model = (
    Blip2ForConditionalGeneration.from_pretrained(  # For better performance on Images
        "Salesforce/blip2-opt-2.7b",
        device_map="auto",
        max_memory={0: "10GB"},
        quantization_config=quantization_config,
        force_download=False,
    )
)


def image_inference(image_url):
    response = requests.get(image_url)
    try:
        image = Image.open(BytesIO(response.content))
    except PIL.UnidentifiedImageError:
        return None

    # The output of 4bit precision seems good and the GPU memory required for the setup is
    # around 10GB with CPU memory max around 12-13 GB
    # The torch data type used here -> torch.float16
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    ):
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

        ### The above steps are done to reduce and get just the outputs of qformers
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

    return generated_text


# For performance on Videos we use Blip1 to get low instance time and high performance
image_size = 384
blip1_model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"
blip1_model = blip_decoder(
    pretrained=blip1_model_url, image_size=image_size, vit="base"
)
blip1_model.eval()
blip1_model = blip1_model.to(device)


def load_demo_image(image_size, device, pil_image):
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    image = transform(pil_image).unsqueeze(0).to(device)
    return image


# Function to extract audio from video
def _extract_audio(video_path, audio_path):
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)


# Function to generate captions using GPT-3.5
def _generate_captions(audio_path):
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

    with open(audio_path, "r") as audio_file:
        audio_text = audio_file.read()

    captions = generator(audio_text, max_length=100, num_return_sequences=1)
    generated_caption = captions[0]["generated_text"]

    return generated_caption


def process_video(index, row, model, device, start_index, end_index):
    vid = row["media"]
    url_pattern = re.compile(r"https?://\S+")

    matches = url_pattern.findall(vid)
    for m in matches:
        # print(m.find('mp4'))
        if m.find("mp4") != -1:
            video_url = m
            break
    try:
        response = requests.get(video_url, stream=True)

        video_filename = "downloaded_video.mp4"
        print("This function works here:")
        with open(video_filename, "wb") as video_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    video_file.write(chunk)
    except requests.exceptions.ContentDecodingError:
        print(index, "Not Done")
        return "", ""
    cap = cv2.VideoCapture(video_filename)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, total_frames - 1, 10, dtype=int)

    frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    video_captions = ""

    for idx, frame in enumerate(frames):
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = load_demo_image(
            image_size=image_size, device=device, pil_image=pil_frame
        )

        caption = model.generate(
            image, sample=False, num_beams=1, max_length=20, min_length=5
        )
        video_captions = video_captions + " " + caption[0] + ":"

    audio_filename = "audio.wav"

    _extract_audio(video_filename, audio_filename)

    audio_captions = _generate_captions(audio_filename)

    video_captions = str(video_captions) + str(audio_captions)

    return video_url, video_captions


def process_gif(index, row, model, device, start_index, end_index):
    vid = row["media"]
    url_pattern = re.compile(r"https?://\S+")

    matches = url_pattern.findall(vid)
    image_urls = []
    gif_url = ""

    for m in matches:
        if m.find("mp4") != -1:
            gif_url = m
            break
        if m.find("jpg") != -1:
            image_urls.append(m)

    gif_filename = "downloaded_gif.mp4"
    gif_captions = ""

    if gif_url != "":
        gif_url = gif_url[:-2]
        try:
            with open(gif_filename, "wb+") as gif_file:
                response = requests.get(gif_url, stream=True)
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        gif_file.write(chunk)
            cap = cv2.VideoCapture(gif_filename)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_indices = np.linspace(0, total_frames - 1, 4, dtype=int)

            frames = []

            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            cap.release()

            for idx, frame in enumerate(frames):
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = load_demo_image(
                    image_size=image_size, device=device, pil_image=pil_frame
                )
                caption = model.generate(
                    image, sample=False, num_beams=1, max_length=20, min_length=5
                )
                gif_captions = gif_captions + " " + caption[0] + ":"

        except requests.exceptions.ContentDecodingError:
            gif_captions = ""

    image_captions = ""

    for image_url in image_urls:
        generated_text = image_inference(image_url[:-2])
        image_captions = str(image_captions) + str(generated_text)

    gif_captions = str(gif_captions) + str(image_captions)

    return gif_url, gif_captions


# Comany data
def company_username_inference(username):
    """Use data obtained from Twitter API and wikidata to do inference."""
    userdata = pd.read_csv(args.username_data)
    return df[df["username"] == username]


results = []
columns = data.columns
columns_dict = {
    "likes": "Expected Likes",
    "username": "Username",
    "inferred company": "Company Name",
    "date": "Date Time",
    "content": "Tweet Content",
}

for index, row in data.iterrows():
    if index < args.start_index or index >= args.end_index:
        continue
    data_item = {}

    if "Photo" in data.loc[index, "media"]:
        media_url = data.iloc[index]["media"]
        image_url = media_url.split("'")[3]
        generated_text = image_inference(image_url)
        data_item["content"] = [generated_text] + data_item.get("content", [])

    elif "Video" in data.loc[index, "media"]:
        video_url, video_captions = process_video(
            index, row, blip1_model, device, start_index, end_index
        )
        data_item["content"] = [video_captions] + data_item.get("content", [])

    elif "Gif" in data.loc[index, "media"]:
        print("gif")
        gif_url, gif_captions = process_video(
            index, row, blip1_model, device, start_index, end_index
        )
        data_item["content"] = [gif_captions] + data_item.get("content", [])

    twitter_data = company_username_inference(data.iloc[index]["username"])

    data_item["content"].append(f"Expected Likes: {data.iloc[index]['likes']}")
    data_item["content"].append(f"Username: {data.iloc[index]['username']}")
    data_item["content"].append(f"Company Name: {data.iloc[index]['inferred company']}")
    data_item["content"].append(f"Date Time: {data.iloc[index]['date']}")
    data_item["content"].append(f"Location: {twitter_data['location'].item()}")
    data_item["content"].append(f"Description: {twitter_data['description'].item()}")

    data_item["content"] = [item for item in data_item["content"] if item is not None]

    data_item["content"] = ", ".join(data_item["content"])

    data_item[
        "instruction"
    ] = "You are a content generator, you have to generate the tweet content based on the expected likes and other additional information."

    if "content" in columns:
        data_item["output"] = data.iloc[index]["content"]

    results.append(data_item)


file_path = ".data/finetuning.json"

# Save the list of dictionaries to a JSON file
with open(file_path, "w") as json_file:
    json.dump(results, json_file, indent=2)
