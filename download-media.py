import requests
import pandas as pd
from tqdm import tqdm

data = pd.read_csv("content_simulation_train.csv")

image_data = data[data['media'].str.contains("Photo")]
video_data = data[data['media'].str.contains("Video")]
gif_data = data[data['media'].str.contains("Gif")]

for index, row in tqdm(image_data.iterrows(), total=len(image_data)):
    try:
        image_url = row['media'].split("'")[3]
        response = requests.get(image_url).content
        file = open(f'./media/images/{index}.jpg', 'wb')
        file.write(response)
        file.close()
    except:
        pass

