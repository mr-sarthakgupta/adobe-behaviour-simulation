from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from torch.utils.data import DataLoader
import torch
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)

loss_fn = losses.ContrastiveLoss(model=model)

import random

def create_dataset(sentences, likes, percentage_threshold, max_samples):
    pairs = []
    labels = []
    count_different = 0
    count_similar = 0

    while count_different + count_similar < max_samples:
        i = random.randint(0, len(sentences) - 1)
        j = random.randint(0, len(sentences) - 1)

        percentage_diff = abs(likes[i] - likes[j]) / (max(likes[i], likes[j]) + 1) * 100
        label = 0 if percentage_diff > percentage_threshold else 1

        if label == 0:
            if count_different < max_samples // 2:
                count_different += 1
                pairs.append((sentences[i], sentences[j]))
                labels.append(label)
        else:
            if count_similar < max_samples // 2:
                count_similar += 1
                pairs.append((sentences[i], sentences[j]))
                labels.append(label)

        if count_different >= max_samples // 2 and count_similar >= max_samples // 2:
            break

    print(f'count_different: {count_different}, count_similar: {count_similar}')

    return pairs, labels

data = pd.read_csv("../data/content_simulation_train.csv")

tweet_likes = data['likes'].values.tolist()
tweet_ids = data['id'].values.tolist()
tweet_content = data['content'].values.tolist()
tweet_companies = data['inferred company'].values.tolist()

tweet_content_new = []
for x, y in zip(tweet_companies, tweet_content):
    tweet_content_new.append(x + ' : ' + y)

pairs, labels = create_dataset(tweet_content_new, tweet_likes, 15, 100000)
print('len of dataset', len(pairs))

training_examples = [
    InputExample(texts=pair, label=label) for pair, label in zip(pairs, labels)
]

train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=64)
train_loss = losses.ContrastiveLoss(model=model)

model.fit([(train_dataloader, train_loss)], show_progress_bar=True, epochs=1, warmup_steps=100)
model.save('all-mpnet-base-v2-10-1M')
