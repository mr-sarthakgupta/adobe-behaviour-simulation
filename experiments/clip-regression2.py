# 2. performs regression

# load embeddings from all .npy files in clip_embeddings
# each file is a dict {'image_embeddings': np.array, 'text_embeddings': np.array, 'likes': np.array}
# concat each image_embedding and text_embedding array to get one embedding per sample
# then perform regression on the embeddings and likes using pytorch mlp

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

class RegressionDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)

        return x

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0

    for batch_idx, (embeddings, labels) in enumerate(train_loader):
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(embeddings)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_idx, (embeddings, labels) in enumerate(val_loader):
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            output = model(embeddings)
            loss = criterion(output, labels)

            val_loss += loss.item()
            predictions.extend(output.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    return val_loss / len(val_loader), predictions, actuals

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_embeddings = []
    text_embeddings = []
    likes = []

    for file in tqdm(os.listdir('clip_embeddings')):
        data = np.load(f'clip_embeddings/{file}', allow_pickle=True).item()
        image_embeddings.extend(data['image_embeddings'])
        text_embeddings.extend(data['text_embeddings'])
        likes.extend(data['likes'])

    image_embeddings = np.concatenate(image_embeddings)
    text_embeddings = np.concatenate(text_embeddings)
    likes = np.array(likes)

    print(f'image embeddings shape: {image_embeddings.shape}')
    print(f'text embeddings shape: {text_embeddings.shape}')
    print(f'likes shape: {likes.shape}')
    print(likes)

    image_train, image_val, text_train, text_val, likes_train, likes_val = train_test_split(image_embeddings, text_embeddings, likes, test_size=0.2, random_state=42)

    print(f'image train shape: {image_train.shape}')
    print(f'image val shape: {image_val.shape}')
    print(f'text train shape: {text_train.shape}')
    print(f'text val shape: {text_val.shape}')
    print(f'likes train shape: {likes_train.shape}')
    print(f'likes val shape: {likes_val.shape}')

    image_train = torch.from_numpy(image_train).float()
    image_val = torch.from_numpy(image_val).float()
    text_train = torch.from_numpy(text_train).float()
    text_val = torch.from_numpy(text_val).float()
    likes_train = torch.from_numpy(likes_train).float()
    likes_val = torch.from_numpy(likes_val).float()

    train_dataset = RegressionDataset(torch.cat((image_train, text_train), 1), likes_train)
    val_dataset = RegressionDataset(torch.cat((image_val, text_val), 1), likes_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = RegressionModel(1024, 256, 1).to(device)
    optimizer = Adam(model.parameters(), lr=.2)
    criterion = nn.MSELoss()

    for epoch in range(10):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, predictions, actuals = evaluate(model, val_loader, criterion, device)
        print(f'epoch: {epoch}, train loss: {train_loss}, val loss: {val_loss}')

    print(f'r2 score: {r2_score(actuals, predictions)}')
    print(f'mean squared error: {mean_squared_error(actuals, predictions)}')
    print(f'mean absolute error: {mean_absolute_error(actuals, predictions)}')

if __name__ == '__main__':
    main()
