import torch
from torch.utils.data import Dataset
import pandas as pd


class TrainingDataset(Dataset):
    def __init__(self, text, emoji, emoji_embeddings) -> None:
        self.text = text
        self.emoji = emoji
        self.emoji_embeddings = emoji_embeddings

    
    def __getitem__(self, index):
        emoji_representation = self.emoji_embeddings[[self.emoji[index]]]
        return self.text[index], emoji_representation

    

    def __len__(self) -> int:
        return self.text.shape[0]