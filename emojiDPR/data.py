import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class TrainingDataset(Dataset):
    def __init__(self, text, emoji_index, emoji_embeddings) -> None:
        self.input_ids = text.input_ids
        self.attention_mask = text.attention_mask
        self.emoji_index = torch.LongTensor(emoji_index)
        self.emoji_embeddings = emoji_embeddings

    
    def __getitem__(self, index):
        emoji_representation = self.emoji_embeddings[self.emoji_index[index]]
        return self.input_ids[index], self.attention_mask[index], emoji_representation


    def __len__(self) -> int:
        return self.emoji_index.shape[0]


def load_data(file):
    if isinstance(file, str):
        file_type = file.split(".")[-1]
        if file_type == "xlsx":
            df = pd.read_excel(file)
            
        elif file_type == "csv":
            df = pd.read_csv(file)
            
        elif file_type == "tsv":
            df = pd.read_csv(file, sep="\t")
            
        else:
            raise TypeError("file must be a excel or csv file")
            
    elif isinstance(file, pd.DataFrame):
        df = file
        
    else:
        raise TypeError("file must be a string or a pandas DataFrame")
    
    return df


def data_pipeline(file, tokenizer, emoji_embeddings, config):
    df = load_data(file)
    text_data = list(df["emoji_names"].values)
    emoji_data = list(df["idx"].values)

    tokenizer_output = tokenizer(text_data, padding=True, max_length=config.max_length, return_tensors = "pt")

    dataset = TrainingDataset(tokenizer_output, emoji_data, emoji_embeddings)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    return dataloader