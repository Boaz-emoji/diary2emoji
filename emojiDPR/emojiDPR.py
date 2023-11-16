from typing import Any
import torch
import lightning as pl
from transformers import AutoTokenizer, AutoModel

class EmojiDPR(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(config.model_path)
        self.config = config
        self.save_hyperparameters()


    def forward(self, input_ids, attention_mask):
        return self.text_encoder(input_ids, attention_mask)
    

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, emoji_representation = batch

        # get text representation by encoder model
        text_representation = self.text_encoder(input_ids, attention_mask)
        sim_matrix = torch.matmul(text_representation, emoji_representation.T)
        loss = torch.nn.functional.nll_loss(sim_matrix, torch.arange(sim_matrix.shape[0]))

        self.log("train_loss", loss)
        return loss
    

    def predict_step(self, batch):
        text, emoji = batch