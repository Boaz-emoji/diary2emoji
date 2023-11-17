from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch.optim import AdamW
import torch.nn.functional as F
import lightning as pl
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


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
        softmax_score = F.log_softmax(sim_matrix, dim=1)
        loss = F.nll_loss(softmax_score , torch.arange(sim_matrix.shape[0]).to(softmax_score.device), reduction="mean")

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, emoji_representation = batch

        # get text representation by encoder model
        text_representation = self.text_encoder(input_ids, attention_mask)
        sim_matrix = torch.matmul(text_representation, emoji_representation.T)
        softmax_score = F.log_softmax(sim_matrix, dim=1)
        loss = F.nll_loss(softmax_score , torch.arange(sim_matrix.shape[0]).to(softmax_score.device), reduction="mean")

        self.log("valid_loss", loss)
        return loss
    

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optim = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
        return [optim]