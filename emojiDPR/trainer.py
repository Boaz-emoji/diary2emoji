import wandb
import torch
import pickle

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
from emojiDPR import EmojiDPR
from data import TrainingDataset, data_pipeline


def train(config):
    devices = None
    accelerator = None
    if config.device == -1:
        accelerator = "cpu"
    else:
        accelerator = "gpu"
        
        temp = config.device.split(",")
        devices = [int(x) for x in temp]
    
    
    tokenizer = BertTokenizerFast.from_pretrained(config.tokenizer_path)
    print("-"*10 + "Tokenizer initialized!" + "-"*10)
    
    model  = EmojiDPR(config=config)
    print("-"*10 + "Model initialized!" + "-"*10)

    with open('./vector_set.pickle', 'rb') as f:
        emoji_embeddings = pickle.load(f)
    

    train_dataloader = data_pipeline(config.train_data_path, tokenizer, emoji_embeddings, config)
    valid_dataloader = None
    if config.valid_data_path is not None:
        valid_dataloader = data_pipeline(config.valid_data_path, tokenizer, emoji_embeddings, config)


    wandb_logger = WandbLogger(project=config.wandb_project, name=f"EmojiDPR-batch_size{config.batch_size}")
    wandb_logger.experiment.config["batch_size"] = config.batch_size
    print("-"*10 + "Wandb Setting Complete!" + "-"*10)
    
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                                          dirpath='./checkpoint',
                                          filename= f"batch_size{config.batch_size}"+'-{valid_loss:.2f}',
                                          save_top_k=1,
                                          save_last=False,
                                          verbose=True,
                                          mode="min")
    

    early_stopping = EarlyStopping(
        monitor='valid_loss', 
        mode='min',
        patience=2,
    )
    
    if config.valid_data_path is None:
        trainer = pl.Trainer(devices=devices,
                         accelerator=accelerator,
                         enable_progress_bar=True,
                         callbacks=[checkpoint_callback],
                         max_epochs=config.max_epochs,
                         num_sanity_val_steps=2,
                         logger=wandb_logger)
    else:
        trainer = pl.Trainer(devices=devices,
                             accelerator=accelerator,
                             enable_progress_bar=True,
                             callbacks=[checkpoint_callback, early_stopping],
                             max_epochs=config.max_epochs,
                             num_sanity_val_steps=2,
                             logger=wandb_logger)
    
    print("-"*10 + "Train Start!" + "-"*10)
    
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    print("-"*10 + "Train Finished!" + "-"*10)
    
    wandb.finish()