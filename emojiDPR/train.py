from argparse import ArgumentParser
from trainer import train
import os


def get_args():
    arg = ArgumentParser()
    arg.add_argument("--model_path", type=str, default="bert-base-uncased")
    arg.add_argument("--tokenizer_path", type=str, default="bert-base-uncased")
    arg.add_argument("--train_data_path", type=str, required=True)
    arg.add_argument("--valid_data_path", type=str, default=None)
    arg.add_argument("--wandb_project", type=str, required=True)
    arg.add_argument("--batch_size", type=int, default=32)
    arg.add_argument("--max_length", type=int, default=128)
    arg.add_argument("--device", type=str, default="0")
    arg.add_argument("--max_epochs", type=int, default=30)
    

    config = arg.parse_args()
    return config


def main(config):
    train(config)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = get_args()
    main(config)