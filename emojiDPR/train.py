from argparse import ArgumentParser


def get_args():
    arg = ArgumentParser()
    arg.add_argument("--model_path", type=str, default="bert-base-uncased")


    config = arg.parse_args()
    return config


def main(config):
    pass


if __name__ == "__main__":
    config = get_args()
    main(config)