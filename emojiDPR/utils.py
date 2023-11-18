import pandas as pd

def replace_text(text):
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace(",", " [SEP]")
    text = text.replace("'", "")

    return text
