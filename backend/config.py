import torch

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SUMMARY_MODEL = "t5-small"
NER_MODEL = "dslim/bert-base-NER"
QA_MODEL = "deepset/roberta-base-squad2"

DEVICE = 0 if torch.cuda.is_available() else -1