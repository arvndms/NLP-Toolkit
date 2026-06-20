from transformers import pipeline 
from config import SENTIMENT_MODEL, SUMMARY_MODEL,NER_MODEL,QA_MODEL, DEVICE

class ModelLoader:
    _sentiment = None
    _summary = None 
    _ner = None
    _qa = None

    @staticmethod
    def get_sentiment():
        if ModelLoader._sentiment is None:
            ModelLoader._sentiment = pipeline(
                "sentiment-analysis",
                model=SENTIMENT_MODEL,
                device=DEVICE
            )
        return ModelLoader._sentiment
    
    @staticmethod
    def get_summary():
        if ModelLoader._summary is None:
            ModelLoader._summary = pipeline(
                "summarization",
                model=SUMMARY_MODEL,
                device=DEVICE
            )
        return ModelLoader._summary

    @staticmethod
    def get_ner():
        if ModelLoader._ner is None:
            ModelLoader._ner = pipeline(
                "ner",
                model=NER_MODEL,
                device=DEVICE,
                aggregation_strategy="first"

            )
        return ModelLoader._ner
    
    @staticmethod
    def get_qa():
        if ModelLoader._qa is None:
            ModelLoader._qa = pipeline(
                "question-answering",
                model=QA_MODEL,
                device=DEVICE
               
           )
        return ModelLoader._qa