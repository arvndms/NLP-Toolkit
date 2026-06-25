from transformers import pipeline 
from backend.config import SENTIMENT_MODEL, SUMMARY_MODEL,NER_MODEL,QA_MODEL, DEVICE
import logging 

logger = logging.getLogger(__name__)

class ModelLoader:
    _sentiment = None
    _summary = None 
    _ner = None
    _qa = None

    @staticmethod
    def get_sentiment():
        if ModelLoader._sentiment is None:
            logger.info("Loading sentiment model")
            ModelLoader._sentiment = pipeline(
                "sentiment-analysis",
                model=SENTIMENT_MODEL,
                device=DEVICE
            )
        return ModelLoader._sentiment
    
    @staticmethod
    def get_summary():
        if ModelLoader._summary is None:
            logger.info("Loading Summarizer")
            ModelLoader._summary = pipeline(
                "summarization",
                model=SUMMARY_MODEL,
                device=DEVICE
            )
        return ModelLoader._summary

    @staticmethod
    def get_ner():
        logger.info("NER Model Loaded")
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
        logger.info("QA Model Loaded")
        if ModelLoader._qa is None:
            ModelLoader._qa = pipeline(
                "question-answering",
                model=QA_MODEL,
                device=DEVICE
               
           )
        return ModelLoader._qa