from models.model_loader import ModelLoader
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")



class SentimentAnalyzer:
    def __init__(self):
        self.pipeline=ModelLoader.get_sentiment()

    def predict(self,text):
        overall =  self.pipeline(text)[0]
        sentences = sent_tokenize(text)

        sentence_results = []

        for sentence in sentences:
            result = self.pipeline(sentence)[0]

            sentence_results.append({
                "Sentence":sentence,
                "Sentiment": result["label"],
                "Score": round(result["score"],3)
            })
        
        return {
            "overall": overall,
            "sentence_wise": sentence_results
        }
        
