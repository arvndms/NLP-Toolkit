from fastapi import FastAPI,HTTPException
from backend.schemas import TextRequest,QARequest
from backend.models.sentiment import SentimentAnalyzer
from backend.models.summarizer import Summarizer
from backend.models.question_answering import QuestionAnswering
from backend.models.ner import NERExtractor
from backend.logger import logger

app = FastAPI()
# Load the models once when the application starts
sentiment_model = SentimentAnalyzer() 
summarizer_model = Summarizer()
qa_model = QuestionAnswering()
ner_model = NERExtractor()

@app.post("/sentiment")
def sentiment(req:TextRequest):
    try:
        logger.info("Sentiment request received")
        result = sentiment_model.predict(req.text) # Analyze the sentiment of the input text
        logger.info("Sentiment prediction complete")

        return result
    except Exception as e :
        logger.error(f"Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
@app.post("/summary")
def summary(req:TextRequest):
    try:
        logger.info("Summarizer request received")
        result = summarizer_model.summarize(req.text) 
        logger.info("Summarized")

        return result
    except Exception as e:
        logger.error(f"Error{e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@app.post("/QuestionAnswer")
def QnA(req:QARequest):
    try:
        logger.info("QA request received")
        answer = qa_model.predict(req.question,req.context)
        logger.info("Model Answered")
        return answer
    except Exception as e:
        logger.error(f"Error{e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@app.post("/NER")
def NER(req:TextRequest):
    try:    
        logger.info("NER request received")
        result = ner_model.predict(req.text)
        logger.info("NER result completed")
        return result
    except Exception as e:
        logger.error(f"Error{e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

