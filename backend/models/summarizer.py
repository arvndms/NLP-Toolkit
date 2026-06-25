from backend.models.model_loader import ModelLoader
import logging
logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self):
        self.pipeline = ModelLoader.get_summary()

    def summarize(self, text):
        logger.info("Running summarizer model")
        return self.pipeline(
            text,
            max_length=250,
            min_length=30,
            do_sample=False
        )