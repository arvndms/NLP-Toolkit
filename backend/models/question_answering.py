from backend.models.model_loader import ModelLoader
import logging

logger = logging.getLogger(__name__)

class QuestionAnswering:
    def __init__(self):
        self.pipeline = ModelLoader.get_qa()

    def predict(self, question, context):
        logger.info("QA Model Running")
        return self.pipeline(
            question=question,
            context=context
        )