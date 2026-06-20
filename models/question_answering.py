from models.model_loader import ModelLoader

class QuestionAnswering:
    def __init__(self):
        self.pipeline = ModelLoader.get_qa()

    def predict(self, question, context):
        return self.pipeline(
            question=question,
            context=context
        )