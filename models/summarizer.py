from models.model_loader import ModelLoader

class Summarizer:
    def __init__(self):
        self.pipeline = ModelLoader.get_summary()

    def summarize(self, text):
        return self.pipeline(
            text,
            max_length=150,
            min_length=30,
            do_sample=False
        )