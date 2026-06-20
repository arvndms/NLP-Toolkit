from models.model_loader import ModelLoader

class NERExtractor:
    def __init__(self):
        self.pipeline = ModelLoader.get_ner()

    def predict(self, text):
        raw = self.pipeline(text)

        entities = []

        for item in raw:
            entities.append({
                "Entity": item["word"],
                "Type": item["entity_group"],
                "Confidence": round(float(item["score"]),3)
            })

        return entities