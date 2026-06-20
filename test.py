from models.ner import NERExtractor

ner = NERExtractor()

print(
    ner.predict(
        "Elon Musk founded SpaceX in California."
    )
)