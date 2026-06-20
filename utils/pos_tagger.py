import spacy

nlp = spacy.load("en_core_web_sm")


class POSTagger:

    def __init__(self):
        self.nlp = nlp

    def predict(self, text):

        doc = self.nlp(text)

        result = []

        for token in doc:
            result.append({
                "word": token.text,
                "pos": token.pos_
            })

        return result