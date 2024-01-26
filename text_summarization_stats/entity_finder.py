import spacy


class EntityFinder:
    def __init__(self, **kwargs):
        for key, item in kwargs.items():
            setattr(self, key, item)
        self.initialize_entity_finder()

    def initialize_entity_finder(self):
        self.ner_model = spacy.load("en_core_web_sm")

    def __call__(self, text):
        assert isinstance(text, str)
        ner_model_output = self.find_entities(text)
        entities = [entity.text for entity in ner_model_output.ents]
        entity_classes = [entity.label_ for entity in ner_model_output.ents]
        return entities, entity_classes

    def run_model(self, text):
        ner_model_output = self.ner_model(text)
        return ner_model_output
