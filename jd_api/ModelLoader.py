from pathlib import Path
import spacy

import sys

sys.path.append(str(Path(__file__).parent.parent))


class ModelLoader:
    """Singleton class used to load and provide ML SPACY Model"""

    __instance = None

    @staticmethod
    def getInstance():
        """Static access method."""
        if ModelLoader.__instance == None:
            ModelLoader()
        return ModelLoader.__instance

    def __init__(self):
        """Virtually private constructor."""
        self.SPACY_CKPTS = str(Path(__file__).parent) + "/jd_model/model-best"
        self.NLP_NER = spacy.load(self.SPACY_CKPTS)
        if ModelLoader.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            ModelLoader.__instance = self

    def get_model(self):
        return self.NLP_NER
