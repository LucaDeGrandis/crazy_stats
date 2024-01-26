from typing import List
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize


nltk_tokenizers_map = {
    'word_tokenize': word_tokenize,
    'wordpunct_tokenize': wordpunct_tokenize,
    'sent_tokenize': sent_tokenize,
}


class NLTKTokenizer():
    """
    Custom NLTK tokenizer.
    """
    def __init__(self, tokenizer_name):
        self.initialize_tokenizer(tokenizer_name)
    
    def initialize_tokenizer(self, tokenizer_name):
        self.tokenizer = nltk_tokenizers_map[tokenizer_name]

    def __call__(self, text):
        return self.tokenizer(text)