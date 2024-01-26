from text_summarization_stats.tokenizers.tokenizer import Tokenizer
from typing import List, Union
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize


import nltk
nltk.download('punkt')


nltk_tokenizers_map = {
    'word_tokenize': word_tokenize,
    'wordpunct_tokenize': wordpunct_tokenize,
    'sent_tokenize': sent_tokenize,
}


class NLTKTokenizer(Tokenizer):
    """ Returns the input text tokenized according to the NLTK tokenizer.
    """
    def __init__(self, **kwargs) -> None:
        for key, item in kwargs.items():
            setattr(self, key, item)
        if 'tokenizer_name' in kwargs:
            assert kwargs['tokenizer_name'] in nltk_tokenizers_map.keys()
            self.tokenizer_name = kwargs['tokenizer_name']
        else:
            self.tokenizer_name = 'word_tokenize'
        super().__init__()
    
    def create_tokenizer(self):
        self.tokenizer = nltk_tokenizers_map[self.tokenizer_name]

    def postprocess_tokenization(
            self,
            text: Union[str, List[str], List[List[str]]]
        ) -> Union[str, List[str]]:
        """
        Creates a tokenizer based on the specified tokenizer name.
        """
        return text