from text_summarization_stats.tokenizers.tokenizer import Tokenizer
from typing import List


def identity_tokenizer(text):
    """ Returns the input text.    
    """
    assert isinstance(text, str)
    return text


class IdentityTokenizer(Tokenizer):
    """ Returns the input text as is.
    """
    def __init__(self, **kwargs) -> None:
        for key, item in kwargs.items():
            setattr(self, key, item)
        super().__init__()

    def create_tokenizer(self) -> None:
        """
        Creates the tokenizer
        """
        self.tokenizer = identity_tokenizer
