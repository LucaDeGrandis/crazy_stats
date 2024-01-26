from text_summarization_stats.tokenizers.tokenizer import Tokenizer
from typing import List, Union


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

    def postprocess_tokenization(
            self,
            text: Union[str, List[str], List[List[str]]]
        ) -> Union[str, List[str]]:
        """
        Creates a tokenizer based on the specified tokenizer name.
        """
        return text