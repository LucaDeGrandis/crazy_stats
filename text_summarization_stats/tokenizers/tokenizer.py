from typing import List, Union
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    def __init__(self) -> None:
        """
        Initializes a Tokenizer object.

        Args:
            tokenizer_name (Optional[str]): The name of the tokenizer to use. Defaults to 'identity' if None.
        """
        self.create_tokenizer()

    @abstractmethod
    def create_tokenizer(self) -> None:
        """
        Creates a tokenizer based on the specified tokenizer name.
        """
        pass

    def __call__(
            self,
            text: str
        ) -> Union[List[str], List[List[str]]]:
        """
        Tokenizes the given text using the tokenizer.

        Args:
            text_blocks (List[text_block]): The list of text blocks to tokenize.

        Returns:
            List[str]: The list of tokenized sentences.
        """
        assert isinstance(text, str)
        tokenizer_output = self.tokenizer(text)
        call_output = self.postprocess_tokenization(tokenizer_output)

        return call_output

    @abstractmethod
    def postprocess_tokenization(
            self,
            text: Union[str, List[str], List[List[str]]]
        ) -> Union[str, List[str]]:
        """
        Creates a tokenizer based on the specified tokenizer name.
        """
        pass