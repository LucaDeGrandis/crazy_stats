from typing import List
from itertools import chain
from .utils import sentence_ranks

def content_distribution(
    selected_sentences: List[List(str)],
    A: List[str]
) -> float:
    """
    Computes the content distribution score as defined in the paper https://arxiv.org/pdf/2309.04269.pdf
    """
    sentences_list = list(chain.from_iterable(selected_sentences))
    ranks = sentence_ranks(sentences_list, A)
    return sum(ranks) / len(ranks)
