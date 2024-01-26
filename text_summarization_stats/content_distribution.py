from typing import List, Tuple
from .utils import sentence_ranks

def content_distribution(
    selected_sentences: List[str],
    A: List[str]
) -> float:
    """
    Computes the content distribution score as defined in the paper https://arxiv.org/pdf/2309.04269.pdf
    """
    ranks = sentence_ranks(selected_sentences, A)
    return sum(ranks) / len(ranks)
