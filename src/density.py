from typing import List
from .extractive_tokens import greedy_match
from .utils import square_len

def density(
        A: List[str],
        S: List[str]
    ) -> float:
    """
    Density algorithm for strings as defined in https://aclanthology.org/N18-1065.pdf.
    """

    extractive_fragments = greedy_match(A, S)
    fragments_len = sum(list(map(square_len, extractive_fragments)))

    return fragments_len / len(S)
