from typing import List
from .extractive_tokens import greedy_match

def Coverage(
        A: List[str],
        S: List[str]
    ) -> float:
    """
    Coverage algorithm for strings as defined in https://aclanthology.org/N18-1065.pdf.
    """

    extractive_fragments = greedy_match(A, S)
    fragments_len = sum(list(map(len, extractive_fragments)))

    return fragments_len / len(S)
