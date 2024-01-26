from typing import List

def square_len(
        S: List[str]
    ) -> float:
    """
    Computes the square of the length of the input list
    """
    return len(S) ** 2

def sentence_ranks(
    selected_sentences: List[str],
    A: List[str]
) -> List[int]:
    """
    Computes the ranks of the selected sentences in the original document
    """
    ranks = []
    for sentence in selected_sentences:
        ranks.append(A.index(sentence) + 1)
    return ranks
    