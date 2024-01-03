from typing import List

def Compression(
        A: List[str],
        S: List[str]
    ) -> float:
    """
    Compression algorithm for strings as defined in https://aclanthology.org/N18-1065.pdf.
    """

    return len(S) / len(A)
