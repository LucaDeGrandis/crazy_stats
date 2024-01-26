from typing import List

def compression(
        reference: List[str],
        summary: List[str]
    ) -> float:
    """
    Compression algorithm for strings as defined in https://aclanthology.org/N18-1065.pdf.
    """

    return len(summary) / len(reference)
