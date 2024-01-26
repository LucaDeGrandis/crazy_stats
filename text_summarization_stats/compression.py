from typing import List

def compression(
        summary: List[str],
        reference: List[str],
    ) -> float:
    """
    Compression algorithm for strings as defined in https://aclanthology.org/N18-1065.pdf.
    """

    return len(summary) / len(reference)
