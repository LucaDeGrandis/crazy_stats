from typing import List

def compression(
        summary: List[str],
        input_document: List[str],
    ) -> float:
    """
    Compression algorithm for strings as defined in https://aclanthology.org/N18-1065.pdf.
    """

    return len(input_document) / len(summary)
