from typing import List, Tuple

def Fusion(
    alignment: List[Tuple[str, List[str]]],  # the alignment
) -> float:
    """
    Computes the fusion score as defined in the paper https://arxiv.org/pdf/2309.04269.pdf
    """
    lengths = [len(x[1]) for x in alignment]

    return sum(lengths) / len(lengths)
