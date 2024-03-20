from typing import List


def compute_abstractiveness(
    fragments_tokens: List[List[str]],
) -> float:
    """Compute the abstractiveness of the summary.

    Args:
        :param fragments: The list of tokenized extractive fragments.

    Returns:
        float: The abstractiveness of a summary.
    """
    if fragments_tokens:
        return sum([0] + [len(fragment)**2 for fragment in fragments_tokens]) / len(fragments_tokens)
    else:
        return .0
