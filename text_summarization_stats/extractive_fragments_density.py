from typing import List


def compute_extractive_fragments_density(
    fragments_tokens: List[List[str]],
    summary_tokens: List[str],
) -> float:
    """Compute the coverage of extractive fragments in a summary.

    Args:
        :param fragments: The list of tokenized extractive fragments.
        :param summary: The tokenized summary.

    Returns:
        float: The coverage of extractive fragments in the summary.
    """
    return sum([0] + [len(fragment)**2 for fragment in fragments_tokens]) / len(summary_tokens)
