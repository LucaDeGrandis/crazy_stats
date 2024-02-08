from typing import List
from blanc import BlancHelp
import torch


def compute_BLANC(
    input_documents: List[str],
    summaries: List[str],
    inference_batch_size: int = 128,
) -> float:
    """ Computes BLANC score for a sample of input documents and their corresponding summaries.

    Args:
        :param input_documents: List of input documents.
        :param summaries: List of summaries.

    Returns:
        The BLANC score.

    """
    assert len(summaries) == len(input_documents)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create the model
    blanc_help = BlancHelp(device=device, inference_batch_size=inference_batch_size)

    # Compute the BLANC score
    scores, _ = blanc_help.eval_pairs(input_documents, summaries)
    blanc_score = sum(scores) / len(scores)

    return blanc_score, scores
