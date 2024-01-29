import copy
import evaluate
import numpy as np
from typing import List, Tuple
from itertools import combinations


rouge = evaluate.load('rouge')


def sentence_alignment_complete(
    target_sentence: str,  # the summary sentences
    document_sentences: List[str],  # the source sentences
    rouge_type: str = 'rouge2',  # rhe ROUGE type to use
    exhaustive: bool = False  # whether to perform exhaustive search
) -> Tuple[float, List[str]]:
    """
    Performs sentence alignment for a single sentence as defined in the paper https://aclanthology.org/P18-1061.pdf
    """
    assert rouge_type in ['rouge1', 'rouge2', 'rougeL', 'rougeLS']
    assert document_sentences
    assert isinstance(document_sentences, list)
    assert isinstance(document_sentences[0], str)
    assert isinstance(target_sentence, str)

    if len(document_sentences) == 1:
        return 0.0, document_sentences

    # initialize
    alignment = []
    ROUGE_score = 1e-6
    k = 1
    while True:
        if k > len(document_sentences):
            break
        document_sentences_samples = list(map(list, combinations(document_sentences, k)))
        target_sentences = [target_sentence] * (len(document_sentences_samples))
        ROUGE_scores = rouge.compute(predictions=target_sentences, references=document_sentences_samples, rouge_types=[rouge_type], use_aggregator=False)[rouge_type]
        best_index = np.argmax(ROUGE_scores)
        if ROUGE_scores[best_index] > ROUGE_score:
            ROUGE_score = ROUGE_scores[best_index]
            alignment = list(document_sentences_samples[best_index])
        elif not exhaustive:
            break
        k += 1

    return ROUGE_score, alignment


def sentence_alignment(
    document_sentences: List[str],  # the source sentences
    summary_sentences: List[str]  # the summary sentences
):
    """
    Performs sentence alignment as defined in the paper https://aclanthology.org/P18-1061.pdf
    """
    alignment = []
    for S_sentence in summary_sentences:
        aligned = []
        remaining = copy.deepcopy(document_sentences)
        while remaining:
            rouge_gain = relative_rouge_gain(S_sentence, aligned, remaining)
            max_pos = np.argmax(rouge_gain)
            if rouge_gain[max_pos] <= .0:
                break
            new_aligned = remaining.pop(max_pos)
            aligned.append(new_aligned)

        alignment.append((S_sentence, aligned))

    return alignment


def relative_rouge_gain(
    sentence: str,  # the summary sentence (singular)
    aligned: List[str],  # the list of aligned source sentences
    remaining: List[str]  # the remaining sentences (plural)
):
    """
    Computes relative rouge gain as described in the paper https://aclanthology.org/P18-1061.pdf
    """

    if aligned:
        last_rouge_score = rouge.compute(predictions=[sentence], references=[[aligned]])['rougeL']
    else:
        last_rouge_score = 0.0

    gains = []
    for remaining_sentence in remaining:
        gains.append(rouge.compute(predictions=[sentence], references=[[aligned] + [remaining_sentence]])['rougeL'] - last_rouge_score)

    return gains
