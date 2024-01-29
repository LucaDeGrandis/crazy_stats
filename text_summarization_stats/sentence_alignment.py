import copy
import evaluate
import numpy as np
from typing import List, Tuple
from itertools import combinations


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

    rouge = evaluate.load('rouge')

    # initialize
    alignment = ''
    ROUGE_score = 1e-6
    k = 1
    aligned_sentences = []
    while True:
        if k > len(document_sentences):
            break
        sentence_combinations = list(combinations(document_sentences, k))
        document_sentences_samples = [[' '.join(combined_sentences)] for combined_sentences in sentence_combinations]
        target_sentences = [target_sentence] * (len(document_sentences_samples))
        ROUGE_scores = rouge.compute(predictions=target_sentences, references=document_sentences_samples, rouge_types=[rouge_type], use_aggregator=False)[rouge_type]
        best_index = np.argmax(ROUGE_scores)
        if ROUGE_scores[best_index] > ROUGE_score:
            ROUGE_score = ROUGE_scores[best_index]
            alignment = document_sentences_samples[best_index][0]
            aligned_sentences = sentence_combinations[best_index]
        elif not exhaustive:
            break
        k += 1

    return ROUGE_score, aligned_sentences


def sentence_alignment_simple(
    target_sentence: str,  # the summary sentences
    document_sentences: List[str],  # the source sentences
    rouge_type: str = 'rouge2',  # rhe ROUGE type to use
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

    rouge = evaluate.load('rouge')

    # initialize
    alignment = ''
    ROUGE_score = 1e-6
    remaining_sentences = copy.deepcopy(document_sentences)
    aligned_sentences = []
    while True:
        document_sentences_samples = [[alignment + ' ' + new_sentence] for new_sentence in remaining_sentences]
        target_sentences = [target_sentence] * (len(document_sentences_samples))
        ROUGE_scores = rouge.compute(predictions=target_sentences, references=document_sentences_samples, rouge_types=[rouge_type], use_aggregator=False)[rouge_type]
        best_index = np.argmax(ROUGE_scores)
        if ROUGE_scores[best_index] > ROUGE_score:
            ROUGE_score = ROUGE_scores[best_index]
            alignment = document_sentences_samples[best_index][0]
            aligned_sentences.append(remaining_sentences.pop(best_index))
        else:
            break
        if not remaining_sentences:
            break

    return ROUGE_score, aligned_sentences


SENTENCE_ALIGNMENT_MAPPING = {
    'simple': sentence_alignment_simple,
    'complete': sentence_alignment_complete,
}


def multiple_sentences_alignment(
    target_sentences: List[str],  # the summary sentences
    document_sentences: List[str],  # the source sentences
    **kwargs
):
    """ Wrapper around the sentence alignment to work on multiple sentences.
    """
    if 'alignment_type' not in kwargs:
        kwargs['alignment_type'] = 'simple'
    if 'rouge_type' not in kwargs:
        kwargs['rouge_type'] = 'rouge2'
    if 'exhaustive' not in kwargs:
        kwargs['exhaustive'] = False
    sentence_aligner = SENTENCE_ALIGNMENT_MAPPING[kwargs['alignment_type']]
    all_alignments = []
    if kwargs['alignment_type'] == 'simple':
        for target_sentence in target_sentences:
            _, alignment = sentence_aligner(target_sentence, document_sentences, rouge_type=kwargs['rouge_type'])
            all_alignments.append(alignment)
    elif kwargs['alignment_type'] == 'complete':
        for target_sentence in target_sentences:
            _, alignment = sentence_aligner(target_sentence, document_sentences, rouge_type=kwargs['rouge_type'], exhaustive=kwargs['exhaustive'])
            all_alignments.append(alignment)

    return all_alignments
