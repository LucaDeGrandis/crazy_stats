from typing import List
import copy
import evaluate
rouge = evaluate.load('rouge')
import numpy as np

def sentence_alignment(
    A: List[str],  # the source sentences
    S: List[str]  # the summary sentences
):
    """
    Performs sentence alignment as defined in the paper https://aclanthology.org/P18-1061.pdf
    """
    alignment = []
    for S_sentence in S:
        aligned = []
        remaining = copy.deepcopy(A)
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