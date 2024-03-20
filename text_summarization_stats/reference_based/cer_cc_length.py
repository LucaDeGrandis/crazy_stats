from typing import List, Dict
from collections import defaultdict
from ..tokenizers.nltk_tokenizer import NLTKTokenizer


def get_length_value(
    input_string: str,
    tokenizer,
) -> int:
    """Returns the length of the input_string with respect to the chosen tokenizer.

    Args:
        input_string (str): The input string to be tokenized.
        tokenizer: The tokenizer object used to tokenize the input string.

    Returns:
        int: The length of the tokenized input string.

    Raises:
        AssertionError: If the input_string is not of type str or the tokenized_input is not of type list.
    """
    assert isinstance(input_string, str)
    tokenized_input = tokenizer(input_string)
    assert isinstance(tokenized_input, list)

    return len(tokenized_input)


def cer_intra(
    dict_gen: Dict[str, List[int]],
    dict_ref: Dict[str, List[int]],
) -> float:
    """Computes the intra CER average.

    Args:
        dict_gen (Dict[str, List[int]]): A dictionary containing the generated text for each key.
        dict_ref (Dict[str, List[int]]): A dictionary containing the reference text for each key.

    Returns:
        float: The average intra CER (Character Error Rate) value.

    Raises:
        AssertionError: If dict_gen or dict_ref is not a dictionary.
        AssertionError: If any key in dict_gen has an empty list as its value.
        AssertionError: If any key in dict_ref has an empty list as its value.

    """
    assert isinstance(dict_gen, dict)
    assert isinstance(dict_ref, dict)
    for key, item in dict_gen.items():
        assert item != [], f"Empty list for key {key} in dict_gen"
    for key, item in dict_ref.items():
        assert item != [], f"Empty list for key {key} in dict_ref"

    diffs = defaultdict(list)
    res = {
        'abs': None,
        'keys': {},
    }

    # Compute differences by length class
    for key in dict_gen:
        for _gen, _ref in zip(dict_gen[key], dict_ref[key]):
            diffs[key].append(abs(_ref - _gen) / _ref)
    for key, item in diffs.items():
        res['keys'][key] = sum(item) / len(item)

    # Compute differences without length class
    all_diffs = []
    for key, item in diffs.items():
        all_diffs.extend(item)
    res['abs'] = sum(all_diffs) / len(all_diffs)

    return res


def compute_cer_length(
    generations: List[str],
    references: List[str],
    lengths: List[str],
) -> float:
    """Computes the CER length.

    Args:
        generations (List[str]): A list of generated texts.
        references (List[str]): A list of reference texts.
        lengths (List[str]): A list of length values corresponding to each text.

    Returns:
        float: The computed CER length.

    Raises:
        AssertionError: If the input lists are not of the same length.

    """
    assert isinstance(generations, list)
    assert isinstance(references, list)
    assert isinstance(lengths, list)
    assert len(generations) == len(references)
    assert len(generations) == len(lengths)

    tokenizer = NLTKTokenizer()

    gen_lengths = defaultdict(list)
    ref_lengths = defaultdict(list)

    for _gen, _ref, _len in zip(generations, references, lengths):
        gen_lengths[_len].append(get_length_value(_gen, tokenizer))
        ref_lengths[_len].append(get_length_value(_ref, tokenizer))

    return cer_intra(gen_lengths, ref_lengths)


def call_cc(
    pre_sample: str,
    cur_sample: str,
    classes_dict: Dict[str, int],
    tokenizer,
) -> float:
    """Computes CC for a single generation-reference pair.
    """
    pre_len = classes_dict[pre_sample['len']]
    cur_len = classes_dict[cur_sample['len']]
    pre_score = get_length_value(pre_sample, tokenizer)
    cur_score = get_length_value(cur_sample, tokenizer)
    return (pre_score - cur_score) / (pre_len - cur_len)


def compute_cc_length(
    generations: List[str],
    references: List[str],
    lengths: List[str],
    metadata: List[Dict[str, str]],
    classes_dict: Dict[str, int],
) -> float:
    """Computes CC length score.
    """
    assert len(generations) == len(references)
    assert len(generations) == len(lengths)
    assert len(generations) == len(metadata)
    length_classes = set(lengths)
    for _class in length_classes:
        assert _class in classes_dict

    tokenizer = NLTKTokenizer()

    len_ccs = []

    for i, (_gen, _ref, _len, _met) in enumerate(zip(
        generations, references, lengths, metadata
    )):
        if i == 0:
            continue
        if _met != metadata[i-1]:
            continue
        if _len == lengths[i-1]:
            continue
        len_ccs.append(call_cc(
            {
                'gen': generations[i-1],
                'ref': references[i-1],
                'len': lengths[i-1],
            },
            {
                'gen': _gen,
                'ref': _ref,
                'len': _len,
            },
            classes_dict,
            tokenizer,
        ))

    return sum(len_ccs) / len(len_ccs)
