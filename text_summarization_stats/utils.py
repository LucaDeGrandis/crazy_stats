from typing import List, Dict, Any, Union
import matplotlib.pyplot as plt
import json
import os


def square_len(
    S: List[str]
) -> float:
    """
    Computes the square of the length of the input list
    """
    return len(S) ** 2


def sentence_ranks(
    selected_sentences: List[str],
    A: List[str],
    normalize: bool = False,
) -> List[int]:
    """
    Computes the ranks of the selected sentences in the original document
    """
    ranks = []
    for sentence in selected_sentences:
        ranks.append(A.index(sentence) + 1)
    if normalize:
        ranks = [rank / len(A) for rank in ranks]
    return ranks


def load_json_file(
	filepath: str
) -> List[Any]:
    """Load a json into a list
    *arguments*
    *filepath* path to the file
    """
    with open(filepath, 'r', encoding='utf8') as reader:
        json_data = json.load(reader)
    return json_data


def write_json_file(
    filepath: str,
    input_dict: Dict[str, Any],
    overwrite: bool =False
) -> None:
    """Write a dictionary into a json
    *arguments*
    *filepath* path to save the file into
    *input_dict* dictionary to be saved in the json file
    *overwrite* whether to force overwriting a file.
        Default is false so you don't delete an existing file.
    """
    if not overwrite:
        assert not os.path.exists(filepath)
    with open(filepath, 'w', encoding='utf8') as writer:
        json.dump(input_dict, writer, indent=4, ensure_ascii=False)


def load_jsonl_file(
	filepath: str
) -> List[Dict[str, Any]]:
    """Load a json into a list
    *arguments*
    *filepath* path to the file
    """
    data = []
    with open(filepath, "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line.strip()))
    return data


def write_jsonl_file(
    filepath: str,
    input_list: List[Any],
    mode: str ='a+',
    overwrite: bool =False
) -> None:
    """Write a list into a jsonl
    *arguments*
    *filepath* path to save the file into
    *input_list* list to be saved in the json file, must be made of json iterable objects
    *overwrite* whether to force overwriting a file.
        When set to False you will append the new items to an existing jsonl file (if the file already exists).
    """
    if overwrite:
        try:
            os.remove(filepath)
        except:
            pass
    with open(filepath, mode, encoding='utf8') as writer:
        for line in input_list:
            writer.write(json.dumps(line) + '\n')
