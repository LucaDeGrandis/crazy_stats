from typing import List, Dict, Any, Tuple, Union
from openai import OpenAI
import json
import tqdm
import time
import re


def run_openai_geval(
        prompt_template: str,
        instance: Dict[str, str],
        openai_key: str,
        model: str,
) -> List[Dict[str, float]]:
    """ Function to compute G-Eval

    Args
        :param prompt_template: the prompt template that will be used for the LLM
        :param summeval_fp: the dataset, it must be a list of dictionaries with keys "source" and "system_output"
        :param key: the openai API key
        :param model: the name of the model
    
    Return
        a list of dictionaries with OpenAI's responses
    """
    new_json = {}

    source = instance['source']
    system_output = instance['system_output']
    cur_prompt = prompt_template.format(**{
        'Document': source,
        'Summary': system_output,
    })
    while True:
        try:
            client = OpenAI(
                api_key=openai_key,
            )
            _response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": cur_prompt}],
                temperature=2,
                max_tokens=5,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                # logprobs=40,
                n=20
            )
            time.sleep(0.5)

            all_responses = [_response.choices[i].message.content for i in
                                range(len(_response.choices))]
            instance['all_responses'] = all_responses
            new_json = instance
            break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
            else:
                break

    return new_json


def parse_output(output):
    matched = re.search("^ ?([\d\.]+)", output)
    if (matched):
        try:
            score = float(matched.group(1))
        except:
            score = None
    else:
        score = None
    return score


def compute_GEval_score(
    GPT_output: List[str]
) -> float:
    """ Computes the G-Eval score from the textual GPT output

    Args
        :param GPT_output: the raw gpt output

    Return
        the G-Eval score
    """
    all_scores = [parse_output(x) for x in GPT_output]
    return sum(all_scores) / len(all_scores)


def run_GEval(
    prompt_templates: Dict[str, str],
    input_data: Dict[str, str],
    openai_key: str,
    model: str,
) -> Dict[str, float]:
    """ Computes the G-Eval score for a list of prompt templates 

    Args
        :param prompt_templates: a dictionary with the prompt templates.
            The dictionary must have the following keys:
                - "fluency"
                - "relevance"
                - "consistency"
                - "coherence"
        :param input_data: the input data
        :param key: the openai API key
        :param model: the name of the model

    Return
        The G-Eval scores for each prompt template

    """
    geval_responses = {}
    geval_scores = {}
    for key, item in prompt_templates.items():
        geval_responses[key] = run_openai_geval(
            item, input_data, openai_key, model
        )
        relevant_responses = list(filter(lambda x: x is not None, geval_responses[key]))
        geval_scores[key] = compute_GEval_score(
            geval_responses[key]['all_responses']
        )

    return geval_scores, geval_responses
