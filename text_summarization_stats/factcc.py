from typing import List

from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
import torch


def compute_FactCC(
    input_documents: List[str],
    summaries: List[str],
    batch_size: int = 8,
) -> float:
    """ Computes FactCC for a set of input data 

    Args:
        :param input_data: List of input data

    Returns:
        The float value corresponging to the FactCC score

    """
    assert len(summaries)==len(input_documents)

    # Create the model and the tokenizer
    model_name = 'manueldeprada/FactCC'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    # Create the input data
    input_dict = tokenizer(input_documents, summaries, max_length=512, padding='max_length', truncation='only_first', return_tensors='pt')
    with torch.no_grad():
        logits = model(**input_dict).logits
        preds = logits.argmax(dim=1).tolist()

    # Run batched evaluation
    predictions = []
    for i in tqdm(range(0, len(summaries), batch_size)):
        batch_docs = input_documents[i:min([i+batch_size, len(summaries)])]
        batch_summs = summaries[i:min([i+batch_size, len(summaries)])]
        input_dict = tokenizer(batch_docs, batch_summs, max_length=512, padding='max_length', truncation='only_first', return_tensors='pt')
        with torch.no_grad():
            logits = model(**input_dict).logits
            preds = logits.argmax(dim=1).tolist()
        predictions.extend(preds)

    return predictions
