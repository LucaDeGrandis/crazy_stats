from typing import List

from transformers import BertForSequenceClassification, BertTokenizer
from ..tokenizers.nltk_tokenizer import NLTKTokenizer
from tqdm import tqdm
import torch


def compute_FactCC(
    input_documents: List[str],
    summaries: List[str],
    model_name_or_path: str = 'manueldeprada/FactCC',
    batch_size: int = 8,
) -> float:
    """ Computes FactCC for a set of input data

    Args:
        :param input_data: List of input data

    Returns:
        The float value corresponging to the FactCC score

    """
    assert len(summaries) == len(input_documents)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create the model and the tokenizer
    tokenizer = BertTokenizer.from_pretrained('manueldeprada/FactCC')
    model = BertForSequenceClassification.from_pretrained(model_name_or_path)
    model.to(device)

    # Use a sentence tokenizer to separate sentences of the system summaries
    sent_tokenizer = NLTKTokenizer(**{'tokenizer_name': 'sent_tokenize'})
    new_documents = []
    new_summaries = []
    for _doc, _summ in zip(input_documents, summaries):
        for _sent in sent_tokenizer(_summ):
            new_documents.append(_doc)
            new_summaries.append(_sent)

    # Run batched evaluation
    predictions = []
    for i in tqdm(range(0, len(new_summaries), batch_size)):
        batch_docs = new_documents[i:min([i+batch_size, len(new_summaries)])]
        batch_summs = new_summaries[i:min([i+batch_size, len(new_summaries)])]
        input_dict = tokenizer(batch_docs, batch_summs, max_length=512, padding='max_length', truncation='only_first', return_tensors='pt')
        input_dict.to(device)
        with torch.no_grad():
            logits = model(**input_dict).logits
            preds = logits.argmax(dim=1).tolist()
        predictions.extend([{
            'input': batch_docs[i],
            'summary': batch_summs[i],
            'label': preds[i]
        } for i in range(len(preds))])

    factcc_score = sum([abs(1-p['label']) for p in predictions]) / len(predictions)

    return factcc_score, predictions
