from typing import List


def compute_entity_density(
        unique_entities: List[str],
        tokens: List[str],
) -> float:
    """ Calculate the entity density of a text.

    Args:
        :unique_entities: a list of unique entities
        :tokens: a list of tokens

    Return:
        :return: a float representing the entity density

    Example usage:
        from text_summarization_stats.entity_finder import EntityFinder
        entity_finder = EntityFinder()
        from text_summarization_stats.tokenizers import NLTKTokenizer
        tokenizer = NLTKTokenizer()

        example_text = "The concept for Kingdom Hearts originated from a discussion between Shinji Hashimoto and Hironobu Sakaguchi"
        example_text_tokenized = tokenizer(summary)
        example_text_entities, _ = entity_finder(input_document)
        unique_entities = list(set(example_text_entities))

        entity_density = compute_entity_density(unique_entities, example_text_tokenized)
        print(density)
        > 0.15
    """
    assert isinstance(unique_entities, list)
    assert len(unique_entities) == len(set(unique_entities)), \
        'There are some duplicated in unique_entities'
    assert isinstance(tokens, list)

    return len(unique_entities) / len(tokens)
