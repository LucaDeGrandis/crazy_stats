from typing import List, Tuple


def Fusion(
    alignments: List[Tuple[str, List[str]]],  # the alignment
) -> float:
    """
    Computes the fusion score as defined in the paper https://arxiv.org/pdf/2309.04269.pdf.

    Example usage:

        input_document: str
        summary: str

        # Tokenization
        from text_summarization_stats.tokenizers import NLTKTokenizer
        tokenizer = NLTKTokenizer(**{'tokenizer_name': 'sent_tokenize'})
        summary_Stokenized = tokenizer(summary)
        input_document_Stokenized = tokenizer(input_document)

        # Multiple alignments
        from text_summarization_stats.sentence_alignment import multiple_sentences_alignment
        multiple_alignment = multiple_sentences_alignment(
            summary_Stokenized,
            input_document_Stokenized,
            **{'alignment_type': 'simple'}
        )
    """
    lengths = list(map(len, alignments))

    return sum(lengths) / len(lengths)
