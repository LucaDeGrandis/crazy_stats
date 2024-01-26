from typing import List


def greedy_match(
    summary_tokens: List[str],  # summary
    document_tokens: List[str],  # source text
) -> List[List[str]]:
    """
    Greedy matchin algorithm for strings.
    """
    F = []
    i, j = 0, 0  # i for summary, j for document
    while i < len(summary_tokens):
        f = []
        while j < len(document_tokens):
            if summary_tokens[i] == document_tokens[j]:
                i_prime, j_prime = i, j
                while summary_tokens[i_prime] == document_tokens[j_prime]:
                    i_prime += 1
                    j_prime += 1

                    if i_prime >= len(summary_tokens):
                        break
                    if j_prime >= len(document_tokens):
                        break

                if len(f) < (i_prime - i - 1):
                    f = summary_tokens[i:i_prime]
                j = j_prime
            else:
                j += 1
        i, j = i + max(len(f), 1), 0
        if f:
            F.append(f)
    return F
