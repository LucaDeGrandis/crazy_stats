from typing import List

def greedy_match(
        A: List[str],  # source text
        S: List[str]  # summary
    ) -> List[List[str]]:
    """
    Greedy matchin algorithm for strings
    """
    F = []
    i, j = 0, 0  # i for S, j for A
    while i < len(S):
        f = []
        while j < len(A):
            if S[i] == A[j]:
                i_prime, j_prime = i, j
                while S[i_prime] == A[j_prime]:
                    i_prime += 1
                    j_prime += 1

                    if i_prime >= len(S):
                        break
                    if j_prime >= len(A):
                        break

                if len(f) < (i_prime - i - 1):
                    f = S[i:i_prime]
                j = j_prime
            else:
                j += 1
        i, j = i + max(len(f), 1), 0
        if f:
            F.append(f)
    return F
