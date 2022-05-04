from typing import List

def ctakes_tok(s: str) -> List[str]:
    # to deal with weird null hangers on
    return [token for token in s.split() if token]

