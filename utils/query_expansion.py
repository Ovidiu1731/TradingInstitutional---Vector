# utils/query_expansion.py
DOMAIN_SYNONYMS = {
    # extend this dict whenever you notice abbreviations
    "mss": "market structure shift",
    "liq": "liquidity",
    "ob":  "order block",
    "imb": "imbalance",
    "ot":  "optimal trade entry",
}

def expand_query(q: str) -> str:
    """
    Return a space‑separated string of domain‑specific expansions
    found in the user's question.  Example:
    >>> expand_query("Show me MSS and OB")
    'market structure shift order block'
    """
    lower = q.lower()
    expansions = [
        value for key, value in DOMAIN_SYNONYMS.items()
        if key in lower and value not in lower
    ]
    return " ".join(expansions)

