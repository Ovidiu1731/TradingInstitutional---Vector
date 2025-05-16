# utils/query_expansion.py

# Trading abbreviations and their expansions
DOMAIN_SYNONYMS = {
    # Market structure terminology
    "mss": "market structure shift schimbare structură",
    "bos": "break of structure rupere structură",
    "choch": "change of character schimbare caracter",
    
    # Gap setups
    "og": "one gap setup",
    "osg": "one simple gap setup",
    "slg": "second leg gap setup",
    "tg": "two gap setup",
    "tcg": "two consecutive gaps setup",
    "3g": "three gaps setup",
    "3cg": "three consecutive gaps setup",
    "mg": "multiple gaps setup",
    
    # Market concepts
    "fvg": "fair value gap spațiu gol",
    "hod": "high of day maximul zilei",
    "lod": "low of day minimul zilei",
    "imb": "imbalance",
    "ob": "order block",
    
    # Liquidity terminology
    "liq": "lichiditate liquidity",
    
    # Trading directions
    "long": "cumpărare bullish poziție long",
    "short": "vânzare bearish poziție short",
    
    # Trade management
    "sl": "stop loss",
    "tp": "take profit",
    "be": "break even",
    "rr": "risk to reward raport risc-randament",
    
    # Timeframes
    "htf": "higher time frame",
    "ltf": "lower time frame",
    "tf": "time frame interval timp",
}

# Concepts and related terms
CONCEPT_TERMS = {
    "setup": ["setup tipuri", "pattern", "strategie"],
    "structur": ["structură", "mss", "trend"],
    "lichiditate": ["liq", "liquidity", "zone de lichiditate"],
    "strategia": ["metodă", "abordare", "plan", "tehnică"],
    "candel": ["lumânare", "candle", "japanese candlestick"],
    "gap": ["fvg", "gap-uri", "spațiu netranzacționat"],
}

def expand_query(query: str) -> str:
    """
    Expand query with trading-specific terminology to improve vector search results.
    
    Args:
        query: Original user query
        
    Returns:
        String with expanded terms added to improve search
    """
    # Lowercase for matching
    query_lower = query.lower()
    expansions = []
    
    # Add abbreviation expansions
    for abbr, expansion in DOMAIN_SYNONYMS.items():
        # Look for whole word matches to avoid partial matches
        if f" {abbr} " in f" {query_lower} " or f"{abbr} " in f"{query_lower} " or f" {abbr}" in f" {query_lower}":
            # Skip if expansion already in query
            if not any(term in query_lower for term in expansion.split()):
                expansions.append(expansion)
    
    # Add concept expansions
    for concept, terms in CONCEPT_TERMS.items():
        if concept in query_lower:
            # Add related terms not already in query
            for term in terms:
                if term not in query_lower and term not in expansions:
                    expansions.append(term)
    
    # Add special handling for specific query types
    if "cum " in query_lower and any(word in query_lower for word in ["fac", "execut", "tranzacționez"]):
        expansions.append("execuție intrare entry strategie")
    
    if "diferența" in query_lower and "între" in query_lower:
        expansions.append("comparație diferență versus")
    
    # Join all expansions
    return " ".join(expansions)
