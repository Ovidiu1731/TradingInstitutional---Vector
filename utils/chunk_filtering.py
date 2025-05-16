# utils/chunk_filtering.py
import re
from typing import List, Dict, Any, Optional
import math

def clean_text(text: str) -> str:
    """Remove timestamps like [00:00:00] and normalize whitespace"""
    # Strip out anything in square brackets (timestamps, noisy markers)
    no_timestamps = re.sub(r"\[.*?\]", "", text)
    # Collapse whitespace
    return re.sub(r"\s+", " ", no_timestamps).strip().lower()

def keyword_overlap_score(text: str, query: str) -> float:
    """Compute a simple Jaccard overlap on cleaned text"""
    tokenize = lambda s: set(re.findall(r"\w+", s.lower()))
    words_text = tokenize(text)
    words_query = tokenize(query)
    
    if not words_query or not words_text:
        return 0.0
    
    overlap = len(words_text & words_query)
    union = len(words_text | words_query)
    
    # Basic Jaccard similarity
    base_score = overlap / union if union > 0 else 0.0
    
    # Boost score if there are many overlapping words
    if overlap > 3:
        base_score *= 1.2
    
    # Boost score if text contains rarer query words
    query_words = list(words_query)
    rare_word_matches = sum(1 for w in query_words if len(w) > 7 and w in words_text)
    if rare_word_matches > 0:
        base_score *= (1.0 + 0.1 * rare_word_matches)
    
    return base_score

def compute_relevance_score(chunk: str, query: str, expanded_query: str) -> Dict[str, float]:
    """
    Compute multiple relevance scores for a chunk against original and expanded queries
    
    Args:
        chunk: Text chunk to score
        query: Original user query
        expanded_query: Query with expansion terms
        
    Returns:
        Dictionary with various relevance scores
    """
    # Clean the texts
    clean_chunk = clean_text(chunk)
    clean_query = clean_text(query)
    clean_expanded = clean_text(expanded_query)
    
    # Compute base overlap score with original query
    base_score = keyword_overlap_score(clean_chunk, clean_query)
    
    # Compute score with expanded query
    expanded_score = keyword_overlap_score(clean_chunk, clean_expanded)
    
    # Check for key terms presence (exact phrases)
    key_phrases = [
        "market structure shift", "lichiditate", "one gap setup", "two gap setup",
        "setup", "strategie", "tranzacționare", "one simple gap", "fair value gap"
    ]
    
    phrase_match_score = 0.0
    for phrase in key_phrases:
        if phrase in clean_chunk and phrase in clean_expanded:
            phrase_match_score += 0.15  # Boost for each matching key phrase
    
    # Check if chunk appears to be a summary
    is_summary = "rezumat" in clean_chunk[:100].lower() or clean_chunk.startswith("lectia")
    summary_boost = 0.3 if is_summary else 0.0
    
    # Compute final combined score
    combined_score = max(base_score, expanded_score) + phrase_match_score + summary_boost
    
    # Cap at 1.0
    final_score = min(combined_score, 1.0)
    
    return {
        "base_score": base_score,
        "expanded_score": expanded_score,
        "phrase_match_score": phrase_match_score,
        "summary_boost": summary_boost,
        "final_score": final_score
    }

def filter_and_rank_chunks(chunks: List[str], query: str, expanded_query: str, 
                           min_score: float = 0.1, max_chunks: int = 5) -> List[str]:
    """
    Filter and rank chunks based on relevance to query
    
    Args:
        chunks: List of text chunks to filter and rank
        query: Original user query
        expanded_query: Query with expansion terms
        min_score: Minimum relevance score to keep (default: 0.1)
        max_chunks: Maximum number of chunks to return (default: 5)
        
    Returns:
        List of chunks sorted by relevance
    """

    # Deduplicate the input chunks first
    unique_chunks = []
    seen_content = set()
    
    for chunk in chunks:
        # Create a simple hash of the content (first 200 chars)
        content_hash = chunk[:200].strip()
        if content_hash not in seen_content:
            unique_chunks.append(chunk)
            seen_content.add(content_hash)

    # Special handling for summaries: Detect if this is a specific type of resource
    resource_patterns = {
        "books": ["carte", "cărți", "recomandate", "trading", "autor"],
        "setup": ["setup", "gap", "og", "tg", "tcg", "3g", "slg", "mg"],
        "psychology": ["psihologie", "mindset", "psihologic", "emoții"],
    }

    query_lower = query.lower()
    is_resource_query = False
    resource_type = None

    for res_type, patterns in resource_patterns.items():
        if any(pattern in query_lower for pattern in patterns):
            is_resource_query = True
            resource_type = res_type
            break

    # For resource queries, prioritize keeping complete summaries
    if is_resource_query:
        # Find summaries related to this resource type
        summaries = []
        for chunk in unique_chunks:
            chunk_lower = chunk.lower()
            # Check if it's a relevant summary
            is_summary = ("rezumat" in chunk_lower[:100] or chunk.strip().lower().startswith("lectia"))
            is_relevant_summary = any(pattern in chunk_lower for pattern in resource_patterns.get(resource_type, []))
            
            if is_summary and is_relevant_summary:
                summaries.append(chunk)
        
        # If we found relevant summaries, prioritize them with a higher limit
        if summaries:
            # For book-related queries, keep all summary chunks to get complete recommendations
            if resource_type == "books":
                # Return all relevant summary chunks, up to 8 max
                return summaries[:8] if max_chunks > 0 else summaries

    # If not a special case, proceed with standard filtering
    scored_chunks = []
    regular_summaries = []
    regular_chunks = []

    for chunk in unique_chunks:  # Use the deduplicated chunks
        # Check if it's a summary
        is_summary = "rezumat" in chunk.lower()[:100] or chunk.strip().lower().startswith("lectia")
        if is_summary:
            regular_summaries.append(chunk)
        else:
            regular_chunks.append(chunk)
    
    # Score regular chunks
    for chunk in regular_chunks:
        scores = compute_relevance_score(chunk, query, expanded_query)
        if scores["final_score"] >= min_score:
            scored_chunks.append((chunk, scores["final_score"]))
    
    # Sort by score (descending)
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Get top chunks
    top_chunks = [chunk for chunk, _ in scored_chunks[:max_chunks]]
    
    # Combine summaries and top chunks
    result = regular_summaries + top_chunks
    
    # Limit final result if needed
    if max_chunks > 0 and len(result) > max_chunks:
        return result[:max_chunks]
    
    return result