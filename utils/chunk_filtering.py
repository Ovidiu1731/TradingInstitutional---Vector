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
    
    query_lower = query.lower()
    query_normalized = query_lower.replace("ă", "a").replace("â", "a").replace("î", "i").replace("ș", "s").replace("ț", "t")
    
    # Enhanced resource pattern detection with weighted scoring
    resource_patterns = {
        "books": {
            "primary": ["carte", "carti", "cărți", "recomandari", "recomandate", "citit", "lectura", "books"],
            "secondary": ["autor", "trading", "douglas", "schwager", "zen trader"],
            "exact_phrases": ["carti recomandate", "ce carti", "trading in the zone"]
        },
        "setup": {
            "primary": ["setup", "gap", "tcg", "og", "tg"],
            "secondary": ["3g", "slg", "mg", "pattern"],
            "exact_phrases": ["cum fac setup", "ce este setup"]
        },
        "psychology": {
            "primary": ["psihologie", "mindset", "psihologic", "emotii", "emoții"],
            "secondary": ["disciplina", "frică", "frica", "emotie", "emoție"],
            "exact_phrases": ["control emotional", "mindset trading"]
        }
    }
    
    # Score each resource type for this query
    resource_scores = {}
    for res_type, patterns in resource_patterns.items():
        score = 0
        # Primary matches (most important)
        for word in patterns["primary"]:
            if word in query_lower or word in query_normalized:
                score += 10
        
        # Secondary matches (supporting evidence)
        for word in patterns["secondary"]:
            if word in query_lower or word in query_normalized:
                score += 3
        
        # Exact phrase matches (highest confidence)
        for phrase in patterns["exact_phrases"]:
            if phrase in query_lower or phrase in query_normalized:
                score += 25
        
        resource_scores[res_type] = score
    
    # Determine if this is a resource query and which type
    is_resource_query = any(score > 0 for score in resource_scores.values())
    resource_type = max(resource_scores, key=resource_scores.get) if is_resource_query else None
    resource_confidence = resource_scores.get(resource_type, 0) if resource_type else 0
    
    logging.info(f"Query '{query}' resource detection: {resource_type if is_resource_query else 'general'} (confidence: {resource_confidence})")
    
    # Handle high-confidence resource queries with special processing
    if is_resource_query and resource_confidence >= 10:
        # Find chunks related to this resource type with both summaries and relevant content
        resource_chunks = []
        summary_chunks = []
        
        for chunk in unique_chunks:
            chunk_lower = chunk.lower().replace("ă", "a").replace("â", "a").replace("î", "i").replace("ș", "s").replace("ț", "t")
            
            # Check if it's a summary
            is_summary = ("rezumat" in chunk_lower[:100] or chunk_lower.strip().startswith("lectia"))
            
            # Score this chunk's relevance to the resource type
            chunk_resource_score = 0
            for primary_term in resource_patterns[resource_type]["primary"]:
                if primary_term in chunk_lower:
                    chunk_resource_score += 5
            
            for secondary_term in resource_patterns[resource_type]["secondary"]:
                if secondary_term in chunk_lower:
                    chunk_resource_score += 2
            
            for exact_phrase in resource_patterns[resource_type]["exact_phrases"]:
                if exact_phrase in chunk_lower:
                    chunk_resource_score += 15
            
            # Add high-relevance chunks to the appropriate list
            if chunk_resource_score >= 5:
                if is_summary:
                    summary_chunks.append((chunk, chunk_resource_score))
                else:
                    resource_chunks.append((chunk, chunk_resource_score))
        
        # Sort both lists by relevance score
        summary_chunks.sort(key=lambda x: x[1], reverse=True)
        resource_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Special handling for books - ensure we get ALL relevant information
        if resource_type == "books":
            # For books, keep ALL relevant summary chunks to avoid missing titles
            logging.info(f"Book query detected, found {len(summary_chunks)} relevant summary chunks")
            
            # Return all summaries first, then relevant non-summary chunks, up to a max total
            result_chunks = [chunk for chunk, _ in summary_chunks]
            
            # Add top-scoring non-summary chunks if needed
            remaining_slots = max(0, 10 - len(result_chunks))
            result_chunks.extend([chunk for chunk, _ in resource_chunks[:remaining_slots]])
            
            logging.info(f"Returning {len(result_chunks)} chunks for book query")
            return result_chunks
        else:
            # For other resource types, combine summaries and relevant chunks
            combined_chunks = [chunk for chunk, _ in summary_chunks] + [chunk for chunk, _ in resource_chunks]
            
            # Limit to max_chunks or a reasonable maximum
            max_to_return = 8 if max_chunks < 8 else max_chunks
            logging.info(f"Returning {min(len(combined_chunks), max_to_return)} chunks for {resource_type} query")
            return combined_chunks[:max_to_return]
    
    # For low-confidence resource queries or general queries, use improved standard filtering
    regular_summaries = []
    high_relevance_chunks = []
    other_chunks = []
    
    # First pass: categorize chunks
    for chunk in unique_chunks:
        # Check if it's a summary
        chunk_lower = chunk.lower()
        is_summary = "rezumat" in chunk_lower[:100] or chunk_lower.strip().startswith("lectia")
        
        if is_summary:
            regular_summaries.append(chunk)
        else:
            # Score the chunk
            scores = compute_relevance_score(chunk, query, expanded_query)
            
            # Categorize based on relevance score
            if scores["final_score"] >= 0.5:  # High relevance threshold
                high_relevance_chunks.append((chunk, scores["final_score"]))
            elif scores["final_score"] >= min_score:
                other_chunks.append((chunk, scores["final_score"]))
    
    # Sort chunks by relevance
    high_relevance_chunks.sort(key=lambda x: x[1], reverse=True)
    other_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Combine in priority order: summaries, high relevance, other relevant
    result = regular_summaries + [chunk for chunk, _ in high_relevance_chunks] + [chunk for chunk, _ in other_chunks]
    
    # Limit final result if needed
    if max_chunks > 0 and len(result) > max_chunks:
        return result[:max_chunks]
    
    return result