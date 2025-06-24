import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "trading-lessons"))

def get_embedding(text):
    """Get embedding for text using OpenAI's API."""
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None

def extract_lesson_info_from_path(path):
    """Extract chapter and lesson numbers from path metadata."""
    chapter_match = re.search(r'Capitolul_(\d+)', path)
    lesson_match = re.search(r'lectia_(\d+)', path)
    
    chapter = chapter_match.group(1) if chapter_match else None
    lesson = lesson_match.group(1) if lesson_match else None
    
    return chapter, lesson

def retrieve_lesson_content(query, chapter=None, lesson=None, top_k=5):
    """Retrieve lesson content based on semantic search and metadata filtering."""
    logger.info(f"Processing query: {query}")
    logger.info(f"Filters - Chapter: {chapter}, Lesson: {lesson}")
    
    try:
        # Expand query with related terms
        expanded_query = expand_query_terms(query)
        search_query = f"{query} {expanded_query}".strip()
        logger.info(f"Expanded query: {search_query}")
        
        # Get embedding for the expanded query
        query_embedding = get_embedding(search_query)
        if not query_embedding:
            logger.error("Failed to get query embedding")
            return []
        
        # Build filter conditions
        filter_conditions = {}
        if chapter:
            filter_conditions["chapter"] = f"Capitolul {chapter}"
        if lesson:
            filter_conditions["lesson_number"] = str(lesson).zfill(2)
        
        # Query Pinecone with semantic search
        results = index.query(
            vector=query_embedding,
            filter=filter_conditions if filter_conditions else None,
            top_k=top_k * 2,  # Get more results initially for better filtering
            include_metadata=True
        )
        
        logger.info(f"Raw results count: {len(results['matches'])}")
        
        # Process and filter results
        processed_results = []
        seen_content = set()  # Changed from seen_paths to seen_content
        
        # Dynamic threshold based on query type
        min_score = get_dynamic_threshold(query)
        logger.info(f"Using minimum score threshold: {min_score}")
        
        for i, match in enumerate(results["matches"]):
            logger.info(f"Raw result {i+1}: score={match.score:.4f}, has_metadata={bool(match.metadata)}")
            
            if not match.metadata:
                logger.info(f"  Skipping result {i+1}: no metadata")
                continue
                
            if match.score < min_score:
                logger.info(f"  Skipping result {i+1}: score {match.score:.4f} below threshold {min_score}")
                continue
                
            # Extract chapter and lesson from path if not in metadata
            path = match.metadata.get("path", "")
            if not match.metadata.get("chapter") and path:
                chapter_num, lesson_num = extract_lesson_info_from_path(path)
                if chapter_num:
                    match.metadata["chapter"] = f"Capitolul {chapter_num}"
                if lesson_num:
                    match.metadata["lesson_number"] = lesson_num
            
            # Better deduplication: check content similarity instead of path
            text_content = match.metadata.get("text", "")
            content_fingerprint = text_content[:100].strip().lower()  # First 100 chars as fingerprint
            
            if content_fingerprint in seen_content:
                logger.info(f"  Skipping result {i+1}: similar content already included")
                continue
            seen_content.add(content_fingerprint)
            
            # Quality scoring for better ranking
            quality_score = calculate_content_quality(text_content)
            combined_score = match.score * 0.7 + quality_score * 0.3  # Weighted combination
            
            # Add to processed results
            processed_results.append({
                "score": match.score,
                "combined_score": combined_score,
                "chapter": match.metadata.get("chapter", "Unknown"),
                "lesson": match.metadata.get("lesson_number", "Unknown"),
                "text": text_content
            })
            
            logger.info(f"✅ Added result {len(processed_results)}: score={match.score:.4f}, quality={quality_score:.2f}")
            logger.info(f"  Chapter: {processed_results[-1]['chapter']}")
            logger.info(f"  Lesson: {processed_results[-1]['lesson']}")
            logger.info(f"  Text preview: {processed_results[-1]['text'][:200]}...")
        
        # Sort by combined score (similarity + quality) and take top_k
        processed_results.sort(key=lambda x: x["combined_score"], reverse=True)
        final_results = processed_results[:top_k]
        
        logger.info(f"Final results count: {len(final_results)}")
        return final_results
        
    except Exception as e:
        logger.error(f"Error in retrieve_lesson_content: {e}")
        return []

def get_dynamic_threshold(query):
    """Get dynamic threshold based on query type and complexity."""
    query_lower = query.lower()
    
    # FIXED: Much lower thresholds based on diagnostic data
    # With ada-002, we get scores 0.80-0.95 for good matches
    if any(term in query_lower for term in ["mss", "market structure", "structura"]):
        return 0.75  # MSS queries - expect high scores with ada-002
    elif any(term in query_lower for term in ["liq", "lichidit", "lichiditate"]):
        return 0.80  # Liquidity queries - should get very high scores
    elif any(term in query_lower for term in ["setup", "fvg", "gap"]):
        return 0.75  # Technical setups
    else:
        return 0.70  # Default threshold - much higher with ada-002

def calculate_content_quality(text):
    """Calculate content quality score to prioritize complete explanations."""
    if not text or len(text.strip()) < 50:
        return 0.0
    
    score = 0.5  # Base score
    text_lower = text.lower()
    
    # Positive indicators (complete explanations)
    if any(indicator in text for indicator in ["###", "**", "1.", "2.", "3.", "- "]):
        score += 0.3  # Structured content
    
    if any(word in text_lower for word in ["definiție", "definire", "exemplu", "spre exemplu", "principalele", "tipuri"]):
        score += 0.2  # Educational content
        
    if len(text) > 300:
        score += 0.2  # Comprehensive content
    
    # Negative indicators (conversational snippets)
    if any(filler in text_lower for filler in ["uite", "bam", "deci", "aia", "să zicem", "nu știu"]):
        score -= 0.2  # Conversational filler
        
    if text.count("...") > 2:
        score -= 0.1  # Truncated content
        
    if any(platform in text_lower for platform in ["platforme", "pe anumite", "nu ți este permis"]):
        score -= 0.1  # Platform-specific rather than educational
    
    return max(0.0, min(1.0, score))  # Clamp between 0 and 1

def expand_query_terms(query):
    """Expand query with related trading terms."""
    query_lower = query.lower()
    expansions = []
    
    # MSS related expansions
    if "mss" in query_lower or "market structure" in query_lower or "structura" in query_lower:
        expansions.extend([
            "market structure shift", "structura de piață", "schimbare structură",
            "pivot", "higher low", "lower high", "agresiv", "normal"
        ])
    
    # Aggressive MSS specific
    if "agresiv" in query_lower:
        expansions.extend([
            "aggressive", "pivot", "candle", "bearish", "bullish", 
            "definire", "identificare", "recunoaștere"
        ])
    
    # Liquidity expansions
    if any(term in query_lower for term in ["liq", "lichidit"]):
        expansions.extend([
            "liquidity", "HOD", "LOD", "majoră", "locală", "minoră",
            "zone", "levels", "sweep"
        ])
    
    # Sessions expansions - CRITICAL for comprehensive session information
    if any(term in query_lower for term in ["sesiuni", "session", "tranzactionare", "trading hours", "ore"]):
        expansions.extend([
            "sesiuni tranzactionare", "londra", "new york", "tokyo", "sydney",
            "program zilnic", "structurarea programului", "intervale orare",
            "10:15", "12:00", "16:15", "16:45", "19:00", "22:00",
            "intervalele recomandate", "toate intervalele", "program trading",
            "organizarea zilei", "structurarea zilei", "cum tranzactionez"
        ])
    
    return " ".join(expansions)

def test_retrieval():
    """Test the retrieval logic with example queries."""
    # Test case 1: Query about liquidity types
    print("\nTest Case 1: Query about liquidity types")
    results = retrieve_lesson_content(
        "What are the types of liquidity?",
        chapter=6,
        lesson=2
    )
    
    if results:
        print("\nResults found:")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['score']:.4f}):")
            print(f"Chapter: {result['chapter']}")
            print(f"Lesson: {result['lesson']}")
            print(f"Text preview: {result['text'][:200]}...")
    else:
        print("No results found")
    
    # Test case 2: Query about Fair Value Gap
    print("\nTest Case 2: Query about Fair Value Gap")
    results = retrieve_lesson_content("Explain Fair Value Gap")
    
    if results:
        print("\nResults found:")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['score']:.4f}):")
            print(f"Chapter: {result['chapter']}")
            print(f"Lesson: {result['lesson']}")
            print(f"Text preview: {result['text'][:200]}...")
    else:
        print("No results found")

if __name__ == "__main__":
    test_retrieval() 