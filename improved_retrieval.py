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
            model="text-embedding-3-small",
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
        # Get embedding for the query
        query_embedding = get_embedding(query)
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
        seen_paths = set()
        
        # Lower confidence threshold for liquidity queries
        min_score = 0.60 if "liq" in query.lower() or "lichidit" in query.lower() else 0.65
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
            
            # Skip if we've seen this path before
            if path in seen_paths:
                logger.info(f"  Skipping result {i+1}: duplicate path {path}")
                continue
            seen_paths.add(path)
            
            # Add to processed results
            processed_results.append({
                "score": match.score,
                "chapter": match.metadata.get("chapter", "Unknown"),
                "lesson": match.metadata.get("lesson_number", "Unknown"),
                "text": match.metadata.get("text", "")
            })
            
            logger.info(f"✅ Added result {len(processed_results)}: score={match.score:.4f}")
            logger.info(f"  Chapter: {processed_results[-1]['chapter']}")
            logger.info(f"  Lesson: {processed_results[-1]['lesson']}")
            logger.info(f"  Text preview: {processed_results[-1]['text'][:200]}...")
        
        # Sort by score and take top_k
        processed_results.sort(key=lambda x: x["score"], reverse=True)
        final_results = processed_results[:top_k]
        
        logger.info(f"Final results count: {len(final_results)}")
        return final_results
        
    except Exception as e:
        logger.error(f"Error in retrieve_lesson_content: {e}")
        return []

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