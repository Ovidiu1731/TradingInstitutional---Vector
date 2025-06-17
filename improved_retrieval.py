import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import json
import re

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "trading-lessons"))

def get_embedding(text):
    """Get embedding for a text using OpenAI's API."""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def extract_lesson_info_from_path(path):
    """Extract chapter and lesson numbers from the path metadata."""
    if not path:
        return None, None
    
    # Try to extract from path format: Capitolul_X/Lectia_Y/...
    chapter_match = re.search(r'Capitolul_(\d+)', path)
    lesson_match = re.search(r'lectia_(\d+)', path)
    
    chapter = chapter_match.group(1) if chapter_match else None
    lesson = lesson_match.group(1) if lesson_match else None
    
    return chapter, lesson

def retrieve_lesson_content(query, chapter=None, lesson=None, top_k=5):
    """
    Retrieve lesson content using semantic search and metadata filtering.
    
    Args:
        query (str): The search query
        chapter (int, optional): Chapter number to filter by
        lesson (int, optional): Lesson number to filter by
        top_k (int): Number of results to return
        
    Returns:
        list: List of matching content with metadata
    """
    # Log the incoming query
    print(f"\nProcessing query: {query}")
    print(f"Filters - Chapter: {chapter}, Lesson: {lesson}")
    
    # Get embedding for the query
    query_embedding = get_embedding(query)
    if not query_embedding:
        print("Failed to get embedding for query")
        return []
    
    # Build filter conditions
    filter_conditions = {}
    
    if chapter is not None:
        filter_conditions["chapter"] = f"Capitolul {chapter}"
    
    if lesson is not None:
        filter_conditions["lesson_number"] = str(lesson).zfill(2)
    
    try:
        # Query Pinecone with semantic search and metadata filters
        results = index.query(
            vector=query_embedding,
            filter=filter_conditions if filter_conditions else None,
            top_k=top_k * 2,  # Get more results initially for filtering
            include_metadata=True
        )
        
        print(f"Raw results count: {len(results.matches)}")
        
        # Process and format results
        formatted_results = []
        seen_paths = set()  # Track unique paths to avoid duplicates
        
        for match in results.matches:
            if not match.metadata:
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
                continue
            seen_paths.add(path)
            
            # Only add results with valid metadata
            if match.metadata.get("text"):
                formatted_results.append({
                    "text": match.metadata["text"],
                    "chapter": match.metadata.get("chapter", "Unknown"),
                    "lesson": match.metadata.get("lesson_number", "Unknown"),
                    "score": match.score
                })
        
        # Sort by score and take top_k
        formatted_results.sort(key=lambda x: x["score"], reverse=True)
        final_results = formatted_results[:top_k]
        
        print(f"Final results count: {len(final_results)}")
        for i, result in enumerate(final_results):
            print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
            print(f"Chapter: {result['chapter']}")
            print(f"Lesson: {result['lesson']}")
            print(f"Text preview: {result['text'][:200]}...")
        
        return final_results
        
    except Exception as e:
        print(f"Error in retrieval: {str(e)}")
        return []

def test_retrieval():
    """Test the retrieval with some example queries."""
    test_cases = [
        {
            "query": "What are the types of liquidity?",
            "chapter": 11,
            "lesson": 2
        },
        {
            "query": "Explain Fair Value Gap",
            "chapter": None,
            "lesson": None
        },
        {
            "query": "What is institutional liquidity?",
            "chapter": None,
            "lesson": None
        }
    ]
    
    for test in test_cases:
        print(f"\nTesting query: {test['query']}")
        print(f"Chapter: {test['chapter']}, Lesson: {test['lesson']}")
        print("-" * 50)
        
        results = retrieve_lesson_content(
            query=test["query"],
            chapter=test["chapter"],
            lesson=test["lesson"]
        )
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['score']:.4f}):")
            print(f"Chapter: {result['chapter']}")
            print(f"Lesson: {result['lesson']}")
            print(f"Text: {result['text'][:200]}...")

if __name__ == "__main__":
    test_retrieval() 