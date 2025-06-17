# Save as check_lesson.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import json

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "trading-lessons"))

def check_liquidity_content():
    print("\nChecking for liquidity content in Chapter 11, Lesson 2...")
    
    # Check standard format
    lesson_id = "capitol_11_lectia_02"
    
    try:
        # First check by ID
        results = index.query(
            vector=[0.0] * 1536,  # Dummy vector
            filter={"lesson_id": lesson_id},
            top_k=5,
            include_metadata=True
        )
        
        if results["matches"]:
            print(f"✅ Found vectors using lesson ID: {lesson_id}")
            for i, match in enumerate(results["matches"]):
                print(f"\nMatch {i+1}:")
                print(f"Score: {match.score}")
                if match.metadata:
                    print(f"Text preview: {match.metadata.get('text', '')[:200]}...")
        else:
            print(f"❌ No vectors found using lesson ID: {lesson_id}")
        
        # Now check by chapter and lesson number
        results = index.query(
            vector=[0.0] * 1536,
            filter={
                "chapter": "Capitolul 11",
                "lesson_number": "02"
            },
            top_k=5,
            include_metadata=True
        )
        
        if results["matches"]:
            print(f"\n✅ Found vectors using chapter/lesson filter")
            for i, match in enumerate(results["matches"]):
                print(f"\nMatch {i+1}:")
                print(f"Score: {match.score}")
                if match.metadata:
                    print(f"Text preview: {match.metadata.get('text', '')[:200]}...")
        else:
            print(f"❌ No vectors found using chapter/lesson filter")
            
    except Exception as e:
        print(f"Error checking liquidity content: {e}")

if __name__ == "__main__":
    check_liquidity_content()