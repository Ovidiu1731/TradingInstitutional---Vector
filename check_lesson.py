# Save as check_lesson.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "trading-lessons"))

# Check for both potential ID formats
def check_lesson(chapter=11, lesson=2):
    print(f"Checking for lesson chapter {chapter}, lesson {lesson}...")
    
    # Check standard format
    standard_id = f"capitol_{chapter}_lectia_{str(lesson).zfill(2)}"
    
    # Check alternate format
    alternate_id = f"11_{chapter}_2_{str(lesson).zfill(2)}"
    
    # Check both formats
    for format_name, lesson_id in [("Standard", standard_id), ("Alternate", alternate_id)]:
        try:
            results = index.query(
                vector=[0.0] * 1536,  # Dummy vector
                filter={"lesson_id": lesson_id},
                top_k=1,
                include_metadata=True
            )
            
            if results["matches"]:
                print(f"✅ Found vectors using {format_name} format ID: {lesson_id}")
                print(f"Sample text: {results['matches'][0]['metadata']['text'][:100]}...")
            else:
                print(f"❌ No vectors found using {format_name} format ID: {lesson_id}")
        except Exception as e:
            print(f"Error checking {format_name} format: {e}")
    
    # Also check by title
    try:
        results = index.query(
            vector=[0.0] * 1536,
            filter={"title": "Carti recomandate in trading"},
            top_k=1,
            include_metadata=True
        )
        
        if results["matches"]:
            print(f"✅ Found vectors using title filter")
            print(f"Sample text: {results['matches'][0]['metadata']['text'][:100]}...")
        else:
            print(f"❌ No vectors found using title filter")
    except Exception as e:
        print(f"Error checking by title: {e}")

if __name__ == "__main__":
    check_lesson()