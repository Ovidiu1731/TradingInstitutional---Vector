#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

# Add the current directory to the path to import local modules
sys.path.append('.')

# Import our retrieval function
from improved_retrieval import retrieve_lesson_content

# Load environment variables
load_dotenv()

def test_sessions_retrieval():
    """Test different queries to see what sessions information is retrieved."""
    
    print("=== TESTING SESSIONS RETRIEVAL ===")
    
    queries = [
        "care sunt sesiunile de tranzactionare?",
        "intervale orare tranzactionare",
        "12:00 16:15 londra",
        "ore tranzactionare program zilnic",
        "structurarea programului zilnic",
        "toate intervalele de tranzactionare"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {query}")
        print('='*80)
        
        # Get the results using the same function as the API
        results = retrieve_lesson_content(query, chapter=None, lesson=None, top_k=5)
        
        if not results:
            print("❌ No results found!")
            continue
        
        print(f"📝 Results count: {len(results)}")
        
        for j, result in enumerate(results, 1):
            print(f"\nResult {j}:")
            print(f"  Chapter: {result.get('chapter', 'Unknown')}")
            print(f"  Lesson: {result.get('lesson', 'Unknown')}")
            print(f"  Text preview: {result.get('text', '')[:150]}...")
            
            # Check if this result contains the missing 12:00-16:15 info
            text_lower = result.get('text', '').lower()
            if '12:00' in text_lower and '16:15' in text_lower:
                print(f"  🎯 CONTAINS 12:00-16:15 INFO!")
            if 'program zilnic' in text_lower or 'structurarea' in text_lower:
                print(f"  📅 CONTAINS DAILY PROGRAM INFO!")

if __name__ == "__main__":
    test_sessions_retrieval() 