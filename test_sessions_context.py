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

def test_sessions_context():
    """Test to see exactly what context is being sent to the AI for sessions questions."""
    
    print("=== TESTING SESSIONS CONTEXT ===")
    
    query = "care sunt sesiunile de tranzactionare?"
    
    print(f"Query: {query}")
    print()
    
    # Get the results using the same function as the API
    results = retrieve_lesson_content(query, chapter=None, lesson=None, top_k=5)
    
    if not results:
        print("❌ No results found!")
        return
    
    # Combine context exactly like the API does
    context_text = "\n\n".join([r["text"] for r in results])
    
    print(f"📝 Context length: {len(context_text)} characters")
    print(f"📝 Results count: {len(results)}")
    print()
    
    print("📄 FULL CONTEXT BEING SENT TO AI:")
    print("=" * 100)
    print(context_text)
    print("=" * 100)
    print()
    
    # Check if 12:00-16:15 is mentioned in the context
    if "12:00" in context_text and "16:15" in context_text:
        print("✅ Context CONTAINS 12:00-16:15 information!")
        
        # Find and highlight the specific mentions
        lines = context_text.split('\n')
        for i, line in enumerate(lines):
            if '12:00' in line and '16:15' in line:
                print(f"🎯 Line {i+1}: {line.strip()}")
    else:
        print("❌ Context does NOT contain 12:00-16:15 information")
    
    # Check what other interval information is present
    intervals_found = []
    for line in context_text.split('\n'):
        line_lower = line.lower().strip()
        if any(time in line for time in ['10:15', '12:00', '16:15', '16:45', '19:00', '22:00']):
            intervals_found.append(line.strip())
    
    print(f"\n📅 INTERVAL INFORMATION FOUND ({len(intervals_found)} lines):")
    for i, interval_line in enumerate(intervals_found, 1):
        print(f"  {i}. {interval_line}")

if __name__ == "__main__":
    test_sessions_context() 