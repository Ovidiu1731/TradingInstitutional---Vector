#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Add the current directory to the path to import local modules
sys.path.append('.')

# Import our retrieval function
from improved_retrieval import retrieve_lesson_content

# Load environment variables
load_dotenv()

def test_sessions_question():
    """Test the sessions question to see if it gives clean, context-based response."""
    
    print("=== TESTING SESSIONS QUESTION FIX ===")
    
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
    
    # Print a preview of the context to see what's being retrieved
    print("📄 CONTEXT PREVIEW:")
    print("-" * 60)
    print(context_text[:500] + "..." if len(context_text) > 500 else context_text)
    print("-" * 60)
    print()
    
    # Test with new API logic (context-aware)
    is_sessions_question = any(term in query.lower() for term in ['sesiuni', 'session', 'londra', 'london', 'new york', 'ny', 'tokyo', 'sydney', 'tranzactionare', 'trading hours', 'ore'])
    
    print(f"🔍 Detected as sessions question: {is_sessions_question}")
    
    # DEBUG: Test interval extraction like in app.py
    if is_sessions_question:
        intervals_found = []
        context_lower = context_text.lower()
        if "12:00" in context_text and "16:15" in context_text:
            intervals_found.append("12:00-16:15 (pentru lichidități)")
        if "16:45" in context_text and "22:00" in context_text:
            intervals_found.append("16:45-22:00 (pentru lichidități)")
        
        print(f"🔍 DEBUG: Intervals found: {intervals_found}")
        
        intervals_reminder = ""
        if intervals_found:
            intervals_reminder = f"\n\nATENȚIE: Am identificat în context următoarele intervale care TREBUIE incluse în răspuns: {', '.join(intervals_found)}"
            print(f"🔍 DEBUG: Intervals reminder: {intervals_reminder}")
    
    print()
    
    # Clean base system prompt without hardcoded information
    base_system_prompt = """You are a professional AI assistant helping students from the Trading Instituțional community. You answer only in Romanian. Your responses must be clear, short, and direct, based strictly on the official course materials taught by Rareș. Do not add general trading theory, made-up examples, or content outside the course materials.

Instrucțiuni pentru răspunsuri:
1. Răspunde în română, fii concis dar complet
2. Bazează-te strict pe informațiile furnizate în context
3. IMPORTANT: Asigură-te că incluzi TOATE tipurile sau categoriile menționate în context, nu omite nimic
4. Evită formulările robotice repetitive precum "este important să..." la sfârșitul fiecărei propoziții
5. Folosește un ton natural, ca al unui coleg de trading cu experiență
6. Nu adăuga informații care nu sunt prezente în contextul furnizat
7. Concentrează-te pe întrebarea specifică și răspunde doar pe baza materialului disponibil
8. Nu folosi informații din afara contextului furnizat, chiar dacă le cunoști din alte surse
"""

    if is_sessions_question:
        user_message = f"Întrebare: {query}\n\nContext din material:\n{context_text}\n\nTe rog să răspunzi pe baza strict a informațiilor din context. Prezintă TOATE intervalele de tranzacționare menționate în context, nu doar primul găsit. IMPORTANT: Nu folosi expresii precum 'cea mai importantă sesiune' sau 'most important session' - prezintă neutral informațiile despre fiecare sesiune fără să faci comparații de importanță.{intervals_reminder}"
    else:
        user_message = f"Întrebare: {query}\n\nContext din material:\n{context_text}\n\nTe rog să răspunzi pe baza informațiilor din context, într-un mod natural și concis."
    
    messages = [
        {"role": "system", "content": base_system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    print("🤖 Calling OpenAI API...")
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        answer = completion.choices[0].message.content.strip()
        
        print("\n📤 AI RESPONSE:")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        
        # Check for the specific problems mentioned
        answer_lower = answer.lower()
        
        print("\n🔍 ANALYSIS:")
        
        # Problem 1: Contradictory language
        if "se tranzacționează următoarele sesiuni" in answer_lower and ("sydney" in answer_lower or "tokyo" in answer_lower):
            if "nu este" in answer_lower or "neimportant" in answer_lower:
                print("⚠️  CONTRADICTION: Says 'se tranzacționează următoarele sesiuni' but then admits some aren't traded")
            else:
                print("✅ No contradiction detected")
        else:
            print("✅ No contradictory language found")
        
        # Problem 2: Misleading London session description
        if "cea mai importantă sesiune" in answer_lower or "most important" in answer_lower:
            print("⚠️  MISLEADING: Claims London is 'the most important session'")
        else:
            print("✅ No misleading 'most important' claim")
            
        if "10:15" in answer and "12:00" in answer:
            if "doar" in answer_lower or "numai" in answer_lower:
                print("⚠️  MISLEADING: Suggests 10:15-12:00 is the only trading time")
            else:
                print("✅ 10:15-12:00 mentioned appropriately")
        
        # Check if it mentions that 12:00-16:15 is also tradeable
        if "12:00" in answer and "16:15" in answer:
            print("✅ Mentions 12:00-16:15 period")
        else:
            print("⚠️  MISSING: Doesn't clearly mention 12:00-16:15 is tradeable")
            
    except Exception as e:
        print(f"❌ Error calling OpenAI API: {e}")

if __name__ == "__main__":
    test_sessions_question() 