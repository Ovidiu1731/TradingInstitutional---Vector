#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
import asyncio

# Add the current directory to the path to import local modules
sys.path.append('.')

# Import our retrieval function
from improved_retrieval import retrieve_lesson_content

# Load environment variables
load_dotenv()

def test_comprehensive_hallucination():
    """Test various questions to check for different types of hallucination."""
    
    print("=== COMPREHENSIVE HALLUCINATION TEST ===")
    
    # Test cases with questions that should NOT trigger specific trading concepts
    test_cases = [
        {
            "query": "ce inseamna FTMO?",
            "should_not_contain": ["lichiditate", "hod", "lod", "major", "local", "minor", "10:15", "12:00", "londra", "new york", "40-60 puncte", "dax", "1.4r", "break-even"],
            "description": "FTMO question should not mention liquidity, sessions, or DAX stop loss"
        },
        {
            "query": "cum folosesc MetaTrader?",
            "should_not_contain": ["lichiditate", "hod", "lod", "displacement", "gap", "mss", "londra", "new york", "10:15"],
            "description": "MetaTrader question should not mention trading concepts"
        },
        {
            "query": "ce este un broker?",
            "should_not_contain": ["lichiditate", "hod", "lod", "15m", "1m", "5m", "displacement", "gap"],
            "description": "Broker question should not mention specific trading strategies"
        },
        {
            "query": "cum calculez loturile?",
            "should_not_contain": ["lichiditate", "hod", "lod", "londra", "new york", "10:15", "12:00"],
            "description": "Lot calculation should not mention sessions or liquidity types"
        },
        {
            "query": "ce este psihologia in trading?",
            "should_not_contain": ["40-60 puncte", "dax", "1.4r", "break-even", "10:15", "12:00", "16:45"],
            "description": "Psychology question should not mention specific instruments or times"
        }
    ]
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        print('='*80)
        
        # Get retrieval results
        results = retrieve_lesson_content(test_case['query'], chapter=None, lesson=None, top_k=5)
        
        if not results:
            print("❌ No results found - skipping test")
            continue
        
        # Combine context
        context_text = "\n\n".join([r["text"] for r in results])
        
        # Check if context contains problematic terms
        context_lower = context_text.lower()
        context_contamination = [term for term in test_case['should_not_contain'] if term in context_lower]
        
        print(f"📝 Context length: {len(context_text)} characters")
        if context_contamination:
            print(f"⚠️  Context already contains: {context_contamination}")
        else:
            print("✅ Context is clean")
        
        # Test with current API logic (context-aware)
        is_liquidity_question = any(term in test_case['query'].lower() for term in ['lichiditate', 'liquidity', 'hod', 'lod', 'major', 'local', 'minor'])
        
        base_system_prompt = """Instrucțiuni pentru răspunsuri:
1. Răspunde în română, fii concis dar complet
2. Bazează-te strict pe informațiile furnizate în context
3. IMPORTANT: Asigură-te că incluzi TOATE tipurile sau categoriile menționate în context, nu omite nimic
4. Evită formulările robotice repetitive precum "este important să..." la sfârșitul fiecărei propoziții
5. Folosește un ton natural, ca al unui coleg de trading cu experiență
6. Nu adăuga informații care nu sunt prezente în contextul furnizat
7. Concentrează-te pe întrebarea specifică și răspunde doar pe baza materialului disponibil
"""

        if is_liquidity_question:
            liquidity_guidance = """

Ghid pentru tipurile de lichiditate (DOAR dacă sunt menționate în context):
- HOD/LOD: Maximele și minimele zilei curente
- Lichiditatea Majoră: Cea mai profitabilă, marcată pe TF 15m în zonele extreme, mai rar întâlnită
- Lichiditatea Locală: Marcată pe TF 1m-5m, nu la fel de puternică ca cea majoră
- Lichiditatea Minoră: Susține trendul, necesită experiență pentru identificare
"""
            system_prompt = base_system_prompt + liquidity_guidance
            user_message = f"Întrebare: {test_case['query']}\n\nContext din material:\n{context_text}\n\nTe rog să incluzi toate tipurile de lichiditate menționate în context, inclusiv HOD/LOD dacă este prezent. Prezintă informațiile într-un mod natural, fără formule robotice."
        else:
            system_prompt = base_system_prompt
            user_message = f"Întrebare: {test_case['query']}\n\nContext din material:\n{context_text}\n\nTe rog să răspunzi pe baza informațiilor din context, într-un mod natural și concis."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        print("\n🤖 Calling OpenAI API...")
        
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = completion.choices[0].message.content.strip()
            
            print("\n📤 AI RESPONSE:")
            print("-" * 60)
            print(answer)
            print("-" * 60)
            
            # Check for hallucination
            answer_lower = answer.lower()
            hallucinated_terms = []
            
            for term in test_case['should_not_contain']:
                if term in answer_lower and term not in context_lower:
                    hallucinated_terms.append(term)
            
            if hallucinated_terms:
                print(f"\n🚨 HALLUCINATION DETECTED!")
                print(f"AI added terms NOT in context: {hallucinated_terms}")
            else:
                print(f"\n✅ CLEAN RESPONSE: No hallucination detected")
                
        except Exception as e:
            print(f"❌ Error calling OpenAI API: {e}")
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_comprehensive_hallucination() 