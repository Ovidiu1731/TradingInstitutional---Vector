import os
from dotenv import load_dotenv
import asyncio
import httpx
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Test cases with expected characteristics
TEST_CASES = [
    {
        "name": "Basic Liquidity Query",
        "query": "What are the types of liquidity?",
        "chapter": 11,
        "lesson": 2,
        "expected_keywords": ["major", "local", "minor", "liquidity"]
    },
    {
        "name": "FVG Query",
        "query": "Explain Fair Value Gap",
        "expected_keywords": ["gap", "imbalance", "institutional"]
    },
    {
        "name": "Setup Query",
        "query": "What is a Two Gap Setup?",
        "expected_keywords": ["TG", "setup", "consecutive"]
    },
    {
        "name": "Session Query",
        "query": "Explain London session trading",
        "expected_keywords": ["london", "session", "trading"]
    }
]

async def test_endpoint(query_data):
    """Test the /ask endpoint with a given query."""
    async with httpx.AsyncClient() as client:
        payload = {
            "question": query_data["query"],
            "chapter": query_data.get("chapter"),
            "lesson": query_data.get("lesson")
        }
        
        try:
            response = await client.post(
                "http://localhost:8000/ask",
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "context": result.get("context", ""),
                    "answer": result.get("answer", ""),
                    "processing_time": result.get("processing_time_ms", 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

def analyze_results(test_case, result):
    """Analyze the results of a test case."""
    if not result["success"]:
        return f"❌ Test failed: {result['error']}"
    
    context = result["context"].lower()
    answer = result["answer"].lower()
    
    # Check for expected keywords
    found_keywords = []
    missing_keywords = []
    
    for keyword in test_case["expected_keywords"]:
        if keyword.lower() in context or keyword.lower() in answer:
            found_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
    
    # Build analysis report
    report = []
    report.append(f"\n{'='*80}")
    report.append(f"Test: {test_case['name']}")
    report.append(f"Query: {test_case['query']}")
    report.append(f"Processing time: {result['processing_time']}ms")
    
    if found_keywords:
        report.append(f"\n✅ Found keywords: {', '.join(found_keywords)}")
    if missing_keywords:
        report.append(f"❌ Missing keywords: {', '.join(missing_keywords)}")
    
    # Add context preview
    if result["context"]:
        preview = result["context"][:200] + "..." if len(result["context"]) > 200 else result["context"]
        report.append(f"\nContext preview:\n{preview}")
    
    return "\n".join(report)

async def run_tests():
    """Run all test cases and generate a report."""
    print(f"\nStarting retrieval tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    for test_case in TEST_CASES:
        print(f"\nRunning test: {test_case['name']}")
        result = await test_endpoint(test_case)
        analysis = analyze_results(test_case, result)
        print(analysis)
        
        # Add a small delay between tests
        await asyncio.sleep(1)
    
    print("\n" + "="*80)
    print("Test suite completed")

if __name__ == "__main__":
    asyncio.run(run_tests()) 