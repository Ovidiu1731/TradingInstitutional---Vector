#!/usr/bin/env python3
"""
Test script for the LLM bridge integration.
Tests the complete flow: User Input → LLM Processing → FMP Service → LLM Formatting → User Output
"""

import asyncio
import aiohttp
import json
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"  # Adjust based on your server

async def test_market_analysis_bridge():
    """Test the new market analysis bridge functionality"""
    
    print("🔵 TESTING LLM BRIDGE INTEGRATION")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "name": "English Date Range",
            "input": "analyze EURUSD from June 21 to June 22",
            "expected_symbol": "EURUSD",
            "expected_date_format": "2024-06-21T00:00:00"
        },
        {
            "name": "Romanian Specific Time",
            "input": "analizeaza EUR/USD pentru 16-03-2024 de la 10:15 pana la 10:30",
            "expected_symbol": "EURUSD",
            "expected_date_format": "2024-03-16T10:15:00"
        },
        {
            "name": "Simple Format",
            "input": "analyze GBPUSD today",
            "expected_symbol": "GBPUSD",
            "expected_date_format": f"{datetime.now().year}"
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📊 TEST CASE {i}: {test_case['name']}")
            print(f"Input: {test_case['input']}")
            
            try:
                # Call the new analyze-market endpoint
                payload = {
                    "question": test_case['input'],
                    "session_id": f"test-session-{i}"
                }
                
                async with session.post(
                    f"{API_BASE_URL}/analyze-market",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        print(f"✅ SUCCESS - Status: {response.status}")
                        print(f"📝 Natural Response: {data.get('answer', 'No answer')[:200]}...")
                        
                        # Check if we have structured data
                        if 'structured_data' in data:
                            print(f"📊 Structured Data Available: Yes")
                            structured = data['structured_data']
                            print(f"   - Analysis Possible: {structured.get('analysis_possible', 'Unknown')}")
                            print(f"   - Trade Direction: {structured.get('final_trade_direction', 'Unknown')}")
                            print(f"   - MSS Type: {structured.get('final_mss_type', 'Unknown')}")
                        
                        # Check API parameters
                        if 'api_params' in data:
                            print(f"🔧 API Parameters:")
                            params = data['api_params']
                            print(f"   - Symbol: {params.get('symbol', 'Unknown')}")
                            print(f"   - Start Date: {params.get('start_date', 'Unknown')}")
                            print(f"   - End Date: {params.get('end_date', 'Unknown')}")
                            print(f"   - Timeframe: {params.get('timeframe', 'Unknown')}")
                            
                            # Validate extracted symbol
                            if params.get('symbol') == test_case['expected_symbol']:
                                print(f"✅ Symbol extraction: PASS")
                            else:
                                print(f"❌ Symbol extraction: FAIL (expected {test_case['expected_symbol']}, got {params.get('symbol')})")
                        
                    else:
                        print(f"❌ FAILED - Status: {response.status}")
                        text = await response.text()
                        print(f"Error: {text}")
                        
            except Exception as e:
                print(f"❌ EXCEPTION: {e}")
            
            print("-" * 40)

async def test_parameter_extraction_only():
    """Test just the parameter extraction function"""
    print("\n🔧 TESTING PARAMETER EXTRACTION ONLY")
    print("=" * 50)
    
    test_inputs = [
        "analyze EURUSD from June 21 to June 22",
        "analizeaza EUR/USD pentru 16-03-2024 de la 10:15 pana la 10:30",
        "check GBPUSD for March 15th 2024",
        "analiza DAX azi",
    ]
    
    for input_text in test_inputs:
        print(f"\n📝 Input: {input_text}")
        
        # This would test the parameter extraction if we could import it
        # For now, we'll test via the API endpoint
        
        payload = {"question": input_text, "session_id": "test-param-extraction"}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{API_BASE_URL}/analyze-market",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'api_params' in data:
                            params = data['api_params']
                            print(f"✅ Extracted: {json.dumps(params, indent=2)}")
                        else:
                            print(f"❌ No parameters extracted")
                    else:
                        print(f"❌ API Error: {response.status}")
            except Exception as e:
                print(f"❌ Exception: {e}")

async def test_logging_output():
    """Test that logging is working properly"""
    print("\n📋 TESTING LOGGING OUTPUT")
    print("=" * 50)
    print("Check the server logs for the following emojis:")
    print("🔵 - Input processing stages")
    print("🟢 - Successful operations") 
    print("🟡 - API calls and responses")
    print("🔴 - Errors and failures")
    
    # Make a single request to generate log output
    payload = {"question": "analyze EURUSD from June 21 to June 22", "session_id": "log-test"}
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{API_BASE_URL}/analyze-market",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    print("✅ Request completed. Check server logs for detailed emoji-coded logging.")
                else:
                    print(f"❌ Request failed with status {response.status}")
        except Exception as e:
            print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    print("🚀 STARTING BRIDGE INTEGRATION TESTS")
    print(f"Server URL: {API_BASE_URL}")
    print("\nMake sure your server is running on the specified URL.")
    input("Press Enter to continue...")
    
    asyncio.run(test_parameter_extraction_only())
    asyncio.run(test_market_analysis_bridge())
    asyncio.run(test_logging_output())
    
    print("\n🏁 TESTS COMPLETED")
    print("Check the server logs to verify all logging points are working correctly.") 