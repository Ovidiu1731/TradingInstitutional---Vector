#!/usr/bin/env python3
"""
Debug script for market analysis queries.
Usage: python3 debug_query.py "your query here"
"""

import asyncio
import json
import sys
import logging
from datetime import datetime
from app import (
    process_user_input_to_api_params, 
    normalize_symbol, 
    process_api_response_to_natural_language,
    MarketDataService
)
from services.setup_analysis import SetupAnalysisService

# Set up logging
logging.basicConfig(level=logging.INFO)

async def debug_query(question: str):
    """Debug a market analysis query and print formatted results"""
    print(f"🔍 DEBUGGING QUERY: {question}")
    print("=" * 80)
    
    try:
        # Step 1: Parameter extraction debug
        print("📝 STEP 1 - PARAMETER EXTRACTION:")
        extraction_result = await process_user_input_to_api_params(question)
        
        if not extraction_result["success"]:
            print(f"❌ EXTRACTION FAILED: {extraction_result['error']}")
            return
        
        api_params = extraction_result["params"]
        raw_symbol = api_params["symbol"]
        normalized_symbol = normalize_symbol(raw_symbol)
        api_params["symbol"] = normalized_symbol
        
        print(f"   Raw Symbol: {raw_symbol}")
        print(f"   Normalized Symbol: {normalized_symbol}")
        print(f"   Romanian Time: {api_params.get('original_romanian_start')} → {api_params.get('original_romanian_end')}")
        print(f"   UTC Time: {api_params['start_date']} → {api_params['end_date']}")
        print()
        
        # Step 2: Get raw candle data
        print("📊 STEP 2 - DATA RETRIEVAL:")
        start_dt = datetime.fromisoformat(api_params["start_date"])
        end_dt = datetime.fromisoformat(api_params["end_date"])
        
        market_data_service = MarketDataService()
        candle_response = await market_data_service.get_candles(
            symbol=normalized_symbol,
            from_date=start_dt.date(),
            to_date=end_dt.date(),
            from_time=start_dt.time(),
            to_time=end_dt.time(),
            timeframe=api_params.get("timeframe", "1min")
        )
        
        api_url = f"https://financialmodelingprep.com/api/v3/historical-chart/{api_params.get('timeframe', '1min')}/{normalized_symbol}"
        print(f"   API URL: {api_url}")
        print(f"   Total Candles: {len(candle_response.candles)}")
        
        if candle_response.candles:
            print(f"   Data Range: {candle_response.candles[0].date.isoformat()} → {candle_response.candles[-1].date.isoformat()}")
            
            # Show sample candle data
            print(f"   Sample Candles (first/last):")
            for i, candle in enumerate(candle_response.candles[:3]):  # First 3
                print(f"     [{i}] {candle.date.isoformat()}: O={candle.open:.5f} H={candle.high:.5f} L={candle.low:.5f} C={candle.close:.5f}")
            
            if len(candle_response.candles) > 6:
                print("     ...")
                for i, candle in enumerate(candle_response.candles[-3:], start=len(candle_response.candles)-3):
                    print(f"     [{i}] {candle.date.isoformat()}: O={candle.open:.5f} H={candle.high:.5f} L={candle.low:.5f} C={candle.close:.5f}")
        else:
            print("   ❌ NO CANDLES RETRIEVED!")
        print()
        
        # Step 3: Setup analysis debug
        print("🔬 STEP 3 - SETUP ANALYSIS:")
        setup_analysis_service = SetupAnalysisService()
        setup_analysis = setup_analysis_service.analyze_setup(candle_response.candles)
        formatted_output = setup_analysis_service.format_analysis_output(setup_analysis)
        
        # MSS Analysis
        mss = setup_analysis.get("mss", {})
        print(f"   MSS: {'✅ Detected' if mss.get('detected') else '❌ Not detected'}")
        if mss.get("detected"):
            print(f"        Type: {mss.get('type')}")
            print(f"        Validity: {mss.get('validity')}")
            print(f"        Broken Level: {mss.get('broken_level')}")
            print(f"        Break Time: {mss.get('break_timestamp')}")
            print(f"        Reason: {mss.get('reason')}")
        
        # Displacement Analysis
        displacement = setup_analysis.get("displacement", {})
        print(f"   Displacement: {'✅ Detected' if displacement.get('detected') else '❌ Not detected'}")
        if displacement.get("detected"):
            print(f"        Count: {displacement.get('count')}")
            movements = displacement.get("movements", [])
            for i, move in enumerate(movements[:3]):  # Show first 3
                print(f"        Movement {i+1}: {move['direction']} {move['movement_percent']:.2f}% ({move['start_time']} → {move['end_time']})")
        else:
            print(f"        Reason: {displacement.get('reason', 'Not specified')}")
        
        # Gaps Analysis
        gaps = setup_analysis.get("gaps", {})
        print(f"   Gaps: {'✅ Detected' if gaps.get('detected') else '❌ Not detected'}")
        if gaps.get("detected"):
            print(f"        Count: {gaps.get('count')}")
            gap_list = gaps.get("gaps", [])
            for i, gap in enumerate(gap_list[:3]):  # Show first 3
                print(f"        Gap {i+1}: {gap['type']} (size: {gap['gap_size']:.5f}) {gap['start_time']} → {gap['end_time']}")
        else:
            print(f"        Reason: {gaps.get('reason', 'Not specified')}")
        
        # Setup Classification
        setup = setup_analysis.get("setup", {})
        print(f"   Setup: {setup.get('type', 'Unknown')}")
        if setup.get("type") != "invalid":
            print(f"        Name: {setup.get('name')}")
            print(f"        Description: {setup.get('description')}")
        else:
            print(f"        Reason: {setup.get('reason')}")
        
        print(f"   Technical Output: {formatted_output}")
        print()
        
        # Step 4: LLM natural language conversion debug
        print("🤖 STEP 4 - FINAL RESPONSE:")
        structured_response = {
            "setup_analysis": setup_analysis,
            "formatted_analysis": formatted_output,
            "candle_count": len(candle_response.candles),
            "timeframe_analyzed": f"{start_dt.isoformat()} to {end_dt.isoformat()}",
            "symbol": normalized_symbol
        }
        
        # Fix datetime serialization
        def fix_datetime_serialization(data):
            if isinstance(data, dict):
                return {k: fix_datetime_serialization(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [fix_datetime_serialization(item) for item in data]
            elif hasattr(data, 'isoformat'):
                return data.isoformat()
            else:
                return data
        
        structured_response = fix_datetime_serialization(structured_response)
        natural_response = await process_api_response_to_natural_language(structured_response, question)
        
        print(f"   Final Answer: {natural_response}")
        print()
        
        # Validation checks
        print("✅ VALIDATION CHECKS:")
        checks = {
            "candle_count_matches": len(candle_response.candles) == structured_response["candle_count"],
            "timeframe_correct": api_params["start_date"] in structured_response["timeframe_analyzed"],
            "symbol_correct": normalized_symbol == structured_response["symbol"]
        }
        
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check}: {passed}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 debug_query.py \"your query here\"")
        print("Example: python3 debug_query.py \"analizeaza gbpusd de la 15:32 la 16:05 pentru 20/06/2025\"")
        sys.exit(1)
    
    question = sys.argv[1]
    asyncio.run(debug_query(question))

if __name__ == "__main__":
    main() 