# ----------------------------------------------------------------------
# fastapi_app.py  (or whichever module mounts this router)
# ----------------------------------------------------------------------
from datetime import datetime, date, time, timedelta
from fastapi import FastAPI, APIRouter, HTTPException, Query
from app.services.market_data import MarketDataService
from app.services.market_analysis import MarketAnalysisService
from app.models.candle import CandleResponse, MarketStructure, SymbolsResponse
from app.models.assistant_contract import AssistantContract

app = FastAPI()
router = APIRouter(prefix="/candles", tags=["candles"])

market_data_service = MarketDataService()
market_analysis_service = MarketAnalysisService()


@router.get(
    "/symbols",
    response_model=SymbolsResponse,
    summary="Fetch all available forex symbols"
)
async def get_symbols():
    """
    Fetch all available forex symbols from the FMP API.
    Returns a list of symbols with their currency pairs and names.
    """
    try:
        symbols = await market_data_service.get_available_symbols()
        return SymbolsResponse(symbols=symbols)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{symbol}", 
    response_model=CandleResponse,
    summary="Fetch raw candle data for a symbol in a date/time range"
)
async def get_candles(
    symbol: str,
    from_date: date = Query(
        ..., 
        description="Start date (YYYY-MM-DD; e.g. 2024-03-22)",
    ),
    to_date: date = Query(
        ..., 
        description="End date (YYYY-MM-DD; e.g. 2024-03-22)",
    ),
    from_time: time | None = Query(
        None, 
        description="Start time (HH:MM:SS). If omitted, defaults to 00:00:00 on from_date."
    ),
    to_time: time | None = Query(
        None, 
        description="End time (HH:MM:SS). If omitted, defaults to 23:59:59 on to_date."
    ),
    timeframe: str = Query(
        "1min",
        description="Candle timeframe (e.g. 1min, 5min, 1h)."
    ),
):
    """
    Fetch candlestick data for a given symbol and date/time range. 
    Handles "same-day" with possible overnight rollover if from_time > to_time.
    """
    # 1) Basic validation
    if from_date > to_date:
        raise HTTPException(status_code=400, detail="`from_date` cannot be after `to_date`.")

    # 2) Delegate to MarketDataService
    try:
        return await market_data_service.get_candles(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            from_time=from_time,
            to_time=to_time,
            timeframe=timeframe
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{symbol}/analysis", 
    response_model=MarketStructure,
    summary="Analyze market structure for a symbol in a date/time range"
)
async def analyze_market(
    symbol: str,
    from_date: date = Query(
        ..., 
        description="Start date (YYYY-MM-DD; e.g. 2024-03-22)"
    ),
    to_date: date = Query(
        ..., 
        description="End date (YYYY-MM-DD; e.g. 2024-03-22)"
    ),
    from_time: time | None = Query(
        None,
        description="Start time (HH:MM:SS). If omitted, defaults to 00:00:00 on from_date."
    ),
    to_time: time | None = Query(
        None,
        description="End time (HH:MM:SS). If omitted, defaults to 23:59:59 on to_date."
    ),
    timeframe: str = Query(
        "1min",
        description="Candle timeframe (e.g. 1min, 5min, 1h)."
    ),
):
    """
    Analyze market structure: First fetch candles (with the same logic as /{symbol}), 
    then run market-structure logic against those candles.
    """
    if from_date > to_date:
        raise HTTPException(status_code=400, detail="`from_date` cannot be after `to_date`.")

    try:
        # 1) Get raw candles
        candle_response = await market_data_service.get_candles(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            from_time=from_time,
            to_time=to_time,
            timeframe=timeframe
        )
        # 2) Analyze structure
        return market_analysis_service.analyze_market_structure(candle_response.candles)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{symbol}/analysis/assistant",
    response_model=dict,
    summary="Get analysis in assistant-ready format"
)
async def get_assistant_ready_analysis(
    symbol: str,
    from_date: date = Query(
        ..., 
        description="Start date (YYYY-MM-DD; e.g. 2024-03-22)"
    ),
    to_date: date = Query(
        ..., 
        description="End date (YYYY-MM-DD; e.g. 2024-03-22)"
    ),
    from_time: time | None = Query(
        None,
        description="Start time (HH:MM:SS). If omitted, defaults to 00:00:00 on from_date."
    ),
    to_time: time | None = Query(
        None,
        description="End time (HH:MM:SS). If omitted, defaults to 23:59:59 on to_date."
    ),
    timeframe: str = Query(
        "1min",
        description="Candle timeframe (e.g. 1min, 5min, 1h)."
    ),
):
    """
    Get market analysis in a format optimized for the assistant service.
    This endpoint transforms the internal MarketStructure into the AssistantContract format.
    """
    if from_date > to_date:
        raise HTTPException(status_code=400, detail="`from_date` cannot be after `to_date`.")

    try:
        # 1) Get raw candles
        candle_response = await market_data_service.get_candles(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            from_time=from_time,
            to_time=to_time,
            timeframe=timeframe
        )
        
        # 2) Analyze structure
        market_structure = market_analysis_service.analyze_market_structure(candle_response.candles)
        
        # 3) Convert to assistant format
        assistant_contract = AssistantContract.from_market_structure(market_structure)
        
        return assistant_contract.model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router)