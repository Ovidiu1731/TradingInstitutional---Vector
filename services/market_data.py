from datetime import datetime, date, time, timezone, timedelta
from typing import List
import httpx
from fastapi import HTTPException
from models.candle import CandleData, CandleResponse, SymbolInfo
from services.config import get_settings

settings = get_settings()

class MarketDataService:
    def __init__(self):
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.api_key = settings.FMP_API_KEY

    async def get_available_symbols(self) -> List[SymbolInfo]:
        """Fetch all available forex symbols from FMP API."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/symbol/available-forex-currency-pairs",
                    params={"apikey": self.api_key}
                )
                response.raise_for_status()
                data = response.json()
                
                symbols = []
                for item in data:
                    symbol = SymbolInfo(
                        symbol=item["symbol"],
                        fromCurrency=item["fromCurrency"],
                        toCurrency=item["toCurrency"],
                        fromName=item["fromName"],
                        toName=item["toName"]
                    )
                    symbols.append(symbol)
                return symbols
            except httpx.HTTPError as e:
                raise HTTPException(status_code=500, detail=f"Error fetching symbols: {str(e)}")

    async def get_candles(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        from_time: time | None = None,
        to_time: time | None = None,
        timeframe: str = "1min"
    ) -> CandleResponse:
        """Fetch candlestick data for a given symbol and date/time range."""
        # Convert dates to timestamps (input is in UTC)
        from_dt_utc = datetime.combine(from_date, from_time or time.min).replace(tzinfo=timezone.utc)
        to_dt_utc = datetime.combine(to_date, to_time or time.max).replace(tzinfo=timezone.utc)
        
        # CRITICAL FIX: Convert UTC to ET time for FMP API
        # FMP API expects ET time, not UTC
        et_timezone = timezone(timedelta(hours=-5))  # EDT (UTC-5) - adjust for EST/EDT as needed
        from_dt_et = from_dt_utc.astimezone(et_timezone)
        to_dt_et = to_dt_utc.astimezone(et_timezone)
        
        # Use ET times for API call but keep UTC for filtering
        from_dt = from_dt_utc.replace(tzinfo=None)  # Remove timezone for filtering
        to_dt = to_dt_utc.replace(tzinfo=None)
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/historical-chart/{timeframe}/{symbol}",
                    params={
                        "apikey": self.api_key,
                        "from": from_dt_et.strftime("%Y-%m-%d %H:%M:%S"),
                        "to": to_dt_et.strftime("%Y-%m-%d %H:%M:%S")
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                candles = []
                for item in data:
                    # Parse FMP timestamp (ET time) and convert to UTC for consistency
                    fmp_timestamp = datetime.fromisoformat(item["date"])
                    # FMP returns ET time, convert to UTC
                    fmp_et = fmp_timestamp.replace(tzinfo=et_timezone)
                    fmp_utc = fmp_et.astimezone(timezone.utc).replace(tzinfo=None)
                    
                    candle = CandleData(
                        date=fmp_utc,
                        open=float(item["open"]),
                        high=float(item["high"]),
                        low=float(item["low"]),
                        close=float(item["close"]),
                        volume=int(item["volume"])
                    )
                    candles.append(candle)
                
                # CRITICAL FIX: Filter candles to the requested time range
                # FMP API ignores time parameters and returns full dataset
                filtered_candles = []
                for candle in candles:
                    # Convert candle time to UTC for comparison
                    candle_time = candle.date
                    if candle_time >= from_dt and candle_time <= to_dt:
                        filtered_candles.append(candle)
                
                # CRITICAL FIX: Sort candles chronologically (oldest first)
                # FMP API returns data in reverse chronological order (newest first)
                filtered_candles.sort(key=lambda x: x.date)
                
                # Log filtering results for debugging
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"🕐 TIMEZONE CONVERSION: UTC {from_dt} to {to_dt} → ET {from_dt_et.strftime('%Y-%m-%d %H:%M:%S')} to {to_dt_et.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"FMP API returned {len(candles)} candles, filtered to {len(filtered_candles)} for time range {from_dt} to {to_dt}")
                if filtered_candles:
                    logger.info(f"Chronological order: {filtered_candles[0].date} to {filtered_candles[-1].date}")
                
                return CandleResponse(
                    candles=filtered_candles,
                    symbol=symbol,
                    timeframe=timeframe,
                    from_date=from_dt,
                    to_date=to_dt
                )
            except httpx.HTTPError as e:
                raise HTTPException(status_code=500, detail=f"Error fetching candles: {str(e)}") 