from datetime import datetime, date, time
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
        # Convert dates to timestamps
        from_dt = datetime.combine(from_date, from_time or time.min)
        to_dt = datetime.combine(to_date, to_time or time.max)
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/historical-chart/{timeframe}/{symbol}",
                    params={
                        "apikey": self.api_key,
                        "from": from_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "to": to_dt.strftime("%Y-%m-%d %H:%M:%S")
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                candles = []
                for item in data:
                    candle = CandleData(
                        date=datetime.fromisoformat(item["date"].replace("Z", "+00:00")),
                        open=float(item["open"]),
                        high=float(item["high"]),
                        low=float(item["low"]),
                        close=float(item["close"]),
                        volume=int(item["volume"])
                    )
                    candles.append(candle)
                
                return CandleResponse(
                    candles=candles,
                    symbol=symbol,
                    timeframe=timeframe,
                    from_date=from_dt,
                    to_date=to_dt
                )
            except httpx.HTTPError as e:
                raise HTTPException(status_code=500, detail=f"Error fetching candles: {str(e)}") 