from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class CandleData(BaseModel):
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

class CandleResponse(BaseModel):
    candles: list[CandleData]
    symbol: str
    timeframe: str = "1min"
    from_date: datetime
    to_date: datetime

class MarketStructurePoint(BaseModel):
    timestamp: datetime
    price: float
    type: str = Field(..., pattern="^(high|low)$")
    strength: int = Field(..., ge=1, le=3)  # 1: weak, 2: moderate, 3: strong
    is_valid: bool = True  # Whether the point passes the 2-candle validation rule

class LiquiditySweep(BaseModel):
    type: str = Field(..., pattern="^(hod_sweep|lod_sweep)$")
    timestamp: datetime
    price: float
    strength: int = Field(..., ge=1, le=3)

class LiquidityZone(BaseModel):
    type: str = Field(..., pattern="^(major|local)$")
    price: float
    strength: int
    description: str

class MarketStructure(BaseModel):
    timestamp: datetime | None = None
    current_trend: str = Field(..., pattern="^(uptrend|downtrend|sideways)$")
    valid_structure_points: list[dict] = []
    mss_detected: bool = False
    mss_type: str | None = None
    mss_level: float | None = None
    hod_level: float | None = None
    lod_level: float | None = None
    hod_swept: bool = False
    lod_swept: bool = False
    valid_gap_detected: bool = False
    institutional_gaps: list[dict] = []
    setup_classified: bool = False
    setup_type: str | None = None
    potential_entry_zone: dict | None = None
    suggested_stop_loss_level: float | None = None
    suggested_take_profit_level: float | None = None
    liquidity_zones: list[dict] = []
    notes: str | None = None

class SymbolInfo(BaseModel):
    symbol: str
    fromCurrency: str
    toCurrency: str
    fromName: str
    toName: str

class SymbolsResponse(BaseModel):
    symbols: list[SymbolInfo] 