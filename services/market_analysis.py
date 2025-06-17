from typing import List
from models.candle import CandleData, MarketStructure, MarketStructurePoint, LiquidityZone

class MarketAnalysisService:
    def analyze_market_structure(self, candles: List[CandleData]) -> MarketStructure:
        """Analyze market structure from a list of candles."""
        if not candles:
            return MarketStructure(
                current_trend="sideways",
                valid_structure_points=[],
                mss_detected=False
            )

        # Sort candles by date
        sorted_candles = sorted(candles, key=lambda x: x.date)
        
        # Initialize market structure
        structure = MarketStructure(
            timestamp=sorted_candles[-1].date,
            current_trend=self._determine_trend(sorted_candles),
            valid_structure_points=self._find_structure_points(sorted_candles),
            mss_detected=False,  # To be implemented
            liquidity_zones=self._identify_liquidity_zones(sorted_candles)
        )
        
        return structure

    def _determine_trend(self, candles: List[CandleData]) -> str:
        """Determine the current market trend."""
        if len(candles) < 2:
            return "sideways"
            
        # Simple trend detection based on last 20 candles
        recent_candles = candles[-20:]
        highs = [c.high for c in recent_candles]
        lows = [c.low for c in recent_candles]
        
        if max(highs) == highs[-1] and min(lows) == lows[0]:
            return "uptrend"
        elif max(highs) == highs[0] and min(lows) == lows[-1]:
            return "downtrend"
        return "sideways"

    def _find_structure_points(self, candles: List[CandleData]) -> List[dict]:
        """Find market structure points (highs and lows)."""
        if len(candles) < 3:
            return []
            
        structure_points = []
        
        for i in range(1, len(candles) - 1):
            # Check for high point
            if candles[i].high > candles[i-1].high and candles[i].high > candles[i+1].high:
                structure_points.append({
                    "timestamp": candles[i].date,
                    "price": candles[i].high,
                    "type": "high",
                    "strength": self._calculate_strength(candles, i, "high")
                })
            
            # Check for low point
            if candles[i].low < candles[i-1].low and candles[i].low < candles[i+1].low:
                structure_points.append({
                    "timestamp": candles[i].date,
                    "price": candles[i].low,
                    "type": "low",
                    "strength": self._calculate_strength(candles, i, "low")
                })
        
        return structure_points

    def _calculate_strength(self, candles: List[CandleData], index: int, point_type: str) -> int:
        """Calculate the strength of a structure point (1-3)."""
        if point_type == "high":
            price = candles[index].high
            # Check how many candles before and after are lower
            before_lower = sum(1 for i in range(max(0, index-3), index) if candles[i].high < price)
            after_lower = sum(1 for i in range(index+1, min(len(candles), index+4)) if candles[i].high < price)
        else:
            price = candles[index].low
            # Check how many candles before and after are higher
            before_lower = sum(1 for i in range(max(0, index-3), index) if candles[i].low > price)
            after_lower = sum(1 for i in range(index+1, min(len(candles), index+4)) if candles[i].low > price)
        
        total_lower = before_lower + after_lower
        if total_lower >= 5:
            return 3
        elif total_lower >= 3:
            return 2
        return 1

    def _identify_liquidity_zones(self, candles: List[CandleData]) -> List[dict]:
        """Identify potential liquidity zones."""
        if len(candles) < 20:
            return []
            
        zones = []
        recent_candles = candles[-20:]
        
        # Find major support/resistance levels
        price_levels = []
        for candle in recent_candles:
            price_levels.extend([candle.high, candle.low])
        
        # Group nearby price levels
        grouped_levels = self._group_price_levels(price_levels)
        
        # Create liquidity zones
        for level, count in grouped_levels.items():
            if count >= 3:  # At least 3 touches to consider it a zone
                zones.append({
                    "type": "major",
                    "price": level,
                    "strength": min(3, count),
                    "description": f"Major {'support' if level < recent_candles[-1].close else 'resistance'} zone"
                })
        
        return zones

    def _group_price_levels(self, levels: List[float], threshold: float = 0.0001) -> dict:
        """Group nearby price levels together."""
        if not levels:
            return {}
            
        grouped = {}
        sorted_levels = sorted(levels)
        
        current_group = [sorted_levels[0]]
        current_avg = sorted_levels[0]
        
        for level in sorted_levels[1:]:
            if abs(level - current_avg) / current_avg <= threshold:
                current_group.append(level)
                current_avg = sum(current_group) / len(current_group)
            else:
                grouped[current_avg] = len(current_group)
                current_group = [level]
                current_avg = level
        
        if current_group:
            grouped[current_avg] = len(current_group)
        
        return grouped 