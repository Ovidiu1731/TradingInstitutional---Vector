from typing import List, Dict, Optional
from models.candle import CandleData

class SetupAnalysisService:
    """
    Analyzes trading setups based on Romanian institutional trading methodology.
    Focuses on MSS (Market Structure Shift), displacement, gaps, and setup classification.
    """
    
    def analyze_setup(self, candles: List[CandleData]) -> Dict:
        """
        Analyze the trading setup within the given timeframe.
        
        Returns:
        - MSS analysis (valid/aggressive, type)
        - Displacement detection
        - Gap analysis
        - Setup classification
        """
        if not candles or len(candles) < 5:
            return {
                "error": "Insufficient data for analysis. Need at least 5 candles.",
                "mss": None,
                "displacement": None,
                "gaps": None,
                "setup": None
            }
        
        # Sort candles by timestamp
        sorted_candles = sorted(candles, key=lambda x: x.date)
        
        # Analyze components
        mss_analysis = self._analyze_mss(sorted_candles)
        displacement_analysis = self._analyze_displacement(sorted_candles)
        gap_analysis = self._analyze_gaps(sorted_candles)
        setup_classification = self._classify_setup(gap_analysis, mss_analysis, displacement_analysis)
        
        return {
            "mss": mss_analysis,
            "displacement": displacement_analysis,
            "gaps": gap_analysis,
            "setup": setup_classification,
            "timeframe_coverage": f"{sorted_candles[0].date} to {sorted_candles[-1].date}"
        }
    
    def _analyze_mss(self, candles: List[CandleData]) -> Dict:
        """Analyze Market Structure Shift using the 2-candle validation rule."""
        if len(candles) < 5:
            return {
                "detected": False,
                "type": None,
                "validity": "insufficient_data",
                "reason": "Need at least 5 candles for MSS analysis"
            }
        
        # Find valid highs and lows
        valid_highs = self._find_valid_highs(candles)
        valid_lows = self._find_valid_lows(candles)
        
        # Analyze structure changes
        mss_uptrend_to_downtrend = self._detect_uptrend_to_downtrend_mss(candles, valid_highs, valid_lows)
        mss_downtrend_to_uptrend = self._detect_downtrend_to_uptrend_mss(candles, valid_highs, valid_lows)
        
        if mss_uptrend_to_downtrend:
            return mss_uptrend_to_downtrend
        elif mss_downtrend_to_uptrend:
            return mss_downtrend_to_uptrend
        else:
            # Check for aggressive MSS
            aggressive_mss = self._detect_aggressive_mss(candles)
            if aggressive_mss:
                return aggressive_mss
            
            return {
                "detected": False,
                "type": None,
                "validity": "no_mss",
                "reason": "No valid or aggressive MSS detected in timeframe"
            }
    
    def _find_valid_highs(self, candles: List[CandleData]) -> List[Dict]:
        """Find valid highs using 2-candle rule: min 2 green before + min 2 red after."""
        valid_highs = []
        
        for i in range(2, len(candles) - 2):
            current_high = candles[i].high
            
            # Check if this is a local high
            if not (current_high > candles[i-1].high and current_high > candles[i+1].high):
                continue
            
            # Count green candles before (minimum 2)
            green_before = 0
            for j in range(i-1, -1, -1):
                if self._is_bullish_candle(candles[j]):
                    green_before += 1
                if green_before >= 2:
                    break
            
            # Count red candles after (minimum 2)
            red_after = 0
            for j in range(i+1, len(candles)):
                if self._is_bearish_candle(candles[j]):
                    red_after += 1
                if red_after >= 2:
                    break
            
            if green_before >= 2 and red_after >= 2:
                valid_highs.append({
                    "index": i,
                    "price": current_high,
                    "timestamp": candles[i].date,
                    "green_before": green_before,
                    "red_after": red_after
                })
        
        return valid_highs
    
    def _find_valid_lows(self, candles: List[CandleData]) -> List[Dict]:
        """Find valid lows using 2-candle rule: min 2 red before + min 2 green after."""
        valid_lows = []
        
        for i in range(2, len(candles) - 2):
            current_low = candles[i].low
            
            # Check if this is a local low
            if not (current_low < candles[i-1].low and current_low < candles[i+1].low):
                continue
            
            # Count red candles before (minimum 2)
            red_before = 0
            for j in range(i-1, -1, -1):
                if self._is_bearish_candle(candles[j]):
                    red_before += 1
                if red_before >= 2:
                    break
            
            # Count green candles after (minimum 2)
            green_after = 0
            for j in range(i+1, len(candles)):
                if self._is_bullish_candle(candles[j]):
                    green_after += 1
                if green_after >= 2:
                    break
            
            if red_before >= 2 and green_after >= 2:
                valid_lows.append({
                    "index": i,
                    "price": current_low,
                    "timestamp": candles[i].date,
                    "red_before": red_before,
                    "green_after": green_after
                })
        
        return valid_lows
    
    def _is_bullish_candle(self, candle: CandleData) -> bool:
        """Check if candle is bullish (green)."""
        return candle.close > candle.open
    
    def _is_bearish_candle(self, candle: CandleData) -> bool:
        """Check if candle is bearish (red)."""
        return candle.close < candle.open
    
    def _detect_uptrend_to_downtrend_mss(self, candles: List[CandleData], valid_highs: List[Dict], valid_lows: List[Dict]) -> Optional[Dict]:
        """Detect MSS from uptrend to downtrend (breaking last higher low)."""
        if len(valid_highs) < 2 or len(valid_lows) < 1:
            return None
        
        # Look for higher high, higher low pattern
        for i in range(len(valid_lows)):
            low = valid_lows[i]
            
            # Find if this low is broken (price goes below it)
            for j in range(low["index"] + 1, len(candles)):
                if candles[j].low < low["price"]:
                    return {
                        "detected": True,
                        "type": "uptrend_to_downtrend",
                        "validity": "valid",
                        "broken_level": low["price"],
                        "break_timestamp": candles[j].date,
                        "reason": f"Last higher low at {low['price']:.5f} broken"
                    }
        
        return None
    
    def _detect_downtrend_to_uptrend_mss(self, candles: List[CandleData], valid_highs: List[Dict], valid_lows: List[Dict]) -> Optional[Dict]:
        """Detect MSS from downtrend to uptrend (breaking last lower high)."""
        if len(valid_lows) < 2 or len(valid_highs) < 1:
            return None
        
        # Look for lower low, lower high pattern
        for i in range(len(valid_highs)):
            high = valid_highs[i]
            
            # Find if this high is broken (price goes above it)
            for j in range(high["index"] + 1, len(candles)):
                if candles[j].high > high["price"]:
                    return {
                        "detected": True,
                        "type": "downtrend_to_uptrend",
                        "validity": "valid",
                        "broken_level": high["price"],
                        "break_timestamp": candles[j].date,
                        "reason": f"Last lower high at {high['price']:.5f} broken"
                    }
        
        return None
    
    def _detect_aggressive_mss(self, candles: List[CandleData]) -> Optional[Dict]:
        """Detect aggressive MSS using invalid highs/lows."""
        # Find local highs/lows without strict validation
        local_highs = []
        local_lows = []
        
        for i in range(1, len(candles) - 1):
            if candles[i].high > candles[i-1].high and candles[i].high > candles[i+1].high:
                local_highs.append({"index": i, "price": candles[i].high})
            
            if candles[i].low < candles[i-1].low and candles[i].low < candles[i+1].low:
                local_lows.append({"index": i, "price": candles[i].low})
        
        # Check for breaks of these levels
        for low in local_lows:
            for j in range(low["index"] + 1, len(candles)):
                if candles[j].low < low["price"]:
                    return {
                        "detected": True,
                        "type": "uptrend_to_downtrend",
                        "validity": "aggressive",
                        "broken_level": low["price"],
                        "break_timestamp": candles[j].date,
                        "reason": f"Aggressive MSS - invalid low at {low['price']:.5f} broken"
                    }
        
        for high in local_highs:
            for j in range(high["index"] + 1, len(candles)):
                if candles[j].high > high["price"]:
                    return {
                        "detected": True,
                        "type": "downtrend_to_uptrend",
                        "validity": "aggressive",
                        "broken_level": high["price"],
                        "break_timestamp": candles[j].date,
                        "reason": f"Aggressive MSS - invalid high at {high['price']:.5f} broken"
                    }
        
        return None
    
    def _analyze_displacement(self, candles: List[CandleData]) -> Dict:
        """Analyze displacement - rapid directional movement."""
        if len(candles) < 3:
            return {
                "detected": False,
                "reason": "Insufficient data for displacement analysis"
            }
        
        displacements = []
        
        # Look for rapid price movements (consecutive candles in same direction)
        for i in range(len(candles) - 2):
            # Check for bullish displacement (3+ consecutive green candles)
            if (self._is_bullish_candle(candles[i]) and 
                self._is_bullish_candle(candles[i+1]) and 
                self._is_bullish_candle(candles[i+2])):
                
                # Calculate movement strength
                start_price = candles[i].open
                end_price = candles[i+2].close
                movement_percent = ((end_price - start_price) / start_price) * 100
                
                if abs(movement_percent) > 0.1:  # Significant movement
                    displacements.append({
                        "start_index": i,
                        "end_index": i+2,
                        "direction": "bullish",
                        "start_price": start_price,
                        "end_price": end_price,
                        "movement_percent": movement_percent,
                        "start_time": candles[i].date,
                        "end_time": candles[i+2].date
                    })
            
            # Check for bearish displacement (3+ consecutive red candles)
            elif (self._is_bearish_candle(candles[i]) and 
                  self._is_bearish_candle(candles[i+1]) and 
                  self._is_bearish_candle(candles[i+2])):
                
                start_price = candles[i].open
                end_price = candles[i+2].close
                movement_percent = ((end_price - start_price) / start_price) * 100
                
                if abs(movement_percent) > 0.1:  # Significant movement
                    displacements.append({
                        "start_index": i,
                        "end_index": i+2,
                        "direction": "bearish",
                        "start_price": start_price,
                        "end_price": end_price,
                        "movement_percent": movement_percent,
                        "start_time": candles[i].date,
                        "end_time": candles[i+2].date
                    })
        
        if displacements:
            return {
                "detected": True,
                "count": len(displacements),
                "movements": displacements
            }
        else:
            return {
                "detected": False,
                "reason": "No significant displacement detected"
            }
    
    def _analyze_gaps(self, candles: List[CandleData]) -> Dict:
        """Analyze institutional gaps within displacement."""
        if len(candles) < 2:
            return {
                "detected": False,
                "count": 0,
                "reason": "Insufficient data for gap analysis"
            }
        
        gaps = []
        
        for i in range(len(candles) - 1):
            current_candle = candles[i]
            next_candle = candles[i + 1]
            
            # Check for bullish gap (gap up)
            if next_candle.low > current_candle.high:
                gap_size = next_candle.low - current_candle.high
                gaps.append({
                    "type": "bullish_gap",
                    "start_index": i,
                    "end_index": i + 1,
                    "gap_start": current_candle.high,
                    "gap_end": next_candle.low,
                    "gap_size": gap_size,
                    "start_time": current_candle.date,
                    "end_time": next_candle.date
                })
            
            # Check for bearish gap (gap down)
            elif next_candle.high < current_candle.low:
                gap_size = current_candle.low - next_candle.high
                gaps.append({
                    "type": "bearish_gap",
                    "start_index": i,
                    "end_index": i + 1,
                    "gap_start": current_candle.low,
                    "gap_end": next_candle.high,
                    "gap_size": gap_size,
                    "start_time": current_candle.date,
                    "end_time": next_candle.date
                })
        
        return {
            "detected": len(gaps) > 0,
            "count": len(gaps),
            "gaps": gaps
        }
    
    def _classify_setup(self, gap_analysis: Dict, mss_analysis: Dict, displacement_analysis: Dict) -> Dict:
        """Classify the trading setup based on gaps, MSS, and displacement."""
        if not gap_analysis["detected"] or not mss_analysis["detected"]:
            return {
                "type": "invalid",
                "reason": "Missing required components (MSS and/or gaps)"
            }
        
        gap_count = gap_analysis["count"]
        gaps = gap_analysis.get("gaps", [])
        
        # Check if gaps are consecutive
        consecutive_gaps = self._are_gaps_consecutive(gaps) if len(gaps) > 1 else False
        
        # Classify based on gap count and structure
        if gap_count == 1:
            return {
                "type": "OSG",
                "name": "One Simple Gap Setup",
                "description": "Single gap after MSS and displacement"
            }
        
        elif gap_count == 2:
            if consecutive_gaps:
                return {
                    "type": "TCG",
                    "name": "Two Consecutive Gaps Setup",
                    "description": "Two consecutive gaps - highest win rate setup"
                }
            else:
                return {
                    "type": "TG", 
                    "name": "Two Gap Setup",
                    "description": "Two non-consecutive gaps"
                }
        
        elif gap_count == 3:
            if consecutive_gaps:
                return {
                    "type": "3CG",
                    "name": "Three Consecutive Gaps Setup",
                    "description": "Three consecutive gaps - execute in middle gap"
                }
            else:
                return {
                    "type": "3G",
                    "name": "Three Gaps Setup", 
                    "description": "Three non-consecutive gaps"
                }
        
        elif gap_count > 3:
            return {
                "type": "MG",
                "name": "Multiple Gaps Setup",
                "description": "More than 3 gaps - not recommended for execution"
            }
        
        else:
            return {
                "type": "unknown",
                "reason": f"Unclassified setup with {gap_count} gaps"
            }
    
    def _are_gaps_consecutive(self, gaps: List[Dict]) -> bool:
        """Check if gaps are consecutive (one after another)."""
        if len(gaps) < 2:
            return False
        
        # Sort gaps by start index
        sorted_gaps = sorted(gaps, key=lambda x: x["start_index"])
        
        # Check if each gap's end_index + 1 equals next gap's start_index
        for i in range(len(sorted_gaps) - 1):
            if sorted_gaps[i]["end_index"] != sorted_gaps[i+1]["start_index"]:
                return False
        
        return True
    
    def format_analysis_output(self, analysis: Dict) -> str:
        """Format the analysis output in a concise, readable format."""
        if "error" in analysis:
            return f"❌ Error: {analysis['error']}"
        
        output = []
        
        # MSS Analysis
        mss = analysis["mss"]
        if mss["detected"]:
            validity = "✅ Valid" if mss["validity"] == "valid" else "⚠️ Aggressive"
            output.append(f"MSS: {validity} - {mss['type'].replace('_', ' → ')}")
            if mss.get("broken_level"):
                output.append(f"   Level broken: {mss['broken_level']:.5f}")
        else:
            output.append(f"MSS: ❌ Not detected - {mss.get('reason', 'Unknown')}")
        
        # Displacement Analysis
        displacement = analysis["displacement"]
        if displacement["detected"]:
            movements = displacement.get("movements", [])
            if movements:
                direction = movements[0]["direction"]
                output.append(f"Displacement: ✅ {direction.title()} displacement detected")
        else:
            output.append(f"Displacement: ❌ Not detected")
        
        # Gap Analysis
        gaps = analysis["gaps"]
        if gaps["detected"]:
            output.append(f"Gaps: ✅ {gaps['count']} gap(s) identified")
        else:
            output.append(f"Gaps: ❌ No gaps detected")
        
        # Setup Classification
        setup = analysis["setup"]
        if setup.get("type") != "invalid":
            output.append(f"Setup: ✅ {setup['type']} - {setup.get('name', 'Unknown')}")
            if setup.get("description"):
                output.append(f"   {setup['description']}")
        else:
            output.append(f"Setup: ❌ Invalid - {setup.get('reason', 'Unknown')}")
        
        return "\n".join(output) 