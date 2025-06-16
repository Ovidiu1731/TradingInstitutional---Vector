from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from datetime import datetime
from .candle import MarketStructure

class AssistantContract(BaseModel):
    analysis_possible: bool = True
    final_mss_type: Optional[str] = None
    final_trade_direction: Optional[str] = None
    fvg_analysis: dict = Field(default_factory=dict)
    liquidity_status_suggestion: str | None = None
    direction_confidence: str | None = None
    timestamp: datetime | None = None
    valid_structure_points: List[dict] = []
    setup_type: str | None = None
    setup_validity_score: float | None = None
    setup_quality_summary: str | None = None
    liquidity_zones: List[dict] = []
    notes: str | None = None

    @classmethod
    def from_market_structure(cls, ms: MarketStructure) -> "AssistantContract":
        """Convert MarketStructure to AssistantContract format"""
        # Count institutional gaps
        fvg_cnt = len(ms.institutional_gaps)
        
        # Infer trade direction
        final_trade_direction = _infer_direction(ms)
        
        # Get liquidity status
        liquidity_status = _liquidity_status(ms)
        
        # Calculate setup validity score (simple heuristic)
        setup_validity_score = _calculate_setup_validity(ms)
        
        # Generate setup quality summary
        setup_quality_summary = _generate_setup_summary(ms)
        
        return cls(
            final_mss_type=ms.mss_type,
            final_trade_direction=final_trade_direction,
            fvg_analysis={
                "count": fvg_cnt,
                "description": _describe_gaps(ms.institutional_gaps),
            },
            liquidity_status_suggestion=liquidity_status,
            direction_confidence=_calculate_direction_confidence(ms, final_trade_direction),
            timestamp=ms.timestamp,
            valid_structure_points=ms.valid_structure_points,
            setup_type=ms.setup_type,
            setup_validity_score=setup_validity_score,
            setup_quality_summary=setup_quality_summary,
            liquidity_zones=ms.liquidity_zones,
            notes=ms.notes
        )

def _infer_direction(ms: MarketStructure) -> str:
    """Infer trade direction from market structure"""
    if ms.setup_type in {"long_after_sweep", "OG"}:
        return "long"
    if ms.setup_type in {"short_after_sweep"}:
        return "short"
    
    # Fallback: compare last two structure points
    if len(ms.valid_structure_points) >= 2:
        last_two = ms.valid_structure_points[-2:]
        if last_two[0]["type"] == "low" and last_two[1]["type"] == "high":
            return "long"
        if last_two[0]["type"] == "high" and last_two[1]["type"] == "low":
            return "short"
    
    return "unknown"

def _describe_gaps(gaps: List[dict]) -> str:
    """Generate a human-readable description of the gaps"""
    if not gaps:
        return "No institutional gaps detected"
    
    gap_types = {}
    for gap in gaps:
        gap_type = gap["type"]
        gap_types[gap_type] = gap_types.get(gap_type, 0) + 1
    
    descriptions = []
    for gap_type, count in gap_types.items():
        descriptions.append(f"{count} {gap_type.replace('_', ' ')}")
    
    return f"Found {len(gaps)} gaps: {', '.join(descriptions)}"

def _liquidity_status(ms: MarketStructure) -> str:
    """Generate liquidity status suggestion"""
    if not ms.liquidity_zones:
        return "No significant liquidity zones detected"
    
    major_zones = [z for z in ms.liquidity_zones if z["type"] == "major"]
    local_zones = [z for z in ms.liquidity_zones if z["type"] == "local"]
    
    if major_zones:
        return f"Strong liquidity at {len(major_zones)} major levels"
    elif local_zones:
        return f"Local liquidity at {len(local_zones)} levels"
    return "Limited liquidity detected"

def _calculate_setup_validity(ms: MarketStructure) -> float:
    """Calculate a validity score for the setup (0.0 to 1.0)"""
    score = 0.0
    
    # Base score from MSS
    if ms.mss_detected:
        score += 0.3
        if "Aggressive" not in str(ms.mss_type):
            score += 0.1
    
    # Add score for liquidity sweeps
    if ms.hod_swept or ms.lod_swept:
        score += 0.2
    
    # Add score for valid gaps
    if ms.valid_gap_detected:
        score += 0.2
    
    # Add score for clear setup type
    if ms.setup_type:
        score += 0.2
    
    return min(score, 1.0)

def _generate_setup_summary(ms: MarketStructure) -> str:
    """Generate a human-readable summary of the setup quality"""
    if not ms.setup_type:
        return "No clear setup detected"
    
    parts = []
    
    # Add MSS info
    if ms.mss_detected:
        parts.append(f"MSS: {ms.mss_type}")
    
    # Add gap info
    if ms.valid_gap_detected:
        parts.append(f"Gaps: {len(ms.institutional_gaps)} valid")
    
    # Add liquidity info
    if ms.hod_swept or ms.lod_swept:
        parts.append("Liquidity swept")
    
    return " | ".join(parts)

def _calculate_direction_confidence(ms: MarketStructure, trade_direction: str) -> str:
    """Calculate confidence level in the trade direction"""
    if not trade_direction or trade_direction == "unknown":
        return "low"
    
    # Count confirming factors
    confirming_factors = 0
    
    # MSS alignment
    if ms.mss_type:
        if (ms.mss_type == "bullish_mss" and trade_direction == "long") or \
           (ms.mss_type == "bearish_mss" and trade_direction == "short"):
            confirming_factors += 1
    
    # Setup type alignment
    if ms.setup_type:
        if (ms.setup_type in {"long_after_sweep", "OG"} and trade_direction == "long") or \
           (ms.setup_type == "short_after_sweep" and trade_direction == "short"):
            confirming_factors += 1
    
    # Liquidity sweep alignment
    if (ms.hod_swept and trade_direction == "short") or \
       (ms.lod_swept and trade_direction == "long"):
        confirming_factors += 1
    
    # Gap alignment
    if ms.valid_gap_detected:
        confirming_factors += 1
    
    # Determine confidence level
    if confirming_factors >= 3:
        return "high"
    elif confirming_factors >= 2:
        return "medium"
    return "low" 