# code
import os
import re
import json
import logging
import time
import copy # Added for deepcopy
from io import BytesIO
from typing import Dict, Any, Optional, List, Union

import requests
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import threading
from collections import deque # Efficient for fixed-size history
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI, RateLimitError, APIError
from pinecone import Pinecone, PineconeException
# --- Conversation History Store (In-Memory) ---
# Stores recent messages for each session_id
# Using deque for automatic size limiting
conversation_history: Dict[str, deque] = {}
history_lock = threading.Lock()
MAX_HISTORY_TURNS = 3 # Number of User/Assistant turn pairs to keep (e.g., 3 turns = 6 messages)
MAX_HISTORY_MESSAGES = MAX_HISTORY_TURNS * 2

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
# REVIEW: Logging setup looks standard and good.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# ENVIRONMENT & GLOBALS
# ---------------------------------------------------------------------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trading-lessons")
FEEDBACK_LOG = os.getenv("FEEDBACK_LOG", "feedback_log.jsonl")

# --- Few-Shot Examples for Vision Model ---
# REVIEW: List definition looks good. Using GitHub raw URLs is a good choice for stability.
# REVIEW: Assumes the JSON strings inside triple quotes are the final, correct versions we created.
# REVIEW: 7 examples included as discussed.
FEW_SHOT_EXAMPLES = [
    # --- Example 1: DE30EUR Aggressive Short ---
    {
        "image_url": "https://raw.githubusercontent.com/Ovidiu1731/Trade-images/main/DE30EUR_2025-05-05_12-29-24_69c08.png",
        "assistant_json_output": """
{
  "analysis_possible": true,
  "candle_colors": "Bullish candles are solid green, Bearish candles are solid white.",
  "is_risk_above_price": true,
  "trade_direction": "short",
  "mss_pivot_analysis": {
    "description": "MSS breaks the preceding higher low (pivot point). This pivot appears formed primarily by one bullish (green) candle.",
    "pivot_bearish_count": 0,
    "pivot_bullish_count": 1,
    "has_minimum_structure": false
  },
  "mss_type": "agresiv",
  "break_direction": "downward",
  "displacement_analysis": {
    "direction": "bearish",
    "strength": "moderate"
  },
  "fvg_analysis": { // Using new structure
    "count": 2, Updated count
    "description": "Two FVGs are visible after the MSS: one larger gap marked by a grey box during the initial displacement, and a smaller subsequent gap above it. (This aligns with a TG - Two Gap setup).",
},
  "liquidity_zones": "Multiple liquidity levels above prior highs (marked by horizontal white lines) were swept before the MSS occurred. The last key high swept appears to be around the 23,255-23,260 price level.",
  "liquidity_status": "swept",
  "trade_outcome": "breakeven",
  "visible_labels": ["MSS", "BE"]
}
"""
    },

    # --- Example 2: Re-entry Normal Long (07.18.15 copy.jpg) ---
    {
        "image_url": "https://raw.githubusercontent.com/Ovidiu1731/Trade-images/main/Screenshot%202025-05-05%20at%2007.18.15%20copy.png",
        "assistant_json_output": """
{
  "analysis_possible": true,
  "candle_colors": "Bullish candles have a solid white body, Bearish candles have a solid black body.",
  "is_risk_above_price": null,
  "trade_direction": "long",
  "mss_pivot_analysis": {
    "description": "MSS breaks the lower high that formed after a re-sweep of sell-side liquidity near the 'Local' marked low.",
    "pivot_bearish_count": 2,
    "pivot_bullish_count": 3,
    "has_minimum_structure": true
  },
  "mss_type": "normal",
  "break_direction": "upward",
  "displacement_analysis": {
    "direction": "bullish",
    "strength": "strong"
  },
  "fvg_analysis": { // Using new structure
    "count": 2, Updated count
    "description": "Yes, two distinct FVGs (marked with blue boxes) were created during the bullish displacement after the MSS.",
},
  "liquidity_zones": "Initial liquidity sweep occurred below the low marked 'LLB' (Lichiditate Locala Buy). Subsequently, price re-swept liquidity near the low marked 'Local' just before the MSS formed (re-entry pattern).",
  "liquidity_status": "swept",
  "trade_outcome": "potential_setup",
  "visible_labels": ["MSS", "Local", "LLB"]
}
"""
    },

    # --- Example 3: Normal Short (11.02.20.jpg) ---
    {
        "image_url": "https://raw.githubusercontent.com/Ovidiu1731/Trade-images/main/Screenshot%202025-05-05%20at%2011.02.20.png",
        "assistant_json_output": """
{
  "analysis_possible": true,
  "candle_colors": "Bullish candles have a solid white body, Bearish candles have a solid black body.",
  "is_risk_above_price": true,
  "trade_direction": "short",
  "mss_pivot_analysis": {
    "description": "Downward structure break (marked by arrow/line) occurs below the prior higher low (marked by small blue circle), following a sweep of the 'LLS' high. The pivot structure (higher low) contains multiple candles.",
    "pivot_bearish_count": 5,
    "pivot_bullish_count": 4,
    "has_minimum_structure": true
  },
  "mss_type": "normal",
  "break_direction": "downward",
  "displacement_analysis": {
    "direction": "bearish",
    "strength": "moderate"
  },
  "fvg_analysis": { // Using new structure
    "count": 1, //Updated count
    "description": "Yes, a FVG (marked by a blue box) was created during the bearish displacement, providing a potential entry area.",
},
  "liquidity_zones": "Liquidity above the prior swing high (marked 'LLS' - Lichiditate Locala Sell) was swept before the downward structure break occurred.",
  "liquidity_status": "swept",
  "trade_outcome": "loss",
  "visible_labels": ["LLS", "BE"]
}
"""
    },

    # --- Example 4: Normal Short SLG+TCG (11.04.14.jpg) ---
    {
        "image_url": "https://raw.githubusercontent.com/Ovidiu1731/Trade-images/main/Screenshot%202025-05-05%20at%2011.04.14.png",
        "assistant_json_output": """
{
  "analysis_possible": true,
  "candle_colors": "Bullish candles have a solid white body, Bearish candles have a solid black body.",
  "is_risk_above_price": true,
  "trade_direction": "short",
  "mss_pivot_analysis": {
    "description": "Downward structure break (marked by black line/arrow) occurs below the prior higher low, following a sweep above the 'LMS' high.",
    "pivot_bearish_count": 4,
    "pivot_bullish_count": 4,
    "has_minimum_structure": true
  },
  "mss_type": "normal",
  "break_direction": "downward",
  "displacement_analysis": {
    "direction": "bearish",
    "strength": "moderate"
  },
  "fvg_analysis":  { // Using new structure
    "count": 2, //Updated count
    "description": "Yes, two FVGs (marked by faint blue rectangles) are visible after the MSS. The pattern of a new high before MSS + two gaps after fits the 'SLG + TCG' setup.",
},
  "liquidity_zones": "Liquidity above the prior swing high (marked 'LMS' - Lichiditate Majora Sell) was swept before the MSS occurred.",
  "liquidity_status": "swept",
  "trade_outcome": "loss",
  "visible_labels": ["LMS"]
}
"""
    },

    # --- Example 5: Normal Short SLG+3G (11.01.39.jpg) ---
    {
        "image_url": "https://github.com/Ovidiu1731/Trade-images/raw/main/Screenshot%202025-05-05%20at%2011.01.39.png",
        "assistant_json_output": """
{
  "analysis_possible": true,
  "candle_colors": "Bullish candles have a solid white body, Bearish candles have a solid black body.",
  "is_risk_above_price": true,
  "trade_direction": "short",
  "mss_pivot_analysis": {
    "description": "Downward structure break occurs below the prior higher low (marked by dark blue circle), following a sweep above the high marked 'LLS'.",
    "pivot_bearish_count": 2,
    "pivot_bullish_count": 3,
    "has_minimum_structure": true
  },
  "mss_type": "normal",
  "break_direction": "downward",
  "displacement_analysis": {
    "direction": "bearish",
    "strength": "strong"
  },
  "fvg_analysis": { // Using new structure
    "count": 3, //Updated Count
    "description": "Yes, three FVGs (marked by dark blue rectangles) were created within the strong bearish displacement move. The pattern of a new valid high before the MSS plus these three gaps fits the 'SLG + 3G' setup.",
},
  "liquidity_zones": "Liquidity above the prior swing high (marked 'LLS' - Lichiditate Locala Sell) was swept before the downward structure break occurred.",
  "liquidity_status": "swept",
  "trade_outcome": "win",
  "visible_labels": ["LLS"]
}
"""
    },

    # --- Example 6: Normal Short TCG (11.04.35.jpg) ---
    {
        "image_url": "https://github.com/Ovidiu1731/Trade-images/raw/main/Screenshot%202025-05-05%20at%2011.04.35.png",
        "assistant_json_output": """
{
  "analysis_possible": true,
  "candle_colors": "Bullish candles have a solid white body, Bearish candles have a solid black body.",
  "is_risk_above_price": true,
  "trade_direction": "short",
  "mss_pivot_analysis": {
    "description": "Downward structure break (marked by arrow) occurs below the prior higher low (marked by small orange circle), following a sweep above the 'Liq Locala' high.",
    "pivot_bearish_count": 3,
    "pivot_bullish_count": 3,
    "has_minimum_structure": true
  },
  "mss_type": "normal",
  "break_direction": "downward",
  "displacement_analysis": {
    "direction": "bearish",
    "strength": "strong"
  },
  "fvg_analysis": { // Using new structure
    "count": 2,
    "description": "Yes, two FVGs (marked by faint orange rectangles/lines) were created after the MSS. This aligns with a TCG (Two Consecutive Gaps) setup.",
},
  "liquidity_zones": "Liquidity above the prior swing high (marked 'Liq Locala') was swept before the downward move began.",
  "liquidity_status": "swept",
  "trade_outcome": "win",
  "visible_labels": ["Liq Locala", "BE"]
}
"""
    },

      # --- Example 7: 202025-05-05%20at%2007.22.35.png ---
    {
        "image_url": "https://raw.githubusercontent.com/Ovidiu1731/Trade-images/main/Screenshot%202025-05-05%20at%2007.22.35.png",
        "assistant_json_output": """
{
  "analysis_possible": true,
  "candle_colors": "Bullish candles have a solid white body, Bearish candles have a solid black body.", // Updated colors
  "is_risk_above_price": false,
  "trade_direction": "long",
  "mss_pivot_analysis": {
    "description": "Upward structure break (marked 'MSS' text and line/arrow) occurs above the prior lower high, following a sweep below the 'LLB' low.", // Updated Liq reference
    "pivot_bearish_count": 4, // Updated count
    "pivot_bullish_count": 6, // Updated count
    "has_minimum_structure": true // Stays true (4>=2 and 6>=2)
  },
  "mss_type": "normal", // Stays normal
  "break_direction": "upward",
  "displacement_analysis": {
    "direction": "bullish",
    "strength": "moderate"
  },
  "fvg_analysis": { // Using new structure
    "count": 2, // Updated count
    "description": "Yes, two FVGs appear to be created within the bullish displacement following the MSS, both marked by blue boxes." // Updated description
  },
  "liquidity_zones": "Liquidity was swept (marked by LLB) and then the buy entry was formed by the MSS and the displacement that created 2 fair value gaps", // Updated label reference
  "liquidity_status": "swept",
  "trade_outcome": "loss", // Updated outcome
  "visible_labels": [ // Updated labels
    "MSS",
    "LLB"
  ]
}
"""
    },

     # --- Example 8: Normal Long Tricky Colors (Screenshot_2.png) ---
     {
        "image_url": "https://raw.githubusercontent.com/Ovidiu1731/Trade-images/main/Screenshot_2.png",
        "assistant_json_output": """
{
  "analysis_possible": true,
  "candle_colors": "Bullish candles appear to have a solid dark gray body, Bearish candles have a solid black body (distinction can be difficult).",
  "is_risk_above_price": false,
  "trade_direction": "long",
  "mss_pivot_analysis": {
    "description": "Upward structure break (marked 'MSS' text and line/arrow) occurs above the prior lower high, following a sweep below the 'Liq Locala' low. The blue box marks an FVG, not the MSS break itself.",
    "pivot_bearish_count": 2,
    "pivot_bullish_count": 4,
    "has_minimum_structure": true
  },
  "mss_type": "normal", // Stays normal
  "break_direction": "upward",
  "displacement_analysis": {
    "direction": "bullish",
    "strength": "moderate"
  },
  "fvg_analysis": { // Using new structure
    "count": 2,
    "description": "Yes, two visual FVGs (imbalances, marked by the blue boxes) appear to be created within the bullish displacement following the MSS, near the entry area."
  },
  "liquidity_zones": "Liquidity below the prior swing low (marked 'Liq Locala') was swept before the MSS occurred.",
  "liquidity_status": "swept",
  "trade_outcome": "running",
  "visible_labels": [ // Updated labels
    "MSS", "LLB"]
 }
 """
     }
]


# --- Model selection - UPDATED ---
# REVIEW: Model selection looks fine.
EMBEDDING_MODEL = "text-embedding-ada-002"
VISION_MODEL = "gpt-4-turbo"
COMPLETION_MODEL = "gpt-4-turbo"
TEXT_MODEL = "gpt-3.5-turbo"

# REVIEW: API Key checks are good.
if not (OPENAI_API_KEY and PINECONE_API_KEY):
    logging.error("Missing OpenAI or Pinecone API key(s) in environment variables.")
    raise ValueError("Missing OpenAI or Pinecone API key(s)")

# Load core system prompt
# REVIEW: Loading prompt from file is good practice. Fallback is okay.
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT_CORE = f.read().strip()
except FileNotFoundError:
    logging.warning("system_prompt.txt not found. Using fallback system prompt.")
    SYSTEM_PROMPT_CORE = (
        "You are an AI assistant trained by Rareș for the Trading Instituțional community. "
        "Answer questions strictly based on the provided course material and visual analysis (if available). "
        "Adopt the persona of a helpful, slightly more experienced trading colleague explaining the analysis clearly. Avoid overly robotic phrasing. "
        "Emulate Rareș's direct, concise teaching style. Be helpful and accurate according to the course rules."
        # Added note for final LLM regarding validated analysis
        "\n\nIMPORTANT: When reviewing the 'Visual Analysis Report', trust the provided 'mss_type' and 'trade_direction'. "
        "If a '_validator_note' field is present, it means these fields were adjusted by internal rules for accuracy."
    )

# Define the core structural definitions (Unchanged)
# REVIEW: These definitions are fine, used for text Qs mainly.
MSS_AGRESIV_STRUCTURAL_DEFINITION = "Definiție Structurală MSS Agresiv: Un MSS agresiv se produce atunci cand ultimul higher low sau lower high care este rupt (unde se produce shift-ul) nu are in structura sa minim 2 candele bearish cu 2 candele bullish."
MSS_NORMAL_STRUCTURAL_DEFINITION = "Definiție Structurală MSS Normal: Este o rupere de structură formată din două sau mai multe lumânări care fac low/high."
FVG_STRUCTURAL_DEFINITION = "Definiție Structurală FVG (Fair Value Gap): Este un gap (spațiu gol) între lumânări creat în momentul în care prețul face o mișcare impulsivă, lăsând o zonă netranzacționată."
DISPLACEMENT_DEFINITION = "Definiție Displacement: Este o mișcare continuă a prețului în aceeași direcție, după o structură invalidată, creând FVG-uri (Fair Value Gaps)."

# SDK clients
# REVIEW: SDK Initialization looks correct.
try:
    openai = OpenAI(api_key=OPENAI_API_KEY)
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    index = pinecone_client.Index(PINECONE_INDEX_NAME)
    logging.info(f"Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")
except PineconeException as e:
    logging.error(f"Failed to initialize Pinecone: {e}")
    raise
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    raise


# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------
# REVIEW: Standard FastAPI setup.
app = FastAPI(title="Trading Instituțional AI Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# FEEDBACK MECHANISM (Unchanged)
# ---------------------------------------------------------------------------
# REVIEW: Feedback mechanism looks good for collecting data.
def log_feedback(session_id: str, question: str, answer: str, feedback: str,
                 query_type: str, analysis_data: Optional[Dict] = None) -> bool:
    """
    Log user feedback to a JSONL file for later analysis.
    Returns True if logging was successful, False otherwise.
    """
    try:
        feedback_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "feedback": feedback,
            "query_type": query_type
        }

        if analysis_data:
            relevant_fields = [
                "trade_direction", "mss_type",
                "pivot_bearish_count", "pivot_bullish_count", # From mss_pivot_analysis
                "trend_direction", "direction_consistency_warning", "mss_consistency_warning",
                "is_risk_above_price", "_validator_note"
            ]
            analysis_extract = {}
            for k in relevant_fields:
                if k == "pivot_bearish_count":
                    analysis_extract[k] = analysis_data.get("mss_pivot_analysis", {}).get(k)
                elif k == "pivot_bullish_count":
                    analysis_extract[k] = analysis_data.get("mss_pivot_analysis", {}).get(k)
                elif k in analysis_data:
                     analysis_extract[k] = analysis_data.get(k)

            feedback_entry["analysis_data"] = analysis_extract

        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + "\n")
        logging.info(f"Feedback logged successfully: {feedback} for session {session_id}")
        return True
    except Exception as e:
        logging.error(f"Failed to log feedback: {e}")
        return False

class FeedbackModel(BaseModel):
    session_id: str
    question: str
    answer: str
    feedback: str  # "positive" or "negative"
    query_type: Optional[str] = "unknown"
    analysis_data: Optional[Dict] = None

@app.post("/feedback")
async def submit_feedback(feedback_data: FeedbackModel) -> Dict[str, str]:
    """Record user feedback about answer quality"""
    analysis_input = feedback_data.analysis_data or {}
    pivot_analysis = analysis_input.get("mss_pivot_analysis", {})
    analysis_input["pivot_bearish_count"] = pivot_analysis.get("pivot_bearish_count")
    analysis_input["pivot_bullish_count"] = pivot_analysis.get("pivot_bullish_count")

    success = log_feedback(
        feedback_data.session_id,
        feedback_data.question,
        feedback_data.answer,
        feedback_data.feedback,
        feedback_data.query_type,
        analysis_input
    )

    if success:
        return {"status": "success", "message": "Feedback înregistrat cu succes. Mulțumim!"}
    else:
        raise HTTPException(
            status_code=500,
            detail="Nu am putut înregistra feedback-ul. Te rugăm să încerci din nou mai târziu."
        )

# ---------------------------------------------------------------------------
# QUERY TYPE IDENTIFICATION (REVISED LOGIC)
# ---------------------------------------------------------------------------
# REVIEW: Using the revised logic we discussed. Looks correct.
def identify_query_type(question: str) -> Dict[str, Any]:
    """
    Identifies the type of query to guide appropriate analysis,
    prioritizing evaluation terms and handling multiple concepts.
    Returns a dictionary with query type flags.
    """
    question_lower = question.lower().strip()

    # Patterns for identification (Using your lists)
    liquidity_patterns = [
        "liq", "lichid", "lichidit", "sunt corect notate", "marchează", "marchea", "marcate"
    ]
    trend_patterns = [
        "trend", "trendul", "tendință", "tendinta"
    ]
    mss_classification_patterns = [
        "mss normal sau", "mss agresiv sau", "mss normal sau agresiv",
        "este un mss normal", "este un mss agresiv", "ce fel de mss",
        "este agresiv sau normal", "este normal sau agresiv", "tip de mss"
    ]
    displacement_patterns = [
        "displacement", "displace", "mișcare", "miscare", "impulsiv", "gap",
        "fvg", "fair value gap", "impulse", "continuitate"
    ]
    fvg_patterns = [
        "fvg", "fair value gap", "gap", "valoare", "spațiu", "spatiu", "gol"
    ]
    trade_evaluation_patterns = [
        "cum arata", "cum arată", "ce parere", "ce părere", "evalueaz", "analizeaz",
        "trade", "setup", "intrare", "valid", "corect", "rezultat"
    ]

    # --- REVISED LOGIC ---

    # 1. Check for explicit evaluation requests FIRST
    if any(p in question_lower for p in trade_evaluation_patterns):
        logging.info("Query identified as 'trade_evaluation' based on evaluation keywords.")
        return {
            "type": "trade_evaluation",
            "requires_full_analysis": True,
            "requires_mss_analysis": True,
            "requires_direction_analysis": True,
            "requires_color_analysis": True,
            "requires_fvg_within_displacement": True
        }

    # 2. If not evaluation, check for specific concepts mentioned
    concepts_found = []
    specific_element_patterns = {
        "liquidity": liquidity_patterns,
        "trend": trend_patterns,
        "mss_classification": mss_classification_patterns,
        "displacement": displacement_patterns,
        "fvg": fvg_patterns,
    }
    for element_type, patterns in specific_element_patterns.items():
        if any(p in question_lower for p in patterns):
            concepts_found.append(element_type)

    # 3. Classify based on number of concepts found
    if len(concepts_found) > 1:
        logging.info(f"Query identified as 'trade_evaluation' based on multiple concepts: {concepts_found}")
        return {
            "type": "trade_evaluation",
            "requires_full_analysis": True,
            "requires_mss_analysis": True,
            "requires_direction_analysis": True,
            "requires_color_analysis": True,
            "requires_fvg_within_displacement": True
        }
    elif len(concepts_found) == 1:
        element_type = concepts_found[0]
        logging.info(f"Query identified as specific element: '{element_type}'")
        is_mss_type = element_type == "mss_classification"
        is_direction_type = element_type in ["trend", "displacement"]
        is_fvg_type = element_type == "fvg"
        return {
            "type": element_type,
            "requires_full_analysis": False,
            "requires_mss_analysis": is_mss_type,
            "requires_direction_analysis": is_direction_type,
            "requires_color_analysis": True,
            "requires_fvg_within_displacement": is_fvg_type
        }
    else:
        logging.info("Query identified as 'general' (no specific keywords matched).")
        return {
            "type": "general",
            "requires_full_analysis": False,
            "requires_mss_analysis": False,
            "requires_direction_analysis": True,
            "requires_color_analysis": True,
            "requires_fvg_within_displacement": False
        }

# ---------------------------------------------------------------------------
# HELPERS (Unchanged logic, added logging)
# ---------------------------------------------------------------------------
# REVIEW: Helper functions unchanged and seem okay.
def extract_text_from_image(image_url: str) -> str:
    """Download an image and return ASCII-cleaned OCR text, or empty string on failure."""
    try:
        logging.info(f"Attempting OCR for image URL: {image_url}")
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        content_type = resp.headers.get('Content-Type', '').lower()
        img = Image.open(BytesIO(resp.content))
        text = pytesseract.image_to_string(img, lang="eng")
        cleaned_text = "".join(ch for ch in text if ord(ch) < 128).strip()
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        logging.info(f"OCR successful. Extracted text length: {len(cleaned_text)}")
        logging.debug(f"OCR Text (first 100 chars): {cleaned_text[:100]}")
        return cleaned_text
    except requests.exceptions.RequestException as err:
        logging.error(f"❌ OCR failed: Network error accessing image URL {image_url}: {err}")
        return ""
    except pytesseract.TesseractNotFoundError:
        logging.error("❌ OCR failed: pytesseract executable not found. Ensure it's installed and in PATH.")
        return ""
    except Exception as err:
        logging.exception(f"❌ OCR failed: Unexpected error processing image {image_url}: {err}")
        return ""


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON string from text that might contain markdown code blocks or other text."""
    logging.debug(f"Attempting to extract JSON from text: {text[:200]}...")
    json_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
    match = re.search(json_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        logging.info("JSON extracted from markdown code block.")
        try:
            json.loads(extracted)
            return extracted
        except json.JSONDecodeError:
            logging.warning("Text in markdown block wasn't valid JSON.")

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        potential_json = brace_match.group(0).strip()
        if potential_json.startswith("{") and potential_json.endswith("}") and '"' in potential_json:
            try:
                json.loads(potential_json)
                logging.info("Potential JSON object found directly in text and seems valid.")
                return potential_json
            except json.JSONDecodeError:
                logging.warning("Found brace-enclosed text, but it's not valid JSON.")
                pass

    logging.warning("Could not extract valid-looking JSON object from text.")
    return None

def generate_session_id() -> str:
    """Generate a unique session ID for tracking feedback"""
    timestamp = int(time.time())
    random_part = os.urandom(4).hex()
    return f"{timestamp}-{random_part}"

# ---------------------------------------------------------------------------
# ROUTES – TEXT ONLY (Unchanged)
# ---------------------------------------------------------------------------
# REVIEW: Text-only route looks fine, unchanged.
@app.post("/ask", response_model=Dict[str, str])
async def ask_question(request: Request) -> Dict[str, str]:
    """Handles text-only questions answered strictly from course material."""
    try:
        body = await request.json()
        question = body.get("question", "").strip()
        session_id = body.get("session_id")
        is_new_session = False
        if not session_id:
            session_id = generate_session_id()
            is_new_session = True
            logging.info(f"New session started: {session_id}")

        if not question:
                logging.warning("Received empty question in /ask request.")
                return {"answer": "Te rog să specifici o întrebare.", "session_id": session_id}

        logging.info(f"Received /ask request. Question: '{question[:100]}...', Session ID: {session_id}")
        # --- Retrieve History ---
        history_messages = []
        with history_lock:
            # Get the deque for the session, or create a new one
            session_deque = conversation_history.setdefault(session_id, deque(maxlen=MAX_HISTORY_MESSAGES))
            history_messages = list(session_deque) # Get current history as a list
            logging.debug(f"Retrieved {len(history_messages)} history messages for session {session_id}")
            question_lower = question.lower()
            is_mss_agresiv_text_q = question_lower == "ce este un mss agresiv"
            is_mss_normal_text_q = question_lower == "ce este un mss normal"
            is_fvg_text_q = question_lower in ["ce este un fvg", "ce este fvg", "ce este fair value gap"]
            is_displacement_text_q = question_lower in ["ce este displacement", "ce inseamna displacement"]

        # 1. Get Embedding
        try:
            emb_response = openai.embeddings.create(model=EMBEDDING_MODEL, input=[question])
            query_embedding = emb_response.data[0].embedding
            logging.info("Successfully generated embedding for the question.")
        except (APIError, RateLimitError) as e:
            logging.error(f"OpenAI Embedding API error: {e}")
            raise HTTPException(status_code=503, detail="Serviciul OpenAI (Embeddings) nu este disponibil momentan.")
        except Exception as e:
            logging.exception(f"Unexpected error during embedding generation: {e}")
            raise HTTPException(status_code=500, detail="A apărut o eroare la procesarea întrebării.")

        # 2. Query Pinecone
        context = ""
        try:
            results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
            matches = results.get("matches", [])
            context = "\n\n---\n\n".join(m["metadata"].get("text", "") for m in matches if m["metadata"].get("text")).strip()
            logging.info(f"Pinecone query returned {len(matches)} matches. Context length: {len(context)}")
            logging.debug(f"DEBUG TXT - Retrieved Course Context Content:\n---\n{context[:1000]}...\n---")

            if not context:
                logging.warning("Pinecone query returned no relevant context.")
                # Handle specific definition questions
                if is_mss_agresiv_text_q: return {"answer": MSS_AGRESIV_STRUCTURAL_DEFINITION.replace("Definiție Structurală MSS Agresiv: ", "") + " Dacă ești la început, este recomandat în program să nu-l folosești încă.", "session_id": session_id}
                if is_mss_normal_text_q: return {"answer": MSS_NORMAL_STRUCTURAL_DEFINITION.replace("Definiție Structurală MSS Normal: ", ""), "session_id": session_id}
                if is_fvg_text_q: return {"answer": FVG_STRUCTURAL_DEFINITION.replace("Definiție Structurală FVG (Fair Value Gap): ", ""), "session_id": session_id}
                if is_displacement_text_q: return {"answer": DISPLACEMENT_DEFINITION.replace("Definiție Displacement: ", ""), "session_id": session_id}
                return {"answer": "Nu am găsit informații relevante în materialele de curs pentru a răspunde la această întrebare.", "session_id": session_id}

            # Inject structural definition if question is exactly about specific concepts and context might be missing it
            definitions_to_inject = []
            if is_mss_agresiv_text_q and MSS_AGRESIV_STRUCTURAL_DEFINITION.lower() not in context.lower(): definitions_to_inject.append(MSS_AGRESIV_STRUCTURAL_DEFINITION)
            if is_mss_normal_text_q and MSS_NORMAL_STRUCTURAL_DEFINITION.lower() not in context.lower(): definitions_to_inject.append(MSS_NORMAL_STRUCTURAL_DEFINITION)
            if is_fvg_text_q and FVG_STRUCTURAL_DEFINITION.lower() not in context.lower(): definitions_to_inject.append(FVG_STRUCTURAL_DEFINITION)
            if is_displacement_text_q and DISPLACEMENT_DEFINITION.lower() not in context.lower(): definitions_to_inject.append(DISPLACEMENT_DEFINITION)

            if definitions_to_inject:
                definition_block = "\n\n".join(definitions_to_inject)
                context = f"{definition_block}\n\n---\n\n{context}"
                logging.info(f"Injected {len(definitions_to_inject)} definitions into context.")

        except PineconeException as e:
            logging.error(f"Pinecone query error: {e}")
            # Handle specific definition questions even if Pinecone fails
            if is_mss_agresiv_text_q: return {"answer": MSS_AGRESIV_STRUCTURAL_DEFINITION.replace("Definiție Structurală MSS Agresiv: ", "") + " Dacă ești la început, este recomandat în program să nu-l folosești încă.", "session_id": session_id}
            if is_mss_normal_text_q: return {"answer": MSS_NORMAL_STRUCTURAL_DEFINITION.replace("Definiție Structurală MSS Normal: ", ""), "session_id": session_id}
            if is_fvg_text_q: return {"answer": FVG_STRUCTURAL_DEFINITION.replace("Definiție Structurală FVG (Fair Value Gap): ", ""), "session_id": session_id}
            if is_displacement_text_q: return {"answer": DISPLACEMENT_DEFINITION.replace("Definiție Displacement: ", ""), "session_id": session_id}
            raise HTTPException(status_code=503, detail="Serviciul de căutare (Pinecone) nu este disponibil momentan.")
        except Exception as e:
            logging.exception(f"Unexpected error during Pinecone query: {e}")
            if is_mss_agresiv_text_q: return {"answer": MSS_AGRESIV_STRUCTURAL_DEFINITION.replace("Definiție Structurală MSS Agresiv: ", "") + " Dacă ești la început, este recomandat în program să nu-l folosești încă.", "session_id": session_id}
            if is_mss_normal_text_q: return {"answer": MSS_NORMAL_STRUCTURAL_DEFINITION.replace("Definiție Structurală MSS Normal: ", ""), "session_id": session_id}
            if is_fvg_text_q: return {"answer": FVG_STRUCTURAL_DEFINITION.replace("Definiție Structurală FVG (Fair Value Gap): ", ""), "session_id": session_id}
            if is_displacement_text_q: return {"answer": DISPLACEMENT_DEFINITION.replace("Definiție Displacement: ", ""), "session_id": session_id}
            raise HTTPException(status_code=500, detail="A apărut o eroare la căutarea informațiilor.")

        # 3. Generate Answer
        try:
            system_message = SYSTEM_PROMPT_CORE + "\n\nAnswer ONLY based on the provided Context and conversation history."
            
            # --- Prepare messages for TEXT_MODEL ---
            messages_for_llm = []
            messages_for_llm.append({"role": "system", "content": system_message})
            # Add history messages retrieved earlier
            messages_for_llm.extend(history_messages)
            # Add current user question + Pinecone context
            user_message_with_context = f"Question: {question}\n\nContext:\n{context}"
            messages_for_llm.append({"role": "user", "content": user_message_with_context})

            logging.debug(f"Sending to {TEXT_MODEL}. Message count: {len(messages_for_llm)}")
            response = openai.chat.completions.create(
                model=TEXT_MODEL,
                messages=messages_for_llm, # Pass history + current context
                temperature=0.3,
                max_tokens=300
            )
            answer = response.choices[0].message.content.strip()

            # --- Store Updated History ---
            with history_lock:
                 # session_deque was retrieved/created earlier
                 session_deque.append({"role": "user", "content": question}) # Store original question
                 session_deque.append({"role": "assistant", "content": answer}) # Store answer
                 # deque automatically handles maxlen trimming
                 logging.debug(f"Stored history for session {session_id}. New length: {len(session_deque)}")

            logging.info(f"Successfully generated answer using {TEXT_MODEL}.")
            return {
                "answer": answer,
                "session_id": session_id # Return session_id so client can use it for follow-ups
            }

        except (APIError, RateLimitError) as e:
            logging.error(f"OpenAI Chat API error ({TEXT_MODEL}): {e}")
            raise HTTPException(status_code=503, detail=f"Serviciul OpenAI ({TEXT_MODEL}) nu este disponibil momentan.")
        except Exception as e:
            logging.exception(f"Unexpected error during {TEXT_MODEL} answer generation: {e}")
            raise HTTPException(status_code=500, detail="A apărut o eroare la generarea răspunsului.")

    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Unhandled exception in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail="A apărut o eroare internă neașteptată.")

# ---------------------------------------------------------------------------
# --- SANITY CHECK VALIDATOR --- (Defined before use)
# ---------------------------------------------------------------------------
# REVIEW: Validator logic seems correct based on previous discussions.
def _sanity_check_analysis(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforces deterministic rules for MSS type and trade direction based on
    expected fields potentially extracted by the Vision model.
    Modifies the dictionary in place for simplicity, but returns it.
    """
    logging.debug("Applying sanity check validator...")
    corrected = False
    original_validator_note = analysis_dict.get("_validator_note") # Store original note if any

    # --- 1️⃣  MSS Type based on Pivot Structure Candle Count ---
    pivot_analysis = analysis_dict.get("mss_pivot_analysis", {})
    bearish_count_val = pivot_analysis.get("pivot_bearish_count") # Get value before parsing
    bullish_count_val = pivot_analysis.get("pivot_bullish_count") # Get value before parsing
    bearish_count = 0
    bullish_count = 0
    original_mss_type = analysis_dict.get("mss_type", "").lower()
    calculated_mss_type = ""
    counts_provided = False

    try:
        if bearish_count_val is not None:
            bearish_count = int(bearish_count_val)
            counts_provided = True
    except (ValueError, TypeError):
        logging.warning(f"Could not parse pivot_bearish_count '{bearish_count_val}' as int.")
    try:
        if bullish_count_val is not None:
            bullish_count = int(bullish_count_val)
            counts_provided = True
    except (ValueError, TypeError):
        logging.warning(f"Could not parse pivot_bullish_count '{bullish_count_val}' as int.")

    if counts_provided:
        if bearish_count >= 2 and bullish_count >= 2:
            calculated_mss_type = "normal"
        else:
            calculated_mss_type = "agresiv"

        if calculated_mss_type != original_mss_type:
            logging.warning(f"Validator correcting MSS Type: Was '{original_mss_type}', "
                            f"became '{calculated_mss_type}' (Bearish: {bearish_count}, Bullish: {bullish_count})")
            analysis_dict["mss_type"] = calculated_mss_type
            corrected = True
    else:
        logging.info("Skipping MSS type validation due to missing pivot counts.")


    # --- 2️⃣  Trade Direction based on Risk Box Placement ---
    risk_above = analysis_dict.get("is_risk_above_price")
    original_direction = analysis_dict.get("trade_direction", "").lower()
    calculated_direction = original_direction # Default to original if no risk box info

    if risk_above is True:
        calculated_direction = "short"
    elif risk_above is False:
        calculated_direction = "long"

    if calculated_direction in ["short", "long"] and calculated_direction != original_direction:
        logging.warning(f"Validator correcting Trade Direction: Was '{original_direction}', "
                        f"became '{calculated_direction}' (is_risk_above_price: {risk_above})")
        analysis_dict["trade_direction"] = calculated_direction
        corrected = True

    # --- 3️⃣ Check displacement direction consistency ---
    disp_analysis = analysis_dict.get("displacement_analysis", {})
    disp_direction = disp_analysis.get("direction") if isinstance(disp_analysis, dict) else None
    trade_direction = analysis_dict.get("trade_direction") # Use potentially corrected direction
    consistency_warning = ""

    if trade_direction == "long" and disp_direction == "bearish":
        consistency_warning = "Warning: Long trade direction identified, but displacement appears bearish."
        corrected = True # Flag that a note should be added or updated
    elif trade_direction == "short" and disp_direction == "bullish":
        consistency_warning = "Warning: Short trade direction identified, but displacement appears bullish."
        corrected = True

    if consistency_warning:
         analysis_dict["direction_consistency_warning"] = consistency_warning
         logging.warning(consistency_warning)

    # Update validator note if corrections or warnings occurred
    if corrected:
        base_note = "NOTE: Analysis adjusted or flagged by internal rules for consistency."
        # Combine notes if one already existed
        if original_validator_note and original_validator_note != base_note:
             analysis_dict["_validator_note"] = f"{original_validator_note} {base_note}"
        else:
             analysis_dict["_validator_note"] = base_note

    logging.debug("Finished sanity check validator.")
    return analysis_dict


# ---------------------------------------------------------------------------
# ROUTES – IMAGE HYBRID (REVISED WITH IMPROVED VISUAL ANALYSIS + VALIDATOR)
# ---------------------------------------------------------------------------

class ImageHybridQuery(BaseModel):
    question: str
    image_url: str
    session_id: Optional[str] = None

@app.post("/ask-image-hybrid", response_model=Dict[str, str])
async def ask_image_hybrid(payload: ImageHybridQuery) -> Dict[str, str]:
    """Handles questions with chart screenshots, aiming for detailed visual analysis."""
    session_id = payload.session_id or generate_session_id()
    logging.info(f"Received /ask-image-hybrid request. Question: '{payload.question[:100]}...', Image URL: {payload.image_url}, Session ID: {session_id}")

    detailed_vision_analysis: Dict[str, Any] = {"error": "Vision analysis not performed"}
    ocr_text: str = ""
    course_context: str = ""

    query_info = identify_query_type(payload.question)
    logging.info(f"Query identified as type: {query_info['type']}, requires_full_analysis: {query_info['requires_full_analysis']}")

    # --- 1️⃣ Detailed Vision Analysis & OCR ---
    try:
        # --- Verify image URL accessibility first ---
        try:
            logging.debug(f"Checking image URL accessibility: {payload.image_url}")
            img_response = requests.head(payload.image_url, timeout=10, allow_redirects=True)
            img_response.raise_for_status()
            content_type = img_response.headers.get('Content-Type', '').lower()
            if not content_type.startswith('image/'):
                logging.warning(f"URL {payload.image_url} does not appear to be an image (Content-Type: {content_type}). Proceeding anyway.")
            logging.info("Image URL is accessible.")
        except requests.exceptions.RequestException as img_err:
            logging.error(f"❌ Image URL access error: {img_err}")
            raise HTTPException(status_code=400, detail="Nu am putut accesa imaginea furnizată. Verifică URL-ul.")

        # --- Call GPT-4 Vision for analysis based on query type ---
        try:
            logging.info(f"Starting {VISION_MODEL} Vision analysis for query type: {query_info['type']}...")

            # --- DETERMINE APPROPRIATE VISION SYSTEM PROMPT BASED ON QUERY TYPE ---
            # REVIEW: All system prompts seem correctly updated with constraints, 1M rule, FVG rule etc.
            # REVIEW: Also added liquidity_status request to trade_evaluation prompt.
            if query_info["type"] == "liquidity":
                detailed_vision_system_prompt = (
                     "You are an expert Trading Instituțional chart analyst specializing in liquidity identification. Analyze this chart "
                     "\n\n**IMPORTANT CONSTRAINTS:**"
                     "\n1. Base your entire analysis SOLELY on the visual information present in the provided chart image. DO NOT invent features, price levels, or patterns that are not clearly visible."
                     "\n2. Adhere strictly to the Trading Instituțional methodology. Analyze ONLY Liquidity, MSS (Normal/Aggressive), and Displacement."
                     "\n3. DO NOT identify or mention unrelated concepts like Order Blocks, general support/resistance, divergences, indicators (unless part of a marked zone), or other chart patterns."
                     "\nFollow the specific JSON structure requested below."
                     "\n---"
                     "focus ONLY on the liquidity zones marked. Output a structured JSON with these fields:"
                     "\n1. 'analysis_possible': boolean"
                     "\n2. 'visible_liquidity_zones': List their positions and whether they appear to be major or minor"
                     "\n3. 'liquidity_quality': Assess the quality of marked liquidity zones based on price action around them"
                     "\n4. 'overall_trend_direction': ONLY 'bullish', 'bearish', or 'sideways' - but focus on marked liquidity"
                     "\n5. 'candle_colors': Specifically identify what colors represent bullish vs bearish candles in THIS chart"
                     "\nDO NOT analyze trade setups, MSS structures, or displacement unless specifically marked as liquidity areas."
                     "\nOnly analyze what's clearly visible and relevant to LIQUIDITY in the image."
                )
            elif query_info["type"] == "trend":
                 detailed_vision_system_prompt = (
                     "You are an expert Trading Instituțional chart analyst specializing in trend identification. Analyze this chart "
                     "\n\n**IMPORTANT CONSTRAINTS:**"
                     "\n1. Base your entire analysis SOLELY on the visual information present in the provided chart image. DO NOT invent features, price levels, or patterns that are not clearly visible."
                     "\n2. Adhere strictly to the Trading Instituțional methodology. Analyze ONLY Liquidity, MSS (Normal/Aggressive), and Displacement."
                     "\n3. DO NOT identify or mention unrelated concepts like Order Blocks, general support/resistance, divergences, indicators (unless part of a marked zone), or other chart patterns."
                     "\nFollow the specific JSON structure requested below."
                     "\n---"
                     "Output a structured JSON with these fields:"
                     "\n1. 'analysis_possible': boolean"
                     "\n2. 'trend_direction': MUST be 'bullish', 'bearish', or 'sideways'"
                     "\n3. 'trend_strength': Assess the strength and clarity of the trend"
                     "\n4. 'trend_structure': Brief description of what makes this a trend (higher highs/lows or lower highs/lows)"
                     "\n5. 'candle_colors': SPECIFICALLY identify what colors represent bullish vs bearish candles in THIS chart"
                     "\n6. 'visible_trendlines': Describe any visible trendlines or support/resistance levels" # Note: This asks for S/R lines, might conflict with constraint 3 if not careful
                     "\nDO NOT analyze MSS structures or specific trade setups unless directly related to the trend."
                     "\nOnly analyze what's clearly visible in the image related to trend direction and strength."
                 )
            elif query_info["type"] == "mss_classification":
                 detailed_vision_system_prompt = (
                     "You are an expert Trading Instituțional chart analyst specializing in MSS classification. "
                     "\n\n**IMPORTANT CONSTRAINTS:**"
                     "\n1. Base your entire analysis SOLELY on the visual information present in the provided chart image. DO NOT invent features, price levels, or patterns that are not clearly visible."
                     "\n2. Adhere strictly to the Trading Instituțional methodology. Analyze ONLY Liquidity, MSS (Normal/Aggressive), and Displacement."
                     "\n3. DO NOT identify or mention unrelated concepts like Order Blocks, general support/resistance, divergences, indicators (unless part of a marked zone), or other chart patterns."
                     "\nFollow the specific JSON structure requested below."
                     "\n---"
                     "Analyze this chart with attention to the following critical criteria:"
                     "\n\n**CRITICAL MSS CLASSIFICATION RULES:**"
                     "\n1. Identify the swing high or low (the 'pivot') that is potentially broken by an MSS."
                     "\n2. Analyze the candle composition FORMING this pivot structure **based on how it would appear on a 1-minute (1M) timeframe chart**."
                     "\n3. Count the number of bearish and bullish candles within this core pivot structure (**using the 1M view**)."
                     "\n4. Determine `has_minimum_structure`: This is TRUE **only if** the pivot contains at least 2 bearish candles AND at least 2 bullish candles."
                     "\n5. Classify `mss_type`:"
                     "   - 'normal': If `has_minimum_structure` is TRUE."
                     "   - 'agresiv': If `has_minimum_structure` is FALSE."
                     "\n6. Identify the `break_direction` ('upward' breaking high, 'downward' breaking low)."
                     "\n\nOutput a structured JSON with these fields:"
                     "\n1. 'analysis_possible': boolean"
                     "\n2. 'mss_location': Description of where MSS is identified or labeled."
                     "\n3. 'mss_pivot_analysis': { "
                     "     'description': 'Text describing the candles forming the pivot structure broken by MSS', "
                     "'pivot_bearish_count': 'INTEGER count of BEARISH candles forming the core pivot structure (analyzed from 1M view)', "
                     "'pivot_bullish_count': 'INTEGER count of BULLISH candles forming the core pivot structure (analyzed from 1M view)', "
                     "     'has_minimum_structure': 'BOOLEAN, true only if bearish_count >= 2 AND bullish_count >= 2'"
                     "   }"
                     "\n4. 'mss_type': MUST be EXACTLY 'normal' (if has_minimum_structure is true) or 'agresiv' (if false)."
                     "\n5. 'break_direction': 'upward' or 'downward'"
                     "\n6. 'candle_colors': Description of bullish vs bearish candle colors in THIS chart."
                     "\nThe PIVOT STRUCTURE COMPOSITION (bearish/bullish counts) is the ONLY factor determining 'normal' vs 'agresiv'."
                 )
            elif query_info["type"] == "displacement":
                 detailed_vision_system_prompt = (
                     "You are an expert Trading Instituțional chart analyst specializing in displacement analysis."
                     "\n\n**IMPORTANT CONSTRAINTS:**"
                     "\n1. Base your entire analysis SOLELY on the visual information present in the provided chart image. DO NOT invent features, price levels, or patterns that are not clearly visible."
                     "\n2. Adhere strictly to the Trading Instituțional methodology. Analyze ONLY Liquidity, MSS (Normal/Aggressive), and Displacement."
                     "\n3. DO NOT identify or mention unrelated concepts like Order Blocks, general support/resistance, divergences, indicators (unless part of a marked zone), or other chart patterns."
                     "\nFollow the specific JSON structure requested below."
                     "\n---"
                     "Your task is to analyze the displacement visible in the chart. Output a structured JSON with these fields:"
                     "\n1. 'analysis_possible': boolean"
                     "\n2. 'displacement_direction': 'bullish' (price moving up) or 'bearish' (price moving down)"
                     "\n3. 'displacement_strength': Assess whether the displacement is strong, moderate, or weak"
                     "\n4. 'fvg_within_displacement': Identify if Fair Value Gaps (FVGs) are created *within* the main displacement move itself. Discuss FVGs **only** if they confirm the displacement."
                     "\n5. 'candle_colors': SPECIFICALLY identify what colors represent bullish vs bearish candles in THIS chart"
                     "\n6. 'trade_direction': Based on displacement, is this likely a 'short' or 'long' trade"
                     "\nFocus ONLY on the displacement aspect - the impulsive price movement creating gaps/imbalances."
                     "\nDisplacement should match trade direction: bearish displacement for short trades, bullish for long trades."
                 )
            elif query_info["type"] == "fvg":
                 detailed_vision_system_prompt = (
                     "You are an expert Trading Instituțional chart analyst specializing in Fair Value Gap (FVG) identification."
                     "\n\n**IMPORTANT CONSTRAINTS:**"
                     "\n1. Base your entire analysis SOLELY on the visual information present in the provided chart image. DO NOT invent features, price levels, or patterns that are not clearly visible."
                     "\n2. Adhere strictly to the Trading Instituțional methodology. Analyze ONLY Liquidity, MSS (Normal/Aggressive), and Displacement."
                     "\n3. DO NOT identify or mention unrelated concepts like Order Blocks, general support/resistance, divergences, indicators (unless part of a marked zone), or other chart patterns."
                     "\nFollow the specific JSON structure requested below."
                     "\n---"
                     "Your task is to analyze the FVGs visible in the chart. Output a structured JSON with these fields:"
                     "\n1. 'analysis_possible': boolean"
                     "\n2. 'fvg_locations': Identify and describe where FVGs are located in the chart"
                     "\n3. 'fvg_types': For each FVG, indicate if it's bullish (created by upward movement) or bearish (created by downward movement)"
                     "\n4. 'fvg_quality': Assess the quality and clarity of the identified FVGs"
                     "\n5. 'candle_colors': SPECIFICALLY identify what colors represent bullish vs bearish candles in THIS chart"
                     "\n6. 'trade_implication': How these FVGs might affect trading decisions"
                     "\nFocus ONLY on the FVG aspects - the gaps/imbalances created by impulsive price movements."
                     "\nRemember: FVGs are created when price moves impulsively, leaving an area of no trading activity."
                 )
            else: # Includes 'trade_evaluation' and 'general' types
                 detailed_vision_system_prompt = (
                     "You are an expert Trading Instituțional chart analyst."
                     "\n\n**IMPORTANT CONSTRAINTS:**"
                     "\n1. Base your entire analysis SOLELY on the visual information present in the provided chart image. DO NOT invent features, price levels, or patterns that are not clearly visible."
                     "\n2. Adhere strictly to the Trading Instituțional methodology. Analyze ONLY Liquidity, MSS (Normal/Aggressive), and Displacement."
                     "\n3. DO NOT identify or mention unrelated concepts like Order Blocks, general support/resistance, divergences, indicators (unless part of a marked zone), or other chart patterns."
                     "\nFollow the specific JSON structure requested below."
                     "\n---"
                     "Analyze this trading chart comprehensively and output a structured JSON with your detailed findings. Follow these specific guidelines:"
                     "\n\n**1. COLOR INTERPRETATION FIRST:**"
                     "\n   - Identify `candle_colors`: Describe colors for bullish/bearish candles in THIS chart."
                     "\n   - Note colors for zones/indicators if obvious."
                     "\n\n**2. TRADE DIRECTION (PRIORITY on Risk Box):**"
                     "\n   - **PRIMARY:** Look for a Risk/Reward box (often colored: Red usually indicates the Risk/Stop zone, Green/Blue the Profit/Target zone)."
                     "\n   - Determine the entry point (often the middle line or near current price if box is active)."
                     "\n   - Determine `is_risk_above_price`: BOOLEAN. Is the defined Risk zone (e.g., the Red part) clearly ABOVE the entry point? True if yes, False if the Risk zone is clearly BELOW the entry point. Null if no clear R/R box or entry is visible." # More explicit check
                     "\n   - Set `trade_direction` based PRIMARILY on this: 'short' if `is_risk_above_price` is True, 'long' if `is_risk_above_price` is False."
                     "\n   - **Secondary:** If no clear risk box, infer `trade_direction` from labels ('SHORT'/'LONG', arrows) or overall strong recent directional movement (displacement)."
                     "\n   - Output: 'short', 'long', or 'undetermined'."
                     "\n\n**3. MSS CLASSIFICATION (Based on PIVOT STRUCTURE):**"
                     "\n   - Look for where 'MSS' is labeled or implied by a structure break."
                     "\n   - Identify the swing high/low (the 'pivot') that was broken."
                     "\n   - Analyze the candle composition FORMING this pivot structure **based on how it would appear on a 1-minute (1M) timeframe chart**."
                     "\n   - Count the number of bearish and bullish candles within this core pivot structure (**using the 1M view**)."
                     "\n   - Determine `has_minimum_structure`: BOOLEAN (True only if count >= 2 for BOTH bearish and bullish)."
                     "\n   - Classify `mss_type`: 'normal' if `has_minimum_structure` is True, else 'agresiv'."
                     "\n   - Identify `break_direction` ('upward' or 'downward')."
                     "\n\n**4. DISPLACEMENT & FVG ANALYSIS:**"
                     "\n   - Identify the main `displacement_analysis`: Direction ('bullish'/'bearish'), strength."
                     "\n   - **CRITICAL FVG CHECK:** Meticulously scan the entire area *after* the confirmed MSS break for ALL visible Fair Value Gaps (FVGs - imbalances between candle 1/3 wicks, often marked by boxes or faint lines/rectangles). Count them accurately." # Emphasize ALL, add detail
                     "\n   - Report findings in `fvg_analysis`: { 'count': integer, 'description': 'Describe all observed FVGs post-MSS, noting markings and SLG pattern if applicable.' }. If none, count is 0." # Changed structure
                     "\n   - Ensure displacement direction aligns with the determined `trade_direction`."
                     "\n\n**5. ZONES, LIQUIDITY & OUTCOME:**"
                     "\n   - Identify the key `liquidity_zones` relevant to the setup (e.g., marked highs/lows)."
                     "\n   - Determine the `liquidity_status`: 'swept' if price clearly traded past the key liquidity level *before* the MSS/setup formed, 'untouched' otherwise." # Emphasized timing
                     "\n   - Describe the liquidity interaction in the `liquidity_zones` field (e.g., 'Buy-side liquidity above high X was swept before MSS')." # Added instruction for description field
                     "\n   - Assess `trade_outcome` ('win', 'loss', 'running', 'undetermined', 'potential_setup') if possible based on price movement after entry."
                     "\n\n**6. ESSENTIAL JSON FIELDS:**" # Added liquidity_status here
                     "\n   - 'analysis_possible': boolean"
                     "\n   - 'candle_colors': description"
                     "\n   - 'is_risk_above_price': boolean | null"
                     "\n   - 'trade_direction': 'short' | 'long' | 'undetermined'"
                     "\n   - 'mss_pivot_analysis': { 'description': description, 'pivot_bearish_count': integer, 'pivot_bullish_count': integer, 'has_minimum_structure': boolean }"
                     "\n   - 'mss_type': 'normal' | 'agresiv' | 'not_identified'"
                     "\n   - 'break_direction': 'upward' | 'downward' | 'none'"
                     "\n   - 'displacement_analysis': { 'direction': 'bullish'|'bearish'|'none', 'strength': description }" # Removed fvg_created from here
                     "\n   - 'fvg_analysis': { 'count': integer, 'description': description }"
                     "\n   - 'liquidity_zones': description"
                     "\n   - 'liquidity_status': 'swept' | 'untouched' | 'unclear'" # Added this field
                     "\n   - 'trade_outcome': 'win'|'loss'|'running'|'undetermined'|'potential_setup'"
                     "\n   - 'visible_labels': list of strings"
                 )


            # --- Craft user prompt --- (Removed redundant constraints)
            # REVIEW: User prompts look fine now without duplicated constraints.
            if query_info["type"] == "liquidity":
                 detailed_vision_user_prompt = (
                     f"Analyze this trading chart FOCUSING ONLY ON LIQUIDITY ZONES marked in the image. "
                     f"The user is asking: '{payload.question}'. "
                     f"Be sure to identify SPECIFICALLY what colors represent bullish vs bearish candles in THIS chart. "
                     f"DO NOT analyze MSS structure, displacement, or trade setups unless they're directly related to liquidity. "
                     f"Look for areas marked as 'Liq', 'Liquidity', or similar designations. "
                     f"Note if liquidity is marked at swing highs (for shorts) or swing lows (for longs). "
                     f"Provide your structured analysis as JSON."
                 )
            elif query_info["type"] == "trend":
                 detailed_vision_user_prompt = (
                     f"Analyze this trading chart FOCUSING ONLY ON THE TREND DIRECTION AND STRENGTH. "
                     f"The user is asking: '{payload.question}'. "
                     f"Be sure to identify SPECIFICALLY what colors represent bullish vs bearish candles in THIS chart. "
                     f"Determine if the visible trend is clearly BULLISH (price moving up), BEARISH (price moving down), or SIDEWAYS. "
                     f"Look for higher highs and higher lows (bullish) or lower highs and lower lows (bearish). "
                     f"Provide your structured analysis as JSON."
                 )
            elif query_info["type"] == "mss_classification":
                 detailed_vision_user_prompt = (
                     f"Analyze this trading chart FOCUSING ONLY ON MSS CLASSIFICATION according to Trading Instituțional methodology. "
                     f"The user is asking: '{payload.question}'. "
                     f"Follow the System Prompt instructions carefully: "
                     f"1. Identify the pivot structure being broken. "
                     f"2. Count bearish and bullish candles FORMING THE PIVOT. "
                     f"3. Determine 'has_minimum_structure' (>=2 AND >=2). "
                     f"4. Set 'mss_type' based ONLY on 'has_minimum_structure'. "
                     f"5. Identify break direction and candle colors. "
                     f"Provide your structured analysis as JSON."
                 )
            elif query_info["type"] == "displacement":
                 detailed_vision_user_prompt = (
                     f"Analyze this trading chart FOCUSING ONLY ON DISPLACEMENT. "
                     f"The user is asking: '{payload.question}'. "
                     f"Be sure to identify SPECIFICALLY what colors represent bullish vs bearish candles in THIS chart. "
                     f"Look for impulsive price movements that create gaps (FVGs) in the chart. "
                     f"Determine if the displacement is BULLISH (price moving up) or BEARISH (price moving down). "
                     f"Note if the displacement creates Fair Value Gaps (FVGs) and how strong the movement is. "
                     f"Provide your structured analysis as JSON."
                 )
            elif query_info["type"] == "fvg":
                 detailed_vision_user_prompt = (
                     f"Analyze this trading chart FOCUSING ONLY ON FAIR VALUE GAPS (FVGs). "
                     f"The user is asking: '{payload.question}'. "
                     f"Be sure to identify SPECIFICALLY what colors represent bullish vs bearish candles in THIS chart. "
                     f"Look for gaps created by impulsive price movements - areas where no trading has occurred. "
                     f"Identify if the FVGs are BULLISH (created by upward movement) or BEARISH (created by downward movement). "
                     f"Note how these FVGs might affect trading decisions. "
                     f"Provide your structured analysis as JSON."
                 )
            else: # General/Evaluation
                 detailed_vision_user_prompt = (
                     f"Analyze this trading chart according to Trading Instituțional methodology for a general setup evaluation. "
                     f"The user is asking: '{payload.question}'. "
                     f"Follow the instructions in the System Prompt precisely: "
                     f"1. Identify candle colors. "
                     f"2. Determine trade direction (PRIORITIZE risk box position). "
                     f"3. Classify MSS based on PIVOT structure candle counts (>=2 bearish AND >=2 bullish = Normal). "
                     f"4. Analyze displacement and FVGs (only if confirming displacement). " # Clarified FVG context here too
                     f"5. Assess liquidity (including status: swept/untouched) and outcome if possible. " # Added liquidity status
                     f"Provide your comprehensive structured analysis as JSON."
                 )

            # --- Build the messages list including few-shot examples ---
            # REVIEW: Logic for building vision_messages looks correct.
            vision_messages = []

            # 1. Add the system prompt
            vision_messages.append({"role": "system", "content": detailed_vision_system_prompt})

            # 2. Add the few-shot examples
            #    NOTE: Consider using fewer examples (e.g., 3-4) if you hit token limits
            example_user_prompt_text = "Analyze this example chart based on the system prompt instructions and provide the JSON output." # Simplified prompt for examples
            for example in FEW_SHOT_EXAMPLES:
                # Add the user turn for the example
                vision_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": example["image_url"]}},
                        {"type": "text", "text": example_user_prompt_text}
                    ]
                })
                # Add the assistant turn (the expected JSON output)
                vision_messages.append({
                    "role": "assistant",
                    "content": example["assistant_json_output"]
                })

            # 3. Add the ACTUAL user request
            vision_messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": payload.image_url}},
                    {"type": "text", "text": detailed_vision_user_prompt},
                ]
            })

            # --- Make the API call using the constructed messages list ---
            # REVIEW: Using vision_messages and increased max_tokens. Looks correct.
            vision_resp = openai.chat.completions.create(
                model=VISION_MODEL,
                messages=vision_messages,
                max_tokens=2500,
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            # --- Process the response (expecting JSON directly) ---
            # REVIEW: JSON parsing and fallback looks okay.
            raw_response_content = vision_resp.choices[0].message.content.strip()
            logging.info(f"{VISION_MODEL} Vision analysis completed.")
            logging.debug(f"Raw Vision JSON Response: {raw_response_content}")

            try:
                detailed_vision_analysis = json.loads(raw_response_content)
                if not isinstance(detailed_vision_analysis, dict) or 'analysis_possible' not in detailed_vision_analysis:
                    logging.warning("Vision JSON structure might be invalid. Setting error.")
                    detailed_vision_analysis = {"error": "Invalid JSON structure received from vision model", "raw_response": raw_response_content}
                else:
                    logging.info(f"Successfully parsed Vision JSON.")

            except json.JSONDecodeError as json_err:
                 logging.error(f"❌ Failed to decode JSON from Vision response: {json_err}.")
                 fallback_json_string = extract_json_from_text(raw_response_content)
                 if fallback_json_string:
                     try:
                         detailed_vision_analysis = json.loads(fallback_json_string)
                         logging.info("Successfully parsed Vision JSON using fallback extractor.")
                     except json.JSONDecodeError as fallback_err:
                         logging.error(f"❌ Fallback JSON extraction also failed: {fallback_err}. Raw string: '{fallback_json_string}'")
                         detailed_vision_analysis = {"error": "Invalid JSON structure from vision model (fallback failed)", "raw_response": raw_response_content}
                 else:
                     detailed_vision_analysis = {"error": "No valid JSON found in vision response", "raw_response": raw_response_content}

            # --- >>> APPLY SANITY CHECK VALIDATOR <<< ---
            # REVIEW: Validator logic looks reasonable and includes consistency check.
            if isinstance(detailed_vision_analysis, dict) and "error" not in detailed_vision_analysis:
                 logging.debug("Original Vision Analysis before validation: %s", json.dumps(detailed_vision_analysis, indent=2, ensure_ascii=False))
                 # Apply the validator function
                 detailed_vision_analysis = _sanity_check_analysis(detailed_vision_analysis)
                 if "_validator_note" in detailed_vision_analysis:
                      logging.warning("Validator applied adjustments to Vision analysis. Final analysis includes '_validator_note'.")
                 logging.debug("Validated Vision Analysis: %s", json.dumps(detailed_vision_analysis, indent=2, ensure_ascii=False))
            else:
                 logging.warning("Skipping validator due to error in vision analysis or non-dict result.")
            # --- >>> END VALIDATOR <<< ---

        except (APIError, RateLimitError) as e:
            logging.error(f"OpenAI Vision API error: {e}")
            detailed_vision_analysis = {"error": f"Vision API error: {str(e)}"}
        except Exception as e:
            logging.exception(f"Unexpected error during Vision processing: {e}")
            detailed_vision_analysis = {"error": "Unexpected vision processing error"}

        # --- Run OCR (keep separate for now) ---
        ocr_text = extract_text_from_image(payload.image_url)

    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Unhandled exception during Vision/OCR stage: {e}")
        if "error" not in detailed_vision_analysis:
             detailed_vision_analysis = {"error": "Unhandled exception in Vision/OCR stage"}

    # --- 2️⃣ Vector Search (RAG) ---
    # REVIEW: Vector search logic looks fine, uses validated analysis.
    try:
        query_parts = [f"Question: {payload.question}"]
        if len(ocr_text) > 10: query_parts.append(f"OCR Text Snippet: {ocr_text[:200]}")
        query_parts.append(f"Query type: {query_info['type']}")

        if isinstance(detailed_vision_analysis, dict):
            if "candle_colors" in detailed_vision_analysis: query_parts.append(f"Chart candle colors: {str(detailed_vision_analysis.get('candle_colors'))[:100]}")
            if detailed_vision_analysis.get("trade_direction") in ["long", "short"]: query_parts.append(f"Trade direction: {detailed_vision_analysis.get('trade_direction')}")
            if detailed_vision_analysis.get("mss_type") in ["agresiv", "normal"]: query_parts.append(f"MSS type: {detailed_vision_analysis.get('mss_type')}")
            if detailed_vision_analysis.get("fvg_analysis"): query_parts.append(f"FVG analysis summary: {str(detailed_vision_analysis.get('fvg_analysis'))[:100]}")
            disp_analysis = detailed_vision_analysis.get("displacement_analysis", {})
            if isinstance(disp_analysis, dict) and disp_analysis.get("direction") in ["bullish", "bearish"]: query_parts.append(f"Displacement direction: {disp_analysis.get('direction')}")
            if detailed_vision_analysis.get("liquidity_status") in ["swept", "untouched"]: query_parts.append(f"Liquidity Status: {detailed_vision_analysis.get('liquidity_status')}") # Added liquidity status to query

        combo_query = " ".join(query_parts)

        logging.info(f"Constructed embedding query (first 200 chars): {combo_query[:200]}...")
        emb_response = openai.embeddings.create(model=EMBEDDING_MODEL, input=[combo_query])
        query_embedding = emb_response.data[0].embedding
        logging.info("Generated embedding for combined query.")
        matches = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        retrieved_matches = matches.get("matches", [])
        course_context = "\n\n---\n\n".join(m["metadata"].get("text", "") for m in retrieved_matches if m["metadata"].get("text")).strip()
        logging.info(f"Pinecone query returned {len(retrieved_matches)} matches. Context length: {len(course_context)}")
        logging.debug(f"DEBUG - Retrieved Course Context Content:\n---\n{course_context}\n---")

        # Inject definitions logic (remains the same, but ensures context is available)
        definitions_to_add = []
        # Always add MSS definitions if MSS is mentioned or part of analysis
        if "mss" in payload.question.lower() or (isinstance(detailed_vision_analysis, dict) and detailed_vision_analysis.get("mss_type")):
             if MSS_AGRESIV_STRUCTURAL_DEFINITION.lower() not in course_context.lower(): definitions_to_add.append(MSS_AGRESIV_STRUCTURAL_DEFINITION)
             # Using corrected Normal definition
             if "definiție structurală mss normal" not in course_context.lower(): definitions_to_add.append("Definiție Structurală MSS Normal: Un MSS normal necesită ca pivotul (swing high/low) rupt să fie format din minim 2 candele bearish ȘI minim 2 candele bullish.")

        if "fvg" in payload.question.lower() or "fair value gap" in payload.question.lower() or (isinstance(detailed_vision_analysis, dict) and detailed_vision_analysis.get("fvg_analysis")):
              if FVG_STRUCTURAL_DEFINITION.lower() not in course_context.lower(): definitions_to_add.append(FVG_STRUCTURAL_DEFINITION)

        if "displacement" in payload.question.lower() or (isinstance(detailed_vision_analysis, dict) and detailed_vision_analysis.get("displacement_analysis")):
              if DISPLACEMENT_DEFINITION.lower() not in course_context.lower(): definitions_to_add.append(DISPLACEMENT_DEFINITION)

        if definitions_to_add:
            definition_block = "\n\n".join(definitions_to_add)
            course_context = f"Definiții Relevante:\n{definition_block}\n\n---\n\nMaterial Curs:\n{course_context}"
            logging.info(f"Injected {len(definitions_to_add)} definitions into context.")

        if not course_context.strip():
             logging.warning("Pinecone query returned no relevant context, and no definitions were injected.")
             course_context = "[Eroare: Niciun context specific din curs nu a fost găsit pentru această combinație.]"
             minimal_definitions = []
             if "mss" in payload.question.lower(): minimal_definitions.extend([MSS_AGRESIV_STRUCTURAL_DEFINITION, "MSS Normal: Pivotul rupt are >=2 bearish AND >=2 bullish."])
             if "fvg" in payload.question.lower(): minimal_definitions.append(FVG_STRUCTURAL_DEFINITION)
             if "displacement" in payload.question.lower(): minimal_definitions.append(DISPLACEMENT_DEFINITION)
             if minimal_definitions: course_context += "\n\nDefiniții de Bază:\n" + "\n".join(minimal_definitions)

    except (APIError, RateLimitError) as e:
        logging.error(f"OpenAI Embedding API error during hybrid search: {e}")
        course_context = "[Eroare: Nu s-a putut genera embedding pentru căutare context]"
        # ... (rest of error handling) ...
    except PineconeException as e:
        logging.error(f"Pinecone query error during hybrid search: {e}")
        course_context = "[Eroare: Nu s-a putut căuta în materialele de curs]"
        # ... (rest of error handling) ...
    except Exception as e:
        logging.exception(f"Unexpected error during vector search stage: {e}")
        course_context = "[Eroare: Problemă neașteptată la căutarea contextului]"
        # ... (rest of error handling) ...

    # --- 3️⃣ Final Answer Generation ---
    # REVIEW: Final answer generation logic looks fine. Uses validated analysis and context.
    try:
        try:
            if isinstance(detailed_vision_analysis, dict):
                visual_analysis_report_str = json.dumps(detailed_vision_analysis, indent=2, ensure_ascii=False)
            else:
                visual_analysis_report_str = json.dumps({"error": "Visual analysis data is not a valid dictionary.", "raw_data": str(detailed_vision_analysis)}, ensure_ascii=False)
                logging.error("Vision analysis result was not a dict, cannot format as JSON for final prompt.")
        except Exception as json_dump_err:
            logging.exception("Error dumping visual analysis to JSON string.")
            visual_analysis_report_str = json.dumps({"error": "Could not format visual analysis.", "details": str(json_dump_err)}, ensure_ascii=False)

        logging.debug("Final Visual Analysis Report string for prompt:\n%s", visual_analysis_report_str)

        def _build_system_prompt(query_type: str, requires_full_analysis: bool) -> str:
            # ... (This function seems fine) ...
            BASE = SYSTEM_PROMPT_CORE
            PROMPTS = {
                 "liquidity": ("\n--- Instructions for Liquidity Zone Analysis Response ---\nFocus your answer *only* on the liquidity analysis provided in the report and context."),
                 "trend": ("\n--- Instructions for Trend Analysis Response ---\nFocus your answer *only* on the trend analysis provided."),
                 "mss_classification": ("\n--- Instructions for MSS Classification Response ---\nExplain the MSS classification based *only* on the pivot structure analysis provided. Reference the course definitions."),
                 "displacement": ("\n--- Instructions for Displacement Analysis Response ---\nDescribe the displacement and any FVGs based on the analysis provided."),
                 "fvg": ("\n--- Instructions for FVG Analysis Response ---\nDescribe the identified FVGs and their potential implications based on the analysis and context."),
                 "trade_evaluation": ("\n--- Instructions for Trade Setup Evaluation Response ---\n1. Provide an **objective analysis** based on the Trading Instituțional methodology.\n2. Do **not** begin by saying you don't give opinions. \n3. Use an active voice ('I see...', 'This indicates...') where appropriate, like an experienced colleague sharing objective findings. \n4. Summarize key elements from the Visual Analysis Report:\n   • trade_direction  • mss_type  • displacement_and_FVGs (including count/pattern)\n  (Mention liquidity interaction *only if* noteworthy or unusual context is present in the report).  • liquidity_status  • validator notes \n5. Relate findings to the Course Context (confluence/divergence)."), # Added liquidity_status here
                 "\nIMPORTANT: Conclude your analysis immediately after covering point 5. DO NOT add any concluding disclaimers about not providing financial advice or personal opinions."
                 "general": ("\n--- Instructions for General Query Response ---\nAnswer by synthesizing the Visual Analysis Report and Course Context. If the question implicitly asks for an evaluation, follow the Trade Setup Evaluation instructions."),
            }
            effective_type = ("trade_evaluation" if query_type == "general" and requires_full_analysis else query_type)
            chosen = PROMPTS.get(effective_type, PROMPTS["trade_evaluation"])
            return BASE + "\n\n" + chosen.strip()

        final_system_prompt = _build_system_prompt(query_info["type"], query_info.get("requires_full_analysis", False))

        final_user_prompt = (
            f"User Question: {payload.question}\n\n"
            f"Visual Analysis Report (JSON):\n```json\n{visual_analysis_report_str}\n```\n\n"
            f"Retrieved Course Context:\n{course_context}\n\n"
            f"Task: The user asked: '{payload.question}'. Respond in Romanian with a **structured technical analysis** "
            "of the setup shown in the image, strictly following Trading Instituțional methodology. Present objective "
            "findings first, then conclude with the disclaimers specified in the system prompt."
        )

        logging.debug("Final System prompt length: %d", len(final_system_prompt))
        logging.debug("Final User prompt length: %d", len(final_user_prompt))
        logging.debug("Final User prompt (first 500 chars): %s", final_user_prompt[:500] + "...")

        try:
            chat_completion = openai.chat.completions.create(
                model=COMPLETION_MODEL,
                messages=[
                    {"role": "system", "content": final_system_prompt},
                    {"role": "user", "content": final_user_prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            final_answer = chat_completion.choices[0].message.content.strip()
            logging.info("Final answer generated successfully using %s.", COMPLETION_MODEL)
            logging.debug("Final Answer (raw): %s...", final_answer[:300])

            return {
                "answer": final_answer,
                "session_id": session_id,
            }

        except (APIError, RateLimitError) as e:
            logging.error("OpenAI Chat API error (%s): %s", COMPLETION_MODEL, e)
            return {"answer": f"Nu am putut genera un răspuns final. Serviciul OpenAI ({COMPLETION_MODEL}) nu este disponibil momentan.", "session_id": session_id, "error": str(e)}
        except Exception as e_final:
            logging.exception("Unexpected error during final answer generation")
            return {"answer": "A apărut o eroare la generarea răspunsului final. Te rugăm să încerci din nou.", "session_id": session_id, "error": str(e_final)}

    except Exception as e_gen:
        logging.exception("Unhandled exception in final response generation stage")
        return {"answer": "A apărut o eroare neașteptată la procesarea răspunsului. Te rugăm să încerci din nou.", "session_id": session_id, "error": str(e_gen)}

# Health check endpoint - moved outside the ask_image_hybrid function
# REVIEW: Health check looks fine.
@app.get("/health")
async def health_check():
    """Simple health check endpoint to verify the service is running."""
    try:
        openai.embeddings.create(model=EMBEDDING_MODEL, input=["test"])
        test_vector = [0.0] * 1536
        index.query(vector=test_vector, top_k=1)

        return {"status": "healthy", "openai": "connected", "pinecone": "connected", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
