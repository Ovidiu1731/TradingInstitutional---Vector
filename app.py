import os
import re
import json
import logging
import time
import copy
import base64 # Added for image encoding in ask_image_hybrid
from io import BytesIO
from typing import Dict, Any, Optional, List, Union

# Async and CV libraries
import asyncio
import httpx
import math
import aiohttp # For async image downloads
import cv2
from utils.chunk_filtering import filter_and_rank_chunks
import numpy as np
import cachetools # For TTLCache

import pytesseract
from PIL import Image # ImageDraw can be added if debugging CV by drawing on images
from dotenv import load_dotenv
from utils.query_expansion import expand_query
# No explicit threading import needed if TTLCache handles its own thread safety for basic ops
# and FastAPI handles request concurrency.
from collections import deque # For conversation history fallback if TTLCache fails or for /ask
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import AsyncOpenAI, OpenAI, RateLimitError, APIError # Added sync OpenAI
import pinecone

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# ENVIRONMENT & GLOBALS
# ---------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trading-lessons")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1-aws")
FEEDBACK_LOG = os.getenv("FEEDBACK_LOG", "feedback_log.jsonl")
MIN_SCORE = float(os.getenv("PINECONE_MIN_SCORE", "0.70"))
TOP_K = int(os.getenv("PINECONE_TOP_K", "5"))

# --- Model selection ---
EMBEDDING_MODEL = "text-embedding-ada-002"
VISION_MODEL = "gpt-4o"
COMPLETION_MODEL = "gpt-4o"
TEXT_MODEL = "gpt-3.5-turbo"

# --- Cache for Vision Model results ---
vision_results_cache = cachetools.TTLCache(maxsize=500, ttl=3600)  # 1-hour TTL, store up to 500 results

if not (OPENAI_API_KEY and PINECONE_API_KEY):
    logging.error("Missing OpenAI or Pinecone API key(s)")
    raise ValueError("Missing OpenAI or Pinecone API key(s)")

# --- Initialize Async Clients (as per mentor's advice) ---
async_openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    http_client=httpx.AsyncClient(http2=True, timeout=30.0)
)
aiohttp_session: Optional[aiohttp.ClientSession] = None # Initialized at startup

# --- Pinecone Sync Client (use with asyncio.to_thread) ---
# --- Pinecone Sync Client (use with asyncio.to_thread) ---
# --- Pinecone Sync Client (use with asyncio.to_thread) ---
# --- Pinecone Sync Client (use with asyncio.to_thread) ---
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    # Initialize Pinecone with new API
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    logging.info(f"Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")
except Exception as e:
    logging.error(f"Failed to initialize Pinecone client: {e}")
    raise

# --- Conversation History Store (Using TTLCache for auto-eviction) ---
conversation_history = cachetools.TTLCache(maxsize=10000, ttl=86400) # 10k sessions, 24hr TTL
MAX_HISTORY_TURNS = 3
MAX_HISTORY_MESSAGES = MAX_HISTORY_TURNS * 2

# --- Concurrency Limiter for OpenAI calls ---
openai_call_limiter = asyncio.Semaphore(8)

# --- System Prompts & Definitions ---
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT_CORE = f.read().strip()
except FileNotFoundError:
    logging.warning("system_prompt.txt not found. Using fallback.")
    SYSTEM_PROMPT_CORE = (
        "You are an AI assistant trained by Rareș for the Trading Instituțional community. "
        "Answer questions strictly based on the provided course material, conversation history, and Technical Analysis Report. "
        "Adopt the persona of a helpful, slightly more experienced trading colleague explaining the analysis clearly and objectively. Avoid overly robotic phrasing. Use an active voice where appropriate (e.g., 'I see...', 'This indicates...')."
        "Emulate Rareș's direct, concise teaching style. Be helpful and accurate according to the course rules."
        "\n\nIMPORTANT: The 'Technical Analysis Report' contains information derived from Computer Vision (CV), Vision Model analysis, and a Rule Engine. "
        "Trust fields like 'final_trade_direction', 'final_mss_type', and 'final_trade_outcome' from this report as they have been validated. "
        "Notes like '_cv_note', '_vision_note', '_rule_engine_notes' indicate how the analysis was derived or adjusted."
    )

MSS_AGRESIV_STRUCTURAL_DEFINITION = "Definiție Structurală MSS Agresiv: Un MSS agresiv se produce atunci cand ultimul higher low sau lower high care este rupt (unde se produce shift-ul) nu are in structura sa minim 2 candele bearish ȘI minim 2 candele bullish."
MSS_NORMAL_STRUCTURAL_DEFINITION = "Definiție Structurală MSS Normal: Un MSS normal necesită ca pivotul (swing high/low) rupt să fie format din minim 2 candele bearish ȘI minim 2 candele bullish."
FVG_STRUCTURAL_DEFINITION = "Definiție Structurală FVG (Fair Value Gap): Este un gap (spațiu gol) între lumânări creat în momentul în care prețul face o mișcare impulsivă, lăsând o zonă netranzacționată."
DISPLACEMENT_DEFINITION = "Definiție Displacement: Este o mișcare continuă a prețului în aceeași direcție, după o structură invalidată, creând FVG-uri (Fair Value Gaps)."

# --- FEW-SHOT EXAMPLES ---
# The assistant_json_output for EACH example below has been updated
# to match the new, simpler JSON structure expected by the vision_system_prompt_template.
FEW_SHOT_EXAMPLES = [
    # --- Example 1: (DE30EUR - Aggressive Short) ---
    {
        "image_url": "https://raw.githubusercontent.com/Ovidiu1731/Trade-images/main/DE30EUR_2025-05-05_12-29-24_69c08.png",
        "assistant_json_output": """
{
  "analysis_possible": true,
  "candle_colors": "Bullish candles are solid green, Bearish candles are solid white.",
  "is_risk_above_entry_suggestion": true,
  "mss_location_description": "MSS is marked by text, breaking the higher low that formed after a sweep of multiple highs (white lines).",
  "mss_pivot_analysis": {
    "description": "The pivot for the MSS (the preceding higher low) appears formed primarily by one bullish (green) candle.",
    "pivot_bearish_count": 0,
    "pivot_bullish_count": 1
  },
  "break_direction_suggestion": "downward",
  "displacement_analysis": { "direction": "bearish", "strength": "moderate" },
  "fvg_analysis": { "count": 2, "description": "Two FVGs are visible after the MSS: one larger gap marked by a grey box during the initial displacement, and a smaller subsequent gap above it. (This aligns with a TG - Two Gap setup)." },
  "liquidity_zones_description": "Multiple liquidity levels above prior highs (marked by horizontal white lines) were swept before the MSS occurred. The last key high swept appears to be around the 23,255-23,260 price level.",
  "liquidity_status_suggestion": "swept",
  "trade_outcome_suggestion": "breakeven",
  "visible_labels_on_chart": ["MSS", "BE"],
  "confidence_level": "high"
}
"""
    },
    # --- Example 2: (Re-entry Normal Long - 07.18.15 copy.jpg) ---
    {
        "image_url": "https://raw.githubusercontent.com/Ovidiu1731/Trade-images/main/Screenshot%202025-05-05%20at%2007.18.15%20copy.png",
        "assistant_json_output": """
{
  "analysis_possible": true,
  "candle_colors": "Bullish candles have a solid white body, Bearish candles have a solid black body.",
  "is_risk_above_entry_suggestion": false,
  "mss_location_description": "MSS breaks the lower high that formed after a re-sweep of liquidity near the 'Local' marked low.",
  "mss_pivot_analysis": { "description": "The pivot is the lower high before the 'MSS' break, following the 'Local' low sweep.", "pivot_bearish_count": 2, "pivot_bullish_count": 3 },
  "break_direction_suggestion": "upward",
  "displacement_analysis": { "direction": "bullish", "strength": "strong" },
  "fvg_analysis": { "count": 2, "description": "Yes, two distinct FVGs (marked with blue boxes) were created during the bullish displacement after the MSS." },
  "liquidity_zones_description": "Initial liquidity sweep occurred below the low marked 'LLB'. Subsequently, liquidity near 'Local' low was re-swept before MSS.",
  "liquidity_status_suggestion": "swept",
  "trade_outcome_suggestion": "potential_setup",
  "visible_labels_on_chart": ["MSS", "Local", "LLB"],
  "confidence_level": "high"
}
"""
    },
    # --- Example 3: (Normal Short with Faint FVGs - TCG - 11.04.35.jpg) ---
    {
        "image_url": "https://github.com/Ovidiu1731/Trade-images/raw/main/Screenshot%202025-05-05%20at%2011.04.35.png",
        "assistant_json_output": """
{
  "analysis_possible": true,
  "candle_colors": "Bullish candles have a solid white body, Bearish candles have a solid black body.",
  "is_risk_above_entry_suggestion": true,
  "mss_location_description": "Downward structure break (marked by arrow) occurs below prior higher low (orange circle), after 'Liq Locala' high sweep.",
  "mss_pivot_analysis": { "description": "Pivot is the higher low marked by orange circle.", "pivot_bearish_count": 3, "pivot_bullish_count": 3 },
  "break_direction_suggestion": "downward",
  "displacement_analysis": { "direction": "bearish", "strength": "strong" },
  "fvg_analysis": { "count": 2, "description": "Yes, two FVGs (marked by faint orange rectangles/lines) after MSS. Aligns with TCG setup." },
  "liquidity_zones_description": "Liquidity above prior swing high (marked 'Liq Locala') was swept before MSS.",
  "liquidity_status_suggestion": "swept",
  "trade_outcome_suggestion": "win",
  "visible_labels_on_chart": ["Liq Locala", "BE"],
  "confidence_level": "medium"
}
"""
    }
]

# ---------------------------------------------------------------------------
# FASTAPI APP LIFECYCLE
# ---------------------------------------------------------------------------
app = FastAPI(title="Trading Instituțional AI Assistant")

@app.on_event("startup")
async def startup_event():
    global aiohttp_session
    aiohttp_session = aiohttp.ClientSession()
    logging.info("aiohttp.ClientSession initialized.")
    
    # Check if Tesseract is available
try:
    test_version = pytesseract.get_tesseract_version()
    logging.info(f"Tesseract OCR available, version: {test_version}")
except pytesseract.TesseractNotFoundError:
    logging.warning("Tesseract OCR not found. OCR functionality will be limited.")
except Exception as e:
    logging.error(f"Error checking Tesseract: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    if aiohttp_session and not aiohttp_session.closed:
        await aiohttp_session.close()
        logging.info("aiohttp.ClientSession closed.")
    if async_openai_client:  # The httpx.AsyncClient is managed by AsyncOpenAI
        await async_openai_client.close()
        logging.info("AsyncOpenAI client closed.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Pydantic Models for Feedback and Requests ---
class FeedbackModel(BaseModel):
    session_id: str
    question: str
    answer: str
    feedback: str
    query_type: Optional[str] = "unknown"
    analysis_data: Optional[Dict] = None

class TextQuery(BaseModel):
    question: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

class ImageHybridQuery(BaseModel):
    question: str
    image_url: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

# --- Feedback Logging ---
def log_feedback(session_id: str, question: str, answer: str, feedback: str,
                 query_type: str, analysis_data: Optional[Dict] = None) -> bool:
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
            # ... existing extraction logic ...
            feedback_entry["analysis_data_from_report"] = analysis_extract
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        logging.error(f"Failed to log feedback: {e}")
        return False

@app.post("/feedback")
async def submit_feedback(feedback_data: FeedbackModel):
    success = await asyncio.to_thread(
        log_feedback,
        feedback_data.session_id,
        feedback_data.question,
        feedback_data.answer,
        feedback_data.feedback,
        feedback_data.query_type,
        feedback_data.analysis_data
    )
    if success:
        return {"status": "success", "message": "Feedback înregistrat."}
    else:
        raise HTTPException(status_code=500, detail="Nu am putut înregistra feedback-ul.")

@app.get("/admin/export-feedback")
async def export_feedback(request: Request, api_key: str = None):
    """Endpoint to export feedback logs securely"""
    # Set a secure API key in Railway variables
    ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
    
    # Validate the API key
    if not ADMIN_API_KEY or api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    try:
        # Check if the feedback log file exists
        if not os.path.exists(FEEDBACK_LOG):
            return {"status": "no_logs", "message": "No feedback logs found"}
        
        # Read the feedback logs
        with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f]
        
        # Return the logs as JSON
        return {
            "status": "success", 
            "count": len(logs), 
            "logs": logs
        }
    except Exception as e:
        logging.error(f"Error exporting feedback logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading logs: {str(e)}")

# --- Query Type Identification ---
def identify_query_type(question: str) -> Dict[str, Any]:
    question_lower = question.lower().strip()
    # ... existing logic ...
    return {"type": "general_image_query"}

# --- Image/JSON Helpers ---
async def download_image_async(image_url: str) -> Optional[bytes]:
    global aiohttp_session
    if not aiohttp_session:
        logging.error("aiohttp_session not initialized for image download attempt.")
        aiohttp_session = aiohttp.ClientSession()
        logging.warning("aiohttp_session re-initialized on-demand in download_image_async.")
    try:
        async with aiohttp_session.get(image_url, timeout=aiohttp.ClientTimeout(total=20)) as response:
            response.raise_for_status()
            logging.info(f"Image {image_url} downloaded successfully.")
            return await response.read()
    except Exception as e:
        logging.error(f"Async image download failed for {image_url}: {e}")
        return None

def _extract_text_from_image_sync(image_content: bytes) -> str:
    try:
        img = Image.open(BytesIO(image_content))
        # ... preprocessing and OCR logic ...
        return cleaned_text
    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract not found. OCR will not function.")
        return ""
    except Exception as e:
        logging.exception(f"OCR failed: {e}")
        return ""

async def extract_text_from_image_async(image_content: bytes) -> str:
    return await asyncio.to_thread(_extract_text_from_image_sync, image_content)

# Also, consider optimizing the image handling
def optimize_image_before_vision(image_content: bytes, max_size: int = 1024*1024) -> bytes:
    """Reduce image size if needed before sending to vision model"""
    if len(image_content) <= max_size:
        return image_content

    try:
        # ... your resizing logic ...
        return optimized_image
    except Exception as e:
        logging.error(f"Image optimization failed: {e}")
        return image_content

        img = Image.open(BytesIO(image_content))
        img_format = img.format
        width, height = img.size
        
        # Calculate new dimensions to maintain aspect ratio
        ratio = min(1.0, math.sqrt(max_size / len(image_content)))
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # Resize image
        resized_img = img.resize((new_width, new_height))
        
        # Save to bytes
        output = BytesIO()
        resized_img.save(output, format=img_format)
        return output.getvalue()
    except Exception as e:
        logging.error(f"Image optimization failed: {e}")
        return image_content  # Return original if optimization fails    

def extract_json_from_text(text: str) -> Optional[str]:
    logging.debug(f"Attempting to extract JSON from text: {text[:200]}...")
    # Priority for markdown ```json ... ```
    json_pattern_markdown = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
    match_md = re.search(json_pattern_markdown, text, re.MULTILINE | re.DOTALL)
    if match_md:
        extracted = match_md.group(1).strip()
def extract_json_from_text(text: str) -> Optional[str]:
    logging.debug(f"Attempting to extract JSON from text: {text[:200]}...")
    # Priority for markdown ```json ... ```
    json_pattern_markdown = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
    match_md = re.search(json_pattern_markdown, text, re.MULTILINE | re.DOTALL)
    if match_md:
        extracted = match_md.group(1).strip()
        try:
            json.loads(extracted)
            logging.info("JSON extracted from markdown code block.")
            return extracted
        except json.JSONDecodeError:
            logging.warning("Text in markdown block wasn't valid JSON. Trying broader search.")

    # Fallback for non-markdown JSON (first '{' to last '}') - more risky but can catch raw output
    start_brace = text.find('{')
    end_brace = text.rfind('}')
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        potential_json = text[start_brace : end_brace+1].strip()
        try:
            json.loads(potential_json)
            logging.info("Potential JSON object found directly in text (fallback) and seems valid.")
            return potential_json
        except json.JSONDecodeError:
            logging.warning("Found brace-enclosed text (fallback), but it's not valid JSON.")

    logging.warning("Could not extract valid JSON object from text using any method.")
    return None

def generate_session_id() -> str:
    return f"{int(time.time())}-{os.urandom(4).hex()}"

# --- Query Expansion Function ---
def expand_query(query: str) -> str:
    """Expand query with trading-specific terminology to improve vector search results."""
    expansion_terms = []
    
    # Common trading abbreviations to expand
    expansions = {
        "OG": ["One Gap Setup", "One Gap", "Gap Setup"],
        "SLG": ["Second Leg Gap", "Second Leg Setup"],
        "TG": ["Two Gap Setup", "Two Gap"],
        "TCG": ["Two Consecutive Gaps", "Consecutive Gaps Setup"],
        "MG": ["Multiple Gaps", "Multiple Gaps Setup"],
        "MSS": ["Market Structure Shift", "structura de piață", "schimbare de structură"],
        "FVG": ["Fair Value Gap", "gap", "spațiu gol"]
    }
    
    # Check for abbreviations to expand
    for abbr, terms in expansions.items():
        if abbr in query.upper().split():
            expansion_terms.extend(terms)
    
    # Add relevant terms based on concepts mentioned
    concept_terms = {
        "setup": ["setup-uri", "tipuri de setup", "pattern", "strategia"],
        "lichid": ["lichidități", "liquidity", "zone de lichiditate"],
        "structur": ["MSS", "market structure shift", "structura"],
        "gap": ["FVG", "Fair Value Gap", "spațiu gol"],
        "long": ["cumpărare", "poziții long", "bullish"],
        "short": ["vânzare", "poziții short", "bearish"]
    }
    
    query_lower = query.lower()
    for key, terms in concept_terms.items():
        if key in query_lower:
            expansion_terms.extend(terms)
    
    # Return expanded string or empty string if no expansion
    return " ".join(expansion_terms)

# --- CV Functions ---
def locate_risk_box_cv(img_np: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Attempt to locate risk and entry boxes on trading charts using color-agnostic edge detection.
    Returns details about potential risk boxes if found.
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply edge detection (Canny)
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter for rectangular contours - potential risk boxes
        candidates = []
        for contour in contours:
            # Check if contour approximates a rectangle
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            if len(approx) == 4:  # If it's a rectangular shape
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                ratio = float(w) / h if h > 0 else 0

                # Filter for reasonable box sizes (not too small, reasonable aspect ratio)
                if area > 500 and 0.1 < ratio < 10:
                    # Extract the region of interest
                    roi = gray[y:y+h, x:x+w]
                    if roi.size > 0:
                        # Calculate average pixel value (brightness)
                        avg_value = np.mean(roi)
                        candidates.append({
                            'x': x, 'y': y, 'width': w, 'height': h,
                            'area': area, 'aspect_ratio': ratio,
                            'avg_value': avg_value
                        })

        if not candidates:
            return None

        # Sort by characteristics typical of risk boxes - typically wider than tall,
        # and often in the right part of the chart
        candidates.sort(key=lambda c: (c['aspect_ratio'] > 1.5, c['x'] > img_np.shape[1]/2, -c['area']), reverse=True)

        best_candidate = candidates[0]
        return {
            'risk_box_found': True,
            'coordinates': {
                'x': best_candidate['x'],
                'y': best_candidate['y'],
                'width': best_candidate['width'],
                'height': best_candidate['height']
            },
            'confidence': 'medium',
            'detection_method': 'edge_detection'
        }
    except Exception as e:
        logging.error(f"Risk box CV error: {e}")
        return None

def cv_pre_process_image(image_content: bytes) -> Dict[str, Any]:
    """
    Pre-process image with CV to extract structural information before sending to vision model.
    Performs:
    1. Risk/entry box detection
    2. Color classification (candle colors)
    3. Text extraction for labels
    """
    results = {
        "cv_analysis_performed": True,
        "risk_box_detected": False,
        "candle_colors": "not_determined",
        "extracted_text": "",
        "_cv_note": "Computer vision pre-processing completed"
    }

    try:
        # Convert image content to numpy array
        img_array = np.frombuffer(image_content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            results["_cv_note"] = "Failed to decode image"
            return results

        # Find risk box if possible
        risk_box_result = locate_risk_box_cv(img)
        if risk_box_result:
            results["risk_box_detected"] = True
            results["risk_box_info"] = risk_box_result["coordinates"]
            results["risk_box_confidence"] = risk_box_result["confidence"]

        # Attempt to detect candle colors
        # Sample dominant colors in the chart area
        height, width = img.shape[:2]
        center_area = img[int(height*0.25):int(height*0.75), int(width*0.25):int(width*0.75)]

        if center_area.size > 0:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(center_area, cv2.COLOR_BGR2HSV)

            # Simple color detection logic
            # Look for common trading chart color schemes
            green_mask = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255))
            red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255)) | cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
            black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))  # Dark colors
            white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))  # Light colors

            # Count pixels for each color
            green_count = np.count_nonzero(green_mask)
            red_count = np.count_nonzero(red_mask)
            black_count = np.count_nonzero(black_mask)
            white_count = np.count_nonzero(white_mask)

            # Determine color scheme based on counts
            if green_count > 1000 and red_count > 1000:
                results["candle_colors"] = "green_red"
            elif green_count > 1000 and black_count > 1000:
                results["candle_colors"] = "green_black"
            elif green_count > 1000 and white_count > 1000:
                results["candle_colors"] = "green_white"
            elif white_count > 1000 and black_count > 1000:
                results["candle_colors"] = "white_black"
            else:
                results["candle_colors"] = "not_determined"

            results["_cv_note"] += f", Color analysis performed: {results['candle_colors']}"

        # Extract text from image using the already-implemented async function
        # For CV preprocessing we'll use a synchronous version
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            extracted_text = pytesseract.image_to_string(img_pil, lang="eng")
            cleaned_text = "".join(ch for ch in extracted_text if ord(ch) < 128).strip()
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

            # Look for specific trading labels
            results["detected_labels"] = []
            for keyword in ["MSS", "FVG", "BE", "SL", "TP", "Win", "Loss", "Risk"]:
                if keyword in cleaned_text or keyword.lower() in cleaned_text:
                    results["detected_labels"].append(keyword)

            if results["detected_labels"]:
                results["_cv_note"] += f", Labels detected: {results['detected_labels']}"

            results["extracted_text"] = cleaned_text
        except Exception as ocr_e:
            results["_cv_note"] += f", OCR failed: {str(ocr_e)}"

        return results
    except Exception as e:
        logging.error(f"CV pre-processing error: {e}")
        results["_cv_note"] = f"CV analysis failed: {str(e)}"
        return results

# --- Rule Engine Helpers & Main Function ---
def classify_mss_type(bullish_count: Optional[Any], bearish_count: Optional[Any],
                      is_last_lh_hl_broken: bool = True # Vision model might need to provide this if rule is strict
                     ) -> str:
    if bullish_count is None or bearish_count is None: return "not_identified"
    try:
        bull, bear = int(bullish_count), int(bearish_count)
        # is_last_lh_hl_broken is True by default. If it's a strict rule, Vision model should suggest this.
        if is_last_lh_hl_broken and bull >= 2 and bear >= 2: return "normal"
        return "agresiv"
    except (ValueError, TypeError):
        logging.warning("RuleEngine: Could not parse MSS pivot counts."); return "not_identified"

def apply_rule_engine(vision_json: Dict[str, Any], cv_findings: Dict[str, Any], query_type: str = "trade_evaluation_image_query") -> Dict[str, Any]:
    """
    Apply rule engine with awareness of query type
    to make deterministic trading decisions based on the rules.

    Returns an enriched analysis with rule-based decisions.
    """
    final_analysis = copy.deepcopy(vision_json)  # Start with vision model output
    final_analysis["_rule_engine_notes"] = f"Rule engine applied for {query_type}"

    # --- Merge CV findings where applicable ---
    if cv_findings and cv_findings.get("cv_analysis_performed", False):
        # Trust CV candle colors if vision didn't determine them
        if cv_findings.get("candle_colors") != "not_determined" and (
           not final_analysis.get("candle_colors") or
           final_analysis.get("candle_colors") == "unknown"): # Vision might return "unknown"
            final_analysis["candle_colors"] = cv_findings["candle_colors"]
            final_analysis["_rule_engine_notes"] += ", Used CV candle colors"

        # Add detected labels from CV if any
        if cv_findings.get("detected_labels"):
            final_analysis["visible_labels_on_chart"] = list(set(
                final_analysis.get("visible_labels_on_chart", []) +
                cv_findings.get("detected_labels", [])
            ))
            final_analysis["_rule_engine_notes"] += ", Added CV-detected labels"

        # Add CV risk box info if detected
        if cv_findings.get("risk_box_detected", False):
            final_analysis["risk_box_info_cv"] = cv_findings.get("risk_box_info", {}) # Distinguish CV risk box
            final_analysis["_rule_engine_notes"] += ", Added risk box data from CV"

    # For concept verification, apply only the relevant rules and skip trade-specific analysis
    if query_type == "concept_verification_image_query" or query_type == "general_image_query":
        # Check if this is a liquidity-related question
        if "liquidity" in final_analysis.get("liquidity_zones_description", "").lower() or \
           "lichidit" in final_analysis.get("liquidity_zones_description", "").lower():
            # Only process liquidity-related rules
            liquidity_status = final_analysis.get("liquidity_status_suggestion", "unknown")
            if liquidity_status not in ["swept", "not_swept", "unclear"]:
                # Try to infer from description if not explicitly set
                liquidity_desc = final_analysis.get("liquidity_zones_description", "").lower()
                if "swept" in liquidity_desc or "taken" in liquidity_desc:
                    final_analysis["liquidity_status_suggestion"] = "swept"
                elif "not swept" in liquidity_desc or "not taken" in liquidity_desc:
                    final_analysis["liquidity_status_suggestion"] = "not_swept"
                else:
                    final_analysis["liquidity_status_suggestion"] = "unclear"
            
            # Add a focused liquidity analysis summary
            final_analysis["focused_analysis"] = {
                "type": "liquidity_verification",
                "liquidity_zones_present": bool(final_analysis.get("liquidity_zones_description")),
                "likely_correct": final_analysis.get("confidence_level", "low") != "low"
            }
            
            final_analysis["_rule_engine_notes"] += ", Applied focused liquidity analysis for concept verification"
            return final_analysis
            
        # For FVG verification questions
        elif "fvg" in str(final_analysis).lower() or "fair value gap" in str(final_analysis).lower():
            # Process only FVG-related fields
            fvg_analysis = final_analysis.get("fvg_analysis", {})
            fvg_count = fvg_analysis.get("count", 0)
            
            if isinstance(fvg_count, int) and fvg_count >= 1:
                final_analysis["has_valid_fvg"] = True
            else:
                final_analysis["has_valid_fvg"] = False
            
            # Add a focused FVG analysis summary
            final_analysis["focused_analysis"] = {
                "type": "fvg_verification",
                "fvg_present": final_analysis.get("has_valid_fvg", False),
                "count": fvg_count,
                "likely_correct": final_analysis.get("confidence_level", "low") != "low"
            }
            
            final_analysis["_rule_engine_notes"] += ", Applied focused FVG analysis for concept verification"
            return final_analysis
            
        # For MSS verification questions
        elif "mss" in str(final_analysis).lower() or "structure" in str(final_analysis).lower():
            # Process only MSS-related fields
            mss_pivot_data = final_analysis.get("mss_pivot_analysis", {})
            pivot_bearish_count = mss_pivot_data.get("pivot_bearish_count")
            pivot_bullish_count = mss_pivot_data.get("pivot_bullish_count")
            
            # Determine MSS type based on pivot counts
            final_analysis["final_mss_type"] = classify_mss_type(
                pivot_bullish_count,
                pivot_bearish_count
            )
            
            # Add a focused MSS analysis summary
            final_analysis["focused_analysis"] = {
                "type": "mss_verification",
                "mss_type": final_analysis["final_mss_type"],
                "likely_correct": final_analysis.get("confidence_level", "low") != "low"
            }
            
            final_analysis["_rule_engine_notes"] += ", Applied focused MSS analysis for concept verification"
            return final_analysis

    # If it's a trade evaluation or the query type doesn't match any concept verification,
    # proceed with the full rule engine analysis
    
    # --- Apply MSS type classification rule ---
    mss_pivot_data = final_analysis.get("mss_pivot_analysis", {})
    pivot_bearish_count = mss_pivot_data.get("pivot_bearish_count")
    pivot_bullish_count = mss_pivot_data.get("pivot_bullish_count")

    # Determine MSS type based on pivot counts
    final_analysis["final_mss_type"] = classify_mss_type(
        pivot_bullish_count,
        pivot_bearish_count
    )
    final_analysis["_rule_engine_notes"] += f", MSS classified as {final_analysis['final_mss_type']}"

    # --- Determine trade direction ---
    # Use vision model's break direction suggestion as primary input
    break_direction = final_analysis.get("break_direction_suggestion", "unknown")
    if break_direction == "upward":
        final_analysis["final_trade_direction"] = "long"
    elif break_direction == "downward":
        final_analysis["final_trade_direction"] = "short"
    else:
        final_analysis["final_trade_direction"] = "unknown"
        final_analysis["_rule_engine_notes"] += ", Could not determine trade direction"

    # --- Determine trade outcome ---
    # Use vision model's trade outcome suggestion as primary input
    outcome_suggestion = final_analysis.get("trade_outcome_suggestion", "unknown")

    if outcome_suggestion in ["win", "breakeven", "loss", "potential_setup"]:
        final_analysis["final_trade_outcome"] = outcome_suggestion
    else:
        # Check visible labels for outcome hints
        labels = final_analysis.get("visible_labels_on_chart", [])
        if any(label.upper() in ["WIN", "W"] for label in labels):
            final_analysis["final_trade_outcome"] = "win"
        elif any(label.upper() in ["BE", "BREAKEVEN"] for label in labels):
            final_analysis["final_trade_outcome"] = "breakeven"
        elif any(label.upper() in ["LOSS", "L"] for label in labels):
            final_analysis["final_trade_outcome"] = "loss"
        else:
            final_analysis["final_trade_outcome"] = "unknown" # Default to unknown if no clear indicators
            # Only add note if outcome_suggestion was also unknown
            if outcome_suggestion == "unknown":
                 final_analysis["_rule_engine_notes"] += ", Could not determine trade outcome from suggestions or labels"

    # --- Check liquidity status ---
    liquidity_status = final_analysis.get("liquidity_status_suggestion", "unknown")
    if liquidity_status not in ["swept", "not_swept", "unclear"]:
        # Try to infer from description if not explicitly set
        liquidity_desc = final_analysis.get("liquidity_zones_description", "").lower()
        if "swept" in liquidity_desc or "taken" in liquidity_desc:
            final_analysis["liquidity_status_suggestion"] = "swept"
        elif "not swept" in liquidity_desc or "not taken" in liquidity_desc:
            final_analysis["liquidity_status_suggestion"] = "not_swept"
        else:
            final_analysis["liquidity_status_suggestion"] = "unclear"

    # --- Validate displacement direction vs trade direction ---
    displacement = final_analysis.get("displacement_analysis", {})
    displacement_direction = displacement.get("direction", "unknown")

    # Displacement direction should match trade direction for valid setups
    if final_analysis["final_trade_direction"] != "unknown" and displacement_direction != "unknown":
        if (final_analysis["final_trade_direction"] == "long" and displacement_direction == "bullish") or \
           (final_analysis["final_trade_direction"] == "short" and displacement_direction == "bearish"):
            final_analysis["displacement_matches_trade"] = True
        else:
            final_analysis["displacement_matches_trade"] = False
            final_analysis["_rule_engine_notes"] += ", Warning: Displacement direction doesn't match trade direction"
    else:
        final_analysis["displacement_matches_trade"] = "not_applicable" # If either is unknown

    # --- Validate FVG presence ---
    fvg_analysis = final_analysis.get("fvg_analysis", {})
    fvg_count = fvg_analysis.get("count", 0) # Assuming count is a number

    if isinstance(fvg_count, int) and fvg_count >= 1: # Check type
        final_analysis["has_valid_fvg"] = True
    else:
        final_analysis["has_valid_fvg"] = False
        final_analysis["_rule_engine_notes"] += ", Warning: No valid FVGs detected or count is not a number"

    # --- Calculate final validity score ---
    validity_score = 50 # Base score

    # Add points for important criteria
    if final_analysis.get("final_mss_type") != "not_identified": validity_score += 10
    if final_analysis.get("final_trade_direction") != "unknown": validity_score += 10
    if final_analysis.get("liquidity_status_suggestion") == "swept": validity_score += 10
    if final_analysis.get("has_valid_fvg") is True: validity_score += 10 # Check for explicit True
    if final_analysis.get("displacement_matches_trade") is True: validity_score += 10 # Check for explicit True

    # Deduct points for negative indicators if any were planned
    # Example: if final_analysis.get("some_negative_indicator"): validity_score -= 5

    validity_score = min(max(validity_score, 0), 100) # Ensure score is between 0 and 100
    final_analysis["setup_validity_score"] = validity_score

    if validity_score >= 80: final_analysis["setup_quality_summary"] = "high_quality"
    elif validity_score >= 60: final_analysis["setup_quality_summary"] = "acceptable"
    else: final_analysis["setup_quality_summary"] = "questionable"

    return final_analysis

# --- System Prompt Builder ---
def _build_system_prompt(query_type: str, requires_full_analysis: bool) -> str:
    """
    Build a system prompt for the completion model based on query type and analysis requirements.
    """
    full_prompt = SYSTEM_PROMPT_CORE

    if query_type == "trade_evaluation_image_query":
        full_prompt += "\n\n" + (
            "The user has submitted a chart for trading analysis. "
            "Focus on analyzing the trading setup presented in the chart according to the rules of Trading Instituțional. "
            "Base your analysis primarily on the Technical Analysis Report which contains computer vision results, "
            "vision model findings, and rule engine determinations."
        )
        full_prompt += "\n\n" + (
            "KEY TRADING RULES TO REFERENCE:\n"
            f"1. {MSS_NORMAL_STRUCTURAL_DEFINITION}\n"
            f"2. {MSS_AGRESIV_STRUCTURAL_DEFINITION}\n"
            f"3. {FVG_STRUCTURAL_DEFINITION}\n"
            f"4. {DISPLACEMENT_DEFINITION}\n"
            "5. A proper setup requires sweeping liquidity before the MSS occurs.\n"
            "6. A valid setup should show at least one FVG after the MSS.\n"
            "7. The displacement after the MSS should match the trading direction (bullish for longs, bearish for shorts)."
        )
        if requires_full_analysis:
            full_prompt += "\n\n" + (
                "Provide a COMPLETE and DETAILED analysis that includes:\n"
                "1. Identification of the MSS type (Normal or Agresiv) based on pivot structure\n"
                "2. Trade direction (Long/Short)\n"
                "3. Liquidity status (Swept/Not Swept)\n"
                "4. FVG presence and characteristics\n"
                "5. Overall validity of the setup (refer to 'setup_quality_summary' and 'setup_validity_score' from the report)\n"
                "6. Trade outcome if visible (Win/Loss/Breakeven)"
            )
        else: # User asked a specific question about a trade eval image
            full_prompt += "\n\n" + (
                "The user has asked a specific question about this trading chart. "
                "Focus on answering their question using the Technical Analysis Report. "
                "You do not need to provide a full, unsolicited analysis unless it directly helps answer the question. "
                "Tailor your response to be concise and targeted."
            )

    elif query_type == "general_image_query":
        full_prompt += "\n\n" + (
            "The user has asked a general question about a trading chart. "
            "Answer their question directly without necessarily providing a full trade setup analysis. "
            "You can reference any visible elements from the chart described in the Technical Analysis Report if relevant. "
            "Keep your response focused and concise, addressing only what was asked."
        )
    # Add general response guidance (common to all image queries)
    if "image_query" in query_type: # Apply to both image query types
        full_prompt += "\n\n" + (
            "GENERAL RESPONSE GUIDELINES FOR IMAGE ANALYSIS:\n"
            "- Always maintain an objective, educational tone (like an experienced trading colleague).\n"
            "- Avoid overly technical language unless the user uses it first - explain concepts in clear, accessible terms.\n"
            "- When referencing specific rules, briefly explain their importance in context.\n"
            "- If the Technical Analysis Report indicates low confidence, missing information, or contradictions, acknowledge these limitations candidly.\n"
            "- Structure your response logically. Use short paragraphs for readability.\n"
            "- Use first-person phrasing occasionally to sound natural (e.g., 'I can see that...' or 'I notice the chart shows...').\n"
            "- Conclude with a brief summary or key takeaway if appropriate for the question.\n"
            "- If applicable, explain what learning can be taken from this example."
        )
    return full_prompt

def get_vision_system_prompt(query_type: str, question: str) -> str:
    """
    Generate a specialized vision system prompt based on query type and question content.
    This allows for more focused analysis on specific concepts when needed.
    
    Args:
        query_type: The type of query (e.g., "concept_verification_image_query", "trade_evaluation_image_query")
        question: The user's question text
        
    Returns:
        A specialized system prompt for the vision model
    """
    # Base shared prompt components
    base_prompt = """You are an expert Trading Analysis Assistant for Trading Instituțional. You will analyze trading charts
and identify key trading elements based on the specific question asked."""
    
    # Standard JSON structure for all responses
    base_json_structure = """
IMPORTANT: Your response must be a valid JSON object with this base structure:
{
  "analysis_possible": true/false,
  "confidence_level": "low"/"medium"/"high"
}
"""
    
    # For liquidity concept verification
    if query_type == "concept_verification_image_query" or query_type == "general_image_query":
        question_lower = question.lower()
        
        # Detect if this is about liquidity
        if "liquidity" in question_lower or "lichidit" in question_lower or "liq" in question_lower:
            return base_prompt + """
For this liquidity concept verification question, focus ONLY on:
- Identifying liquidity zones in the chart (horizontal lines at swing highs/lows)
- Whether they are correctly marked on the chart
- Whether they have been swept (price has moved beyond them) or not
- The color and placement of the liquidity markers

DO NOT analyze trade direction, MSS type, or other setups unless specifically asked.
""" + base_json_structure + """
Your JSON response should include these specific fields:
{
  "analysis_possible": true/false,
  "candle_colors": "description of candle colors",
  "liquidity_zones_description": "detailed description of liquidity zones on the chart and if marked correctly",
  "liquidity_status_suggestion": "swept"/"not_swept"/"unclear",
  "visible_labels_on_chart": ["any visible text labels"],
  "confidence_level": "low"/"medium"/"high"
}
"""

        # Detect if this is about FVGs (Fair Value Gaps)
        elif "fvg" in question_lower or "fair value gap" in question_lower or "gap" in question_lower:
            return base_prompt + """
For this FVG (Fair Value Gap) verification question, focus ONLY on:
- Identifying FVGs in the chart (gaps between candles during impulsive moves)
- Whether they are correctly marked on the chart
- The number and size of FVGs present
- Whether they have been filled or not

DO NOT analyze trade direction, MSS type, or other setups unless specifically asked.
""" + base_json_structure + """
Your JSON response should include these specific fields:
{
  "analysis_possible": true/false,
  "candle_colors": "description of candle colors",
  "fvg_analysis": { "count": number, "description": "description of FVGs and whether they're correctly marked" },
  "visible_labels_on_chart": ["any visible text labels"],
  "confidence_level": "low"/"medium"/"high"
}
"""

        # Detect if this is about MSS (Market Structure Shifts)
        elif "mss" in question_lower or "market structure" in question_lower or "structure" in question_lower:
            return base_prompt + """
For this MSS (Market Structure Shift) verification question, focus ONLY on:
- Identifying MSS points in the chart
- Analyzing the pivot structure (number of bullish/bearish candles)
- Whether it's a normal or aggressive MSS
- The direction of the break (upward/downward)

DO NOT conduct a full trade analysis unless specifically asked.
""" + base_json_structure + """
Your JSON response should include these specific fields:
{
  "analysis_possible": true/false,
  "candle_colors": "description of candle colors",
  "mss_location_description": "description of where the MSS is located",
  "mss_pivot_analysis": {
    "description": "description of the pivot structure",
    "pivot_bearish_count": number,
    "pivot_bullish_count": number
  },
  "break_direction_suggestion": "upward"/"downward"/"unclear",
  "visible_labels_on_chart": ["any visible text labels"],
  "confidence_level": "low"/"medium"/"high"
}
"""
        # General, non-specific verification question - return a simplified prompt
        else:
            return base_prompt + """
Focus on answering the specific question about the chart elements. Provide relevant details
about the elements mentioned in the question without doing a full trade analysis.
""" + base_json_structure + """
Include relevant fields in your JSON response based on the question asked.
"""

    # Default to full trade analysis prompt for trade evaluation
    vision_system_prompt_template = base_prompt + """
Provide a comprehensive analysis of the trading chart, focusing on these key aspects:
- Market Structure Shifts (MSS) - both Normal and Aggressive types
- Candle and color patterns
- Direction of the break (upward/downward)
- FVG (Fair Value Gap) presence
- Liquidity zones and whether they've been swept
- Risk placement (above/below entry)
- Potential trade outcome if visible

Definitions:
- MSS (Market Structure Shift): A break of market structure that signals a potential trend change.
- Normal MSS: The pivot (higher low or lower high) that is broken must have at least 2 bearish AND 2 bullish candles.
- Aggressive MSS: The pivot has fewer than 2 bearish OR fewer than 2 bullish candles.
- FVG (Fair Value Gap): An unfilled gap created when price makes an impulsive move.
- Liquidity: Price levels where stop losses or take profits are clustered.
""" + base_json_structure + """
Your JSON response MUST include the following full structure:
{
  "analysis_possible": true/false,
  "candle_colors": "description of bullish/bearish candle colors or 'unknown'",
  "is_risk_above_entry_suggestion": true/false/null,
  "mss_location_description": "description of where the MSS is located",
  "mss_pivot_analysis": {
    "description": "description of the pivot structure",
    "pivot_bearish_count": number,
    "pivot_bullish_count": number
  },
  "break_direction_suggestion": "upward"/"downward"/"unclear",
  "displacement_analysis": { "direction": "bullish"/"bearish"/"unclear", "strength": "weak"/"moderate"/"strong" },
  "fvg_analysis": { "count": number, "description": "description of FVGs present" },
  "liquidity_zones_description": "description of liquidity zones and if they were swept",
  "liquidity_status_suggestion": "swept"/"not_swept"/"unclear",
  "trade_outcome_suggestion": "win"/"loss"/"breakeven"/"potential_setup"/"unknown",
  "visible_labels_on_chart": ["MSS", "BE", etc.],
  "confidence_level": "low"/"medium"/"high"
}
"""
    
    # We now need to add the few-shot examples
    for example in FEW_SHOT_EXAMPLES:
        vision_system_prompt_template += f"\n\nImage URL: {example['image_url']}\nAnalysis:\n{example['assistant_json_output']}\n"
    
    vision_system_prompt_template += """
Analyze the provided chart image following the same format as the examples.
Provide only the JSON output, no additional text before or after.
If a field cannot be determined, use "unknown" or null where appropriate for the type.
"""
    
    return vision_system_prompt_template

# --- API Routes ---
@app.post("/ask", response_model=Dict[str, Any]) # Return type more flexible
async def ask_question(query: TextQuery):
    start_time = time.time()
    session_id = query.session_id or generate_session_id()
    question = query.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Întrebarea nu poate fi goală.")

    history_store_key = f"text_only:{session_id}" # Separate history for text-only
    history = []
    if query.conversation_history: # Allow user to send full history if they manage it client-side
        history = query.conversation_history[-MAX_HISTORY_MESSAGES:]
    elif history_store_key in conversation_history:
        history = conversation_history[history_store_key][-MAX_HISTORY_MESSAGES:]

    logging.info(f"Text query received. Session: {session_id}, History length: {len(history)}")

    # NEW — expand abbreviations for better retrieval 
    expanded = expand_query(question)
    search_query = f"{question} {expanded}".strip()
    context_text = "" # Initialize context_text

    try:
        async with openai_call_limiter:
            embedding_response = await async_openai_client.embeddings.create(
                input=search_query, model=EMBEDDING_MODEL
            )
        query_vector = embedding_response.data[0].embedding

        pinecone_results = await asyncio.to_thread(
            index.query,
            vector=query_vector,
            top_k=5,  # You can define TOP_K as a constant at the top of your file
            include_metadata=True
        )
        
        logging.info([m.score for m in pinecone_results.matches])
        # First collect all chunks regardless of score
        all_chunks = [
            match.metadata["text"] for match in pinecone_results.matches
            if match.metadata and "text" in match.metadata
        ]
        # No filtering
        if all_chunks:
            context_text = "\n\n".join(all_chunks)
            logging.info(f"Using all {len(all_chunks)} context chunks for query.")
        else:
            context_text = ""
        logging.info(f"Retrieved {len(all_chunks)} relevant context chunks for text query.")

    except Exception as pe:
        logging.error(f"Pinecone vector search error: {pe}")
        context_text = "Am întâmpinat o problemă la accesarea materialului de curs din baza de date Pinecone."
    except APIError as ae: # More specific OpenAI error
        logging.error(f"OpenAI API error during embedding: {ae}")
        raise HTTPException(status_code=503, detail="Serviciul OpenAI (embeddings) nu răspunde. Te rog să încerci mai târziu.")
    except RateLimitError:
        logging.warning("OpenAI rate limit hit during text query embedding.")
        raise HTTPException(status_code=429, detail="Prea multe solicitări către OpenAI. Te rog să încerci mai târziu.")
    except Exception as e: # Catch-all for other unexpected errors
        logging.error(f"Unexpected vector search error: {e}")
        context_text = "A apărut o eroare neașteptată la căutarea informațiilor."


    messages = [{"role": "system", "content": SYSTEM_PROMPT_CORE}] # Use core prompt for text
    for turn in history:
        messages.append({"role": "user", "content": turn.get("user", "")})
        if "assistant" in turn: # Ensure assistant message exists
            messages.append({"role": "assistant", "content": turn.get("assistant", "")})

    ABBREV_MAP = """
    TG  = Two Gap Setup (Two Consecutive Gap)
    TCG = Two Consecutive Gap Setup
    3CG = Three Consecutive Gap Setup
    SLG = Second Leg Setup
    """.strip()   

    if context_text and "problemă" not in context_text and "eroare" not in context_text: # Check if context is useful
        current_prompt = (
            f"Întrebare: {question}\n\n"
            "### GLOSAR\n"
            f"{ABBREV_MAP}\n\n"
            "### CONTEXT (copiază exact formulările)\n"
            f"{context_text}\n"
            "### END CONTEXT\n\n"
            "Folosind **doar informațiile din CONTEXT**, răspunde la întrebarea utilizatorului. "
            "Nu inventa termeni; citează formulările exact așa cum apar. "
            "Dacă informația nu există în CONTEXT, spune explicit „Informația nu e disponibilă în materialul de curs."
            "Răspuns concis, stil Trading Instituțional."
        )
    else:
        current_prompt = f"Întrebare: {question}\n\n{context_text}\nRăspunde la întrebarea utilizatorului pe baza cunoștințelor tale generale despre trading și conversația anterioară, respectând stilul Trading Instituțional. Fii concis și la obiect."
    
    messages.append({"role": "user", "content": current_prompt})
    
    logging.info(f"\n─── CONTEXT SENT TO LLM ───\n{context_text}\n────────────────────────")

    try:
        async with openai_call_limiter:
            completion = await async_openai_client.chat.completions.create(
                model=TEXT_MODEL, messages=messages, temperature=0.5, max_tokens=800
            )
        answer = completion.choices[0].message.content.strip()

        if history_store_key not in conversation_history:
            conversation_history[history_store_key] = deque(maxlen=MAX_HISTORY_MESSAGES)
        conversation_history[history_store_key].append({"user": question, "assistant": answer})

        duration_ms = int((time.time() - start_time) * 1000)
        logging.info(f"Text query completed in {duration_ms}ms. Session: {session_id}")
        return {"answer": answer, "session_id": session_id, "query_type": "text_only", "processing_time_ms": duration_ms}

    except RateLimitError:
        logging.warning("OpenAI rate limit hit during text query completion.")
        raise HTTPException(status_code=429, detail="Prea multe solicitări către OpenAI. Te rog să încerci mai târziu.")
    except APIError as e:
        logging.error(f"OpenAI API error during completion: {e}")
        raise HTTPException(status_code=503, detail=f"Serviciul OpenAI nu răspunde: {e}")
    except Exception as e:
        logging.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail="A apărut o eroare la procesarea întrebării.")

    try:
        async with openai_call_limiter:
            completion = await async_openai_client.chat.completions.create(
                model=TEXT_MODEL,
                messages=messages,
                temperature=0.5,
                max_tokens=800
            )
        answer = completion.choices[0].message.content.strip()

        if history_store_key not in conversation_history:
            conversation_history[history_store_key] = deque(maxlen=MAX_HISTORY_MESSAGES)
        conversation_history[history_store_key].append({"user": question, "assistant": answer})

        duration_ms = int((time.time() - start_time) * 1000)
        logging.info(f"Text query completed in {duration_ms}ms. Session: {session_id}")
        return {"answer": answer, "session_id": session_id, "query_type": "text_only", "processing_time_ms": duration_ms}

    except RateLimitError:
        logging.warning("OpenAI rate limit hit during text query completion.")
        raise HTTPException(status_code=429, detail="Prea multe solicitări către OpenAI. Te rog să încerci mai târziu.")
    except APIError as e:
        logging.error(f"OpenAI API error during completion: {e}")
        raise HTTPException(status_code=503, detail=f"Serviciul OpenAI nu răspunde: {e}")
    except Exception as e:
        logging.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail="A apărut o eroare la procesarea întrebării.")

@app.post("/ask-image-hybrid", response_model=Dict[str, Any]) # Return type more flexible
async def ask_image_hybrid(payload: ImageHybridQuery) -> Dict[str, Any]:
    start_time = time.time()
    session_id = payload.session_id or generate_session_id()
    question = payload.question.strip()
    expanded = expand_query(question)
    image_url = payload.image_url.strip()

    if not question: raise HTTPException(status_code=400, detail="Întrebarea nu poate fi goală.")
    if not image_url: raise HTTPException(status_code=400, detail="URL-ul imaginii nu poate fi gol.")

    history_store_key = f"image_hybrid:{session_id}" # Separate history for image queries
    history = []
    if payload.conversation_history:
        history = payload.conversation_history[-MAX_HISTORY_MESSAGES:]
    elif history_store_key in conversation_history:
        history = conversation_history[history_store_key][-MAX_HISTORY_MESSAGES:]
    logging.info(f"Image-hybrid query received. Session: {session_id}, History length: {len(history)}")

    query_info = identify_query_type(question)
    query_type = query_info.get("type", "general_image_query")
    requires_full_analysis = query_type == "trade_evaluation_image_query"

    image_content = await download_image_async(image_url)
    if not image_content:
        raise HTTPException(status_code=400, detail="Nu am putut descărca imaginea. Verifică URL-ul și încearcă din nou.")

    # --- CV, OCR, Vision, Rule Engine ---
    # These can potentially run in parallel if structured carefully, but for now sequential
    cv_analysis = await asyncio.to_thread(cv_pre_process_image, image_content)
    ocr_text = await extract_text_from_image_async(image_content) # General OCR of the whole image
    logging.info(f"OCR (full image) extracted text length: {len(ocr_text)}")


    # Vision Model Analysis
    vision_json = {"analysis_possible": False, "_vision_note": "Vision analysis not performed yet"}
    cache_key = f"{image_url}:{query_type}"
    # Check if we have a cached result
    if cache_key in vision_results_cache:
        vision_json = vision_results_cache[cache_key]
        logging.info(f"Using cached vision analysis for {cache_key}")
    else:
        try:
            # Get specialized prompt based on query type
            vision_system_prompt_template = get_vision_system_prompt(query_type, question)
            
            # Optimize the image before sending to vision model
            optimized_image_content = await asyncio.to_thread(optimize_image_before_vision, image_content)
            if optimized_image_content != image_content:
                logging.info(f"Image optimized: original size {len(image_content)} bytes, new size {len(optimized_image_content)} bytes")

            content_items = [
                {"type": "text", "text": vision_system_prompt_template},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(optimized_image_content).decode('utf-8')}"}}
            ]
            
            async with openai_call_limiter:
                vision_response = await async_openai_client.chat.completions.create(
                    model=VISION_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a trading chart analysis assistant that outputs structured JSON."},
                        {"role": "user", "content": content_items}
                    ],
                    max_tokens=2000, temperature=0.1 # Very low temp for structured JSON
                )
            
            vision_analysis_text = vision_response.choices[0].message.content
            vision_json_str = extract_json_from_text(vision_analysis_text)
        
            if not vision_json_str:
                vision_json = {"analysis_possible": False, "_vision_note": "Could not extract JSON from vision model response."}
            else:
                try:
                    vision_json = json.loads(vision_json_str)
                    vision_json["_vision_note"] = "Vision analysis completed."
                    logging.info("Vision model analysis extracted successfully.")
                    # Cache the successful result
                    vision_results_cache[cache_key] = vision_json
                
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse vision model JSON: {vision_json_str[:500]}")
                    # Try to salvage partial JSON
                    try:
                        # Look for patterns that might indicate valid but incomplete JSON
                        if '{' in vision_json_str and '"analysis_possible"' in vision_json_str:
                            # Try to clean up and repair common JSON formatting issues
                            cleaned_str = re.sub(r',\s*}', '}', vision_json_str)  # Remove trailing commas
                            cleaned_str = re.sub(r',\s*]', ']', cleaned_str)     # Remove trailing commas in arrays
                             # Find the largest valid JSON subset
                            start_idx = vision_json_str.find('{')
                            end_idx = vision_json_str.rfind('}')
                            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                                partial_json_str = cleaned_str[start_idx:end_idx+1]
                                vision_json = json.loads(partial_json_str)
                                vision_json["_vision_note"] = "Partial vision analysis (recovered from malformed JSON)"
                                logging.info("Recovered partial JSON from vision model response")
                            else:
                                raise ValueError("Could not find valid JSON object boundaries")   
                        else:
                            raise ValueError("No valid JSON pattern found")   
                    except Exception as recovery_error:
                        logging.error(f"JSON recovery attempt failed: {recovery_error}")        
                        #Fall back to a basic structure
                        vision_json = {
                            "analysis_possible": False,
                            "confidence_level": "low",
                            "_vision_note": f"Failed to parse JSON from vision model: {str(recovery_error)}"
                        }
                
        except RateLimitError:
            logging.warning("OpenAI rate limit hit during vision analysis.")
            raise HTTPException(status_code=429, detail="Prea multe solicitări către OpenAI (Vision). Te rog să încerci mai târziu.")
        except APIError as e:
            logging.error(f"OpenAI API error during vision analysis: {e}")
            vision_json = {"analysis_possible": False, "_vision_note": f"Vision analysis API error: {str(e)}"}
        except Exception as e:
            logging.error(f"Unexpected vision model error: {e}")
            vision_json = {"analysis_possible": False, "_vision_note": f"Unexpected vision analysis error: {str(e)}"}

        if not vision_json_str:
            logging.warning("Could not extract JSON from vision model. Creating basic fallback structure.")
    
            # Create a fallback JSON structure based on the query type and question
            question_lower = question.lower()
    
            vision_json = {
                "analysis_possible": True,
                "candle_colors": "unknown",
                "confidence_level": "low",
                "_vision_note": "Used fallback analysis due to JSON extraction failure"
            }
    
            # If question is about liquidity, add basic liquidity info
            if "liquidity" in question_lower or "lichidit" in question_lower or "liq" in question_lower:
                vision_json["liquidity_zones_description"] = "Liquidity zones appear to be present in the chart"
                vision_json["liquidity_status_suggestion"] = "unclear"
        
                # Try to infer from OCR text
                if ocr_text:
                    if "swept" in ocr_text.lower():
                        vision_json["liquidity_status_suggestion"] = "swept"
                        vision_json["_vision_note"] += ", Inferred liquidity status from OCR text"
    
            # If question is about FVGs
            elif "fvg" in question_lower or "fair value gap" in question_lower:
                vision_json["fvg_analysis"] = {
                    "count": 0,
                    "description": "Could not determine FVG details with confidence"
                }
    
            # If question is about MSS
            elif "mss" in question_lower or "structure" in question_lower:
                vision_json["mss_location_description"] = "MSS location could not be determined with confidence"
                vision_json["mss_pivot_analysis"] = {
                    "description": "Pivot structure unclear",
                    "pivot_bearish_count": None,
                    "pivot_bullish_count": None
                }
                vision_json["break_direction_suggestion"] = "unclear"
    
            # Add general fallback for trade evaluation queries
            else:
                vision_json["mss_location_description"] = "Could not identify MSS with confidence"
                vision_json["break_direction_suggestion"] = "unclear"
                vision_json["liquidity_zones_description"] = "Could not analyze liquidity zones with confidence"
                vision_json["fvg_analysis"] = {
                    "count": 0,
                    "description": "Could not identify FVGs with confidence"
                }
                vision_json["trade_outcome_suggestion"] = "unknown"
        else:
            try:
                vision_json = json.loads(vision_json_str)
                vision_json["_vision_note"] = "Vision analysis completed."
                logging.info("Vision model analysis extracted successfully.")
        
                # Cache the successful result
                vision_results_cache[cache_key] = vision_json     
            except json.JSONDecodeError:
                logging.error(f"Failed to parse vision model JSON: {vision_json_str[:500]}")
                # Try to salvage partial JSON
                try:
                    # Look for patterns that might indicate valid but incomplete JSON
                    if '{' in vision_json_str and '"analysis_possible"' in vision_json_str:
                        # Try to clean up and repair common JSON formatting issues
                        cleaned_str = re.sub(r',\s*}', '}', vision_json_str)  # Remove trailing commas
                        cleaned_str = re.sub(r',\s*]', ']', cleaned_str)     # Remove trailing commas in arrays
                
                        # Find the largest valid JSON subset
                        start_idx = vision_json_str.find('{')
                        end_idx = vision_json_str.rfind('}')
                
                        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                            partial_json_str = cleaned_str[start_idx:end_idx+1]
                            vision_json = json.loads(partial_json_str)
                            vision_json["_vision_note"] = "Partial vision analysis (recovered from malformed JSON)"
                            logging.info("Recovered partial JSON from vision model response")
                        else:
                            raise ValueError("Could not find valid JSON object boundaries")
                    else:
                        raise ValueError("No valid JSON pattern found")
                except Exception as recovery_error:
                    logging.error(f"JSON recovery attempt failed: {recovery_error}")
            
                    # Fall back to a basic structure
                    vision_json = {
                        "analysis_possible": False,
                        "confidence_level": "low",
                        "_vision_note": f"Failed to parse JSON from vision model: {str(recovery_error)}"
                    }


    # Rule Engine
    try:
        final_analysis_report = await asyncio.to_thread(apply_rule_engine, vision_json, cv_analysis)
        logging.info("Rule engine processing completed.")
    except Exception as e:
        logging.error(f"Rule engine error: {e}")
        final_analysis_report = vision_json # Fallback to vision results
        final_analysis_report["_rule_engine_notes"] = (final_analysis_report.get("_rule_engine_notes","") + f" Rule engine failed: {str(e)}").strip()


    # RAG
    context_text = ""
    try:
        search_terms = [question]
        if final_analysis_report.get("final_mss_type") not in [None, "not_identified", "unknown"]:
            search_terms.append(f"MSS {final_analysis_report['final_mss_type']}")
        if final_analysis_report.get("final_trade_direction") not in [None, "unknown"]:
            search_terms.append(f"trade {final_analysis_report['final_trade_direction']}")
        if "FVG" in question.upper() or final_analysis_report.get("has_valid_fvg") is True:
            search_terms.append("Fair Value Gap FVG")
  # Include expanded query terms for better retrieval
        if expanded:
            search_terms.append(expanded)
        search_query = " ".join(list(set(search_terms))) # Unique terms

        async with openai_call_limiter:
            embedding_response = await async_openai_client.embeddings.create(input=search_query, model=EMBEDDING_MODEL)
        query_vector = embedding_response.data[0].embedding
        pinecone_results = await asyncio.to_thread(
            index.query, vector=query_vector, top_k=3, include_metadata=True # Reduced top_k for brevity
        )
        # First collect all chunks regardless of score
        all_chunks = [
            match.metadata["text"] for match in pinecone_results.matches
            if match.metadata and "text" in match.metadata
        ]

        # Apply sophisticated filtering to prioritize relevant content
        if all_chunks:
            context_text = "\n\n".join(all_chunks)
            logging.info(f"Using all {len(all_chunks)} context chunks for query.")
        else:
            context_text = ""
        logging.info(f"Retrieved {len(context_chunks)} relevant context chunks for image query.")
    except Exception as e: # Non-critical, so don't raise HTTPException
        logging.error(f"RAG retrieval error: {e}")
        context_text = "Nu am putut prelua informații suplimentare din materialul de curs pentru această imagine."


    # Final Response Generation
    try:
        system_prompt_for_completion = _build_system_prompt(query_type, requires_full_analysis)
        messages_for_completion = [{"role": "system", "content": system_prompt_for_completion}]
        for turn in history:
            messages_for_completion.append({"role": "user", "content": turn.get("user", "")})
            if "assistant" in turn:
                messages_for_completion.append({"role": "assistant", "content": turn.get("assistant", "")})

        tech_analysis_json_str = json.dumps(final_analysis_report, indent=2, ensure_ascii=False)
# --- Start of the corrected block ---
        ocr_section_content = ""
        if ocr_text:
            ocr_section_content = f"\nFull Text Extracted from Image (OCR): {ocr_text}" # Added newline for spacing if present
        course_material_content = ""
        if context_text and "Nu am putut prelua" not in context_text:
            # The \n is correctly placed here.
            course_material_content = f"\nRelevant Course Material (for additional context only):\n{context_text}"
        # --- End of the corrected block ---
        user_prompt_for_completion = f"""
User Question: {question}

Technical Analysis Report (This is the primary source of truth for chart features):
```json
{tech_analysis_json_str}
{ocr_section_content}{course_material_content}

Based on the Technical Analysis Report, any relevant course material, and our conversation history, please answer the user's question.
Adhere to the persona and guidelines provided in the initial system prompt.
"""
        messages_for_completion.append({"role": "user", "content": user_prompt_for_completion})

        async with openai_call_limiter:
            completion_response = await async_openai_client.chat.completions.create(
                model=COMPLETION_MODEL, messages=messages_for_completion, temperature=0.6, max_tokens=1200
            )
        answer = completion_response.choices[0].message.content.strip()

        if history_store_key not in conversation_history:
            conversation_history[history_store_key] = deque(maxlen=MAX_HISTORY_MESSAGES)
        conversation_history[history_store_key].append({
            "user": question, "assistant": answer, "image_url": image_url
        })

        duration_ms = int((time.time() - start_time) * 1000)
        logging.info(f"Image-hybrid query completed in {duration_ms}ms. Session: {session_id}")
        return {
            "answer": answer, "session_id": session_id, "query_type": query_type,
            "processing_time_ms": duration_ms, "analysis_data": final_analysis_report
        }

    except RateLimitError:
        logging.warning("OpenAI rate limit hit during final completion.")
        raise HTTPException(status_code=429, detail="Prea multe solicitări către OpenAI (Completion). Te rog să încerci mai târziu.")
    except APIError as e:
        logging.error(f"OpenAI API error during final completion: {e}")
        raise HTTPException(status_code=503, detail=f"Serviciul OpenAI (Completion) nu răspunde: {e}")
    except Exception as e:
        logging.error(f"Final completion error: {e}")
        raise HTTPException(status_code=500, detail="A apărut o eroare la generarea răspunsului final.")

@app.get("/pinecone-stats")
async def pinecone_stats():
    """
    Returns statistics about the Pinecone index
    """
    try:
        # Get index stats
        index_stats = await asyncio.to_thread(index.describe_index_stats)
        
        # Return detailed information
        return {
            "status": "ok",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pinecone_index": {
                "name": PINECONE_INDEX_NAME,
                "total_vector_count": index_stats.total_vector_count
            },
            "environment": {
                "openai_model": COMPLETION_MODEL,
                "embedding_model": EMBEDDING_MODEL
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# --- Health Check (Async) ---
@app.get("/health")
async def health_check():
    """
    Health check endpoint that verifies all components are functioning.
    Checks:
    1. OpenAI API connectivity
    2. Pinecone connectivity
    3. Basic server status
    """
    start_time = time.time()
    status = {
        "status": "ok",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "components": {
            "server": {"status": "ok", "message": "FastAPI server is running."},
            "openai_api": {"status": "unknown"},
            "pinecone": {"status": "unknown"}
        }
    }
    http_status_code = 200

    # Check OpenAI API
    try:
        async with openai_call_limiter:
            # Use a minimal model call to verify API connectivity
            models = await async_openai_client.models.list(timeout=10) # Added timeout
            if models and models.data:
                status["components"]["openai_api"]["status"] = "ok"
                status["components"]["openai_api"]["models_checked"] = len(models.data)
            else:
                status["components"]["openai_api"]["status"] = "error"
                status["components"]["openai_api"]["error"] = "No models returned or empty list"
                status["status"] = "degraded"
                http_status_code = 503
    except Exception as e:
        status["components"]["openai_api"]["status"] = "error"
        status["components"]["openai_api"]["error"] = str(e)
        status["status"] = "degraded"
        http_status_code = 503

    # Check Pinecone
    try:
        # Simple stats check to verify connectivity
        index_stats = await asyncio.to_thread(index.describe_index_stats)
        if index_stats and hasattr(index_stats, 'total_vector_count'):
            status["components"]["pinecone"]["status"] = "ok"
            status["components"]["pinecone"]["index_name"] = PINECONE_INDEX_NAME
            status["components"]["pinecone"]["vector_count"] = index_stats.total_vector_count
            # Accessing namespaces more safely
            namespace_stats = index_stats.namespaces.get("", None) # Assuming default namespace might be empty string
            if namespace_stats and hasattr(namespace_stats, 'vector_count'):
                 status["components"]["pinecone"]["default_namespace_vector_count"] = namespace_stats.vector_count
            else:
                 status["components"]["pinecone"]["default_namespace_vector_count"] = "N/A or empty"
        else:
            status["components"]["pinecone"]["status"] = "error"
            status["components"]["pinecone"]["error"] = "Failed to get valid index stats"
            status["status"] = "degraded"
            http_status_code = 503
    except Exception as e:
        status["components"]["pinecone"]["status"] = "error"
        status["components"]["pinecone"]["error"] = str(e)
        status["status"] = "degraded"
        http_status_code = 503

    # Final overall status update
    if status["status"] == "degraded" and all(comp["status"] == "error" for comp_name, comp in status["components"].items() if comp_name != "server"):
        status["status"] = "error" # If all external services fail, system is in error
        http_status_code = 503
    elif status["status"] == "degraded" and http_status_code == 200: # Ensure degraded also sends 503 if not already set
        http_status_code = 503


    duration_ms = int((time.time() - start_time) * 1000)
    status["response_time_ms"] = duration_ms

    return Response(content=json.dumps(status, ensure_ascii=False),
                media_type="application/json",
                status_code=http_status_code)
