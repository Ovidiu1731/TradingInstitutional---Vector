import os
import re
import json
import logging
import time
import copy
import base64
from io import BytesIO
from typing import Dict, Any, Optional, List, Union

# Async and CV libraries
import asyncio
import httpx
import math
import aiohttp
import cv2
from utils.chunk_filtering import filter_and_rank_chunks
import numpy as np
import cachetools

import pytesseract
from PIL import Image
from dotenv import load_dotenv
from utils.query_expansion import expand_query
from collections import deque
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import AsyncOpenAI, OpenAI, RateLimitError, APIError
import pinecone
from datetime import datetime
from routers import candles
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from services.config import get_settings
from services.market_data import MarketDataService
from services.market_analysis import MarketAnalysisService

# Import the improved retrieval function
from improved_retrieval import retrieve_lesson_content

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.rate_limit_requests}/{settings.rate_limit_period}"]
)

# Initialize FastAPI app
app = FastAPI(
    title="Market Analysis API",
    description="API for analyzing market structure and candlestick patterns",
    version=settings.version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(candles.router)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

@app.get("/health")
@limiter.limit("5/minute")
async def health_check(request: Request):
    return {"status": "ok"}

@app.get("/version")
@limiter.limit("5/minute")
async def version(request: Request):
    return {"version": settings.version}

@app.get("/ready")
@limiter.limit("5/minute")
async def ready(request: Request):
    return {"ready": True}

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
TOP_K = int(os.getenv("PINECONE_TOP_K", "7"))

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
    http_client=httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(30.0, connect=10.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        # Add retry mechanism
        transport=httpx.AsyncHTTPTransport(retries=3)
    )
)

# Helper function to verify client before each request
async def ensure_valid_client():
    global async_openai_client
    
    # Check if client needs recreation
    if not async_openai_client:
        logging.warning("OpenAI client was None, recreating...")
        async_openai_client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            http_client=httpx.AsyncClient(
                http2=True,
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                transport=httpx.AsyncHTTPTransport(retries=3)
            )
        )
    return async_openai_client
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
    "pivot_type": "higher_low",
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
  "mss_pivot_analysis": { 
    "description": "The pivot is the lower high before the 'MSS' break, following the 'Local' low sweep.",
    "pivot_type": "higher_low", 
    "pivot_bearish_count": 2, 
    "pivot_bullish_count": 3 
},
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
  "mss_pivot_analysis": { 
    "description": "Pivot is the higher low marked by orange circle.",
    "pivot_type": "higher_low", 
    "pivot_bearish_count": 3, 
    "pivot_bullish_count": 3 
},
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
async def refresh_clients_periodically():
    """Periodically refresh API clients to prevent stale connections"""
    while True:
        try:
            await asyncio.sleep(3600)  # Refresh every hour
            logging.info("Performing scheduled API client refresh")
            
            global async_openai_client
            # Close the old client
            if async_openai_client:
                await async_openai_client.close()
                
            # Create a new client
            async_openai_client = AsyncOpenAI(
                api_key=OPENAI_API_KEY,
                http_client=httpx.AsyncClient(
                    http2=True, 
                    timeout=httpx.Timeout(30.0, connect=10.0),
                    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                    transport=httpx.AsyncHTTPTransport(retries=3)
                )
            )
            logging.info("API clients refreshed successfully")
            
        except Exception as e:
            logging.error(f"Error during scheduled client refresh: {e}")

@app.on_event("startup")
async def startup_event():
    global aiohttp_session
    
    # Initialize aiohttp session
    aiohttp_session = aiohttp.ClientSession()
    logging.info("aiohttp.ClientSession initialized.")
    
    # Validate OpenAI API key
    try:
        test_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        models = await test_client.models.list(timeout=5.0)
        if models and models.data:
            logging.info(f"OpenAI API key validated successfully. Available models: {len(models.data)}")
        else:
            logging.error("OpenAI API key validation failed: No models returned")
    except Exception as e:
        logging.error(f"OpenAI API key validation failed: {e}")
    
    # Initialize market analysis services
    app.state.market_data_service = MarketDataService()
    app.state.market_analysis_service = MarketAnalysisService()
    logger.info("Market analysis services initialized")
    
    # Start background task for client refresh
    asyncio.create_task(refresh_clients_periodically())
    
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

def normalize_diacritics(text: str) -> str:
    """Remove diacritics from Romanian text"""
    replacements = {
        'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ț': 't',
        'Ă': 'A', 'Â': 'A', 'Î': 'I', 'Ș': 'S', 'Ț': 'T'
    }
    for rom, eng in replacements.items():
        text = text.replace(rom, eng)
    return text

# --- Pydantic Models for Feedback and Requests ---
class FeedbackModel(BaseModel):
    session_id: str
    question: str
    answer: str
    feedback: str
    query_type: Optional[str] = "unknown"
    analysis_data: Optional[Dict] = None
    image_url: Optional[str] = None

class TextQuery(BaseModel):
    question: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    chapter: Optional[str] = None
    lesson: Optional[str] = None

class ImageHybridQuery(BaseModel):
    question: str
    image_url: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    chapter: Optional[str] = None
    lesson: Optional[str] = None

@app.get("/ping")
async def ping():
    """Simple endpoint to verify API connection without complex processing"""
    return {"status": "ok", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

# --- Feedback Logging ---
def log_feedback(session_id: str, question: str, answer: str, feedback: str,
                 query_type: str, analysis_data: Optional[Dict] = None,
                 image_url: Optional[str] = None) -> bool:
    try:
        feedback_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "session_id": session_id,
            "question": question, "answer": answer, "feedback": feedback, "query_type": query_type
        }

        # Include image URL in the feedback entry if provided
        if image_url:
            feedback_entry["image_url"] = image_url

        if analysis_data:
            relevant_fields = [
                "final_trade_direction", "final_mss_type", "final_trade_outcome",
                "is_risk_above_entry_suggestion",
                "mss_pivot_analysis",
                "fvg_analysis",
                "liquidity_status_suggestion",
                "break_direction_suggestion",
                "confidence_level",
                "setup_validity_score", # Added from rule engine
                "setup_quality_summary", # Added from rule engine
                "_cv_note", "_vision_note", "_rule_engine_notes"
            ]
            analysis_extract = {}
            for k in relevant_fields:
                if k in analysis_data:
                     analysis_extract[k] = analysis_data.get(k)

            # Pivot counts are directly in mss_pivot_analysis
            if "mss_pivot_analysis" in analysis_data and isinstance(analysis_data["mss_pivot_analysis"], dict):
                analysis_extract["pivot_bearish_count_vision"] = analysis_data["mss_pivot_analysis"].get("pivot_bearish_count")
                analysis_extract["pivot_bullish_count_vision"] = analysis_data["mss_pivot_analysis"].get("pivot_bullish_count")

            feedback_entry["analysis_data_from_report"] = analysis_extract
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        logging.error(f"Failed to log feedback: {e}"); return False

# Add this function to app.py
def retrieve_relevant_content(question: str, pinecone_results: list) -> str:
    """
    Get the most relevant content for this question with optimizations for:
    - Summary prioritization
    - Topic awareness
    - Deduplication
    - Quality filtering
    """
    
    # Extract all chunks from results
    all_chunks = [
        match.metadata["text"] for match in pinecone_results.matches
        if match.metadata and "text" in match.metadata
    ]
    
    if not all_chunks:
        return ""

    # Normalize user query to handle both with and without diacritics
    question_lower = question.lower()
    question_normalized = (question_lower
        .replace("ă", "a").replace("â", "a").replace("î", "i")
        .replace("ș", "s").replace("ț", "t"))

    # Explicit check for book-related terms in question
    book_terms = ["carte", "carti", "cărți", "carți", "recomandat", "recomanda", "citit", "lectura", "books"]
    is_book_query = any(term in question_lower or term in question_normalized for term in book_terms)

    if is_book_query:
        logging.info("Book-related query detected - special handling enabled")
        # Look specifically for book-related content
        book_chunks = []
        for chunk in all_chunks:
            chunk_lower = chunk.lower()
            # Check if this chunk mentions books or recommendations
            if any(term in chunk_lower for term in ["trading in the zone", "carte", "cărți", "recomand", "locul", "citit"]):
                book_chunks.append(chunk)

        # If we found book-related chunks, return all of them
        if book_chunks:
            logging.info(f"Found {len(book_chunks)} book-related chunks")
            return "\n\n".join(book_chunks)
    
    # Identify which chunks are from summaries vs transcripts
    summary_chunks = []
    transcript_chunks = []
    
    for match in pinecone_results.matches:
        if match.metadata and "text" in match.metadata:
            # Check if this is a summary chunk
            text = match.metadata.get("text", "")
            if (match.metadata.get("section_type") == "summary" or 
                "Rezumat" in text or
                "### " in text):
                summary_chunks.append(text)
            else:
                transcript_chunks.append(text)
    
    # Define key topics that benefit from structured responses
    key_topics = {
        "sesiuni": ["sesiune", "tranzacționare", "trading session", "londra", "new york"],
        "mss": ["market structure", "structură", "shift", "schimbare"],
        "fvg": ["fair value gap", "gap", "imbalance"],
        "setup": ["setup", "tcg", "gap setup", "og", "tg", "tcg"],
        "lichiditate": ["lichiditate", "liq", "LIQ"],
        "carti": ["cărți", "carte", "recomandate", "recomand", "citit", "books", 
          "book", "trading in the zone", "recomandari", "literatura",
          "program", "curs", "mentionat", "citesc", "lectura"]
    }
    
    # Check if the question is about a key topic
    question_lower = question.lower()
    question_normalized = normalize_diacritics(question_lower)

    matched_topic = None
    for topic, keywords in key_topics.items():
        if any(keyword in question_lower for keyword in keywords) or \
            any(keyword in question_normalized for keyword in keywords):
             matched_topic = topic
             break
    
    # Prioritize different content based on question type
    if matched_topic:
        # For key topics, focus on summary content first
        prioritized_chunks = summary_chunks + transcript_chunks
        logging.info(f"Question about {matched_topic}: Prioritizing summary chunks")
    else:
        # For general questions, use both but still put summaries first
        prioritized_chunks = summary_chunks + transcript_chunks
    
    # Deduplicate chunks
    unique_chunks = []
    seen_content = set()
    
    for chunk in prioritized_chunks:
        # Create a simplified representation for comparison (first 100 chars)
        chunk_fingerprint = chunk[:100].strip()
        
        # Only include if we haven't seen this content
        if chunk_fingerprint not in seen_content:
            unique_chunks.append(chunk)
            seen_content.add(chunk_fingerprint)
    
    # Filter out low-quality chunks
    filtered_chunks = []
    for chunk in unique_chunks:
        # Skip very short chunks
        if len(chunk.strip()) < 50:
            continue
            
        # Skip chunks that are mostly timestamps or formatting
        timestamp_count = chunk.count('[00:')
        if timestamp_count > 3 or chunk.count('\n') > chunk.count('.') * 2:
            continue
            
        filtered_chunks.append(chunk)
    
    # If we filtered too aggressively and have nothing left, use the deduplicated chunks
    if not filtered_chunks and unique_chunks:
        filtered_chunks = unique_chunks
    
    # Return the final processed chunks
    result = "\n\n".join(filtered_chunks)
    logging.info(f"Retrieved {len(all_chunks)} chunks, {len(unique_chunks)} after deduplication, {len(filtered_chunks)} after filtering")
    
    return result

@app.post("/feedback")
async def submit_feedback(feedback_data: FeedbackModel):
    success = await asyncio.to_thread(
        log_feedback,
        feedback_data.session_id,
        feedback_data.question,
        feedback_data.answer,
        feedback_data.feedback,
        feedback_data.query_type,
        feedback_data.analysis_data,
        feedback_data.image_url
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
    
    # More specific concept verification patterns
    concept_verification_patterns = ["marcate corect", "sunt corecte", "este corect", "e corect"]
    if any(p in question_lower for p in concept_verification_patterns):
        logging.info("Query identified as 'concept_verification_image_query'.")
        return {"type": "concept_verification_image_query"}
    
    # Trade evaluation patterns as a fallback
    trade_evaluation_patterns = ["cum arata", "cum arată", "ce parere", "ce părere", "evalueaz", "analizeaz", 
                                "trade", "setup", "intrare", "valid", "rezultat"]
    if any(p in question_lower for p in trade_evaluation_patterns) or "?" not in question_lower:
        logging.info("Query identified as 'trade_evaluation_image_query'.")
        return {"type": "trade_evaluation_image_query"}
        
    logging.info("Query identified as 'general_image_query'.")
    return {"type": "general_image_query"}

# --- Image/JSON Helpers ---
async def download_image_async(image_url: str) -> Optional[bytes]:
    global aiohttp_session # Move this line to the top of the function
    if not aiohttp_session:
        logging.error("aiohttp_session not initialized for image download attempt.")
        aiohttp_session = aiohttp.ClientSession() # Re-init if somehow missed
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
        
        # Try different preprocessing techniques to improve OCR results
        # Convert to grayscale
        img_gray = img.convert('L')
        
        # First attempt with original image
        text = pytesseract.image_to_string(img, lang="eng")
        
        # If text is too short, try with grayscale and additional processing
        if len(text.strip()) < 10:
            # Apply thresholding to improve contrast
            threshold = 150
            img_bw = img_gray.point(lambda x: 0 if x < threshold else 255, '1')
            text = pytesseract.image_to_string(img_bw, lang="eng", config='--psm 6')
        
        # Clean the extracted text
        cleaned_text = "".join(ch for ch in text if ord(ch) < 128).strip()
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        logging.info(f"OCR: Extracted text length: {len(cleaned_text)}")
        return cleaned_text
    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract not found. Please ensure it is installed and in your PATH. OCR will not function.")
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

def analyze_mss_and_displacement(vision_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the relationship between MSS and displacement to determine trade direction.
    """
    # Extract MSS location and text descriptions
    mss_location = vision_json.get("mss_location_description", "")
    
    # Find MSS label coordinates if possible
    mss_coords = None
    for label in vision_json.get("visible_labels_on_chart", []):
        if "MSS" in label.upper():
            # If we have label coordinates, store them
            if isinstance(label, dict) and "position" in label:
                mss_coords = label["position"]
                break
    
    # Analyze price path AFTER MSS by looking at price movement descriptions
    displacement_info = vision_json.get("displacement_analysis", {})
    
    # Determine if price goes UP or DOWN after MSS
    # This is critical - we need to detect the actual path regardless of "break direction"
    
    # Check for explicit direction in displacement description
    displacement_description = displacement_info.get("description", "").lower()
    
    # Look for clear direction indicators in description
    upward_indicators = ["upward", "higher", "increases", "rises", "bullish", "moving up"]
    downward_indicators = ["downward", "lower", "decreases", "falls", "bearish", "moving down"]
    
    displacement_direction = "unknown"
    
    # Check for upward movement indicators
    if any(indicator in displacement_description for indicator in upward_indicators):
        displacement_direction = "bullish"
    # Check for downward movement indicators
    elif any(indicator in displacement_description for indicator in downward_indicators):
        displacement_direction = "bearish"
    # Fallback to the explicit direction field if available
    elif displacement_info.get("direction") in ["bullish", "bearish"]:
        displacement_direction = displacement_info.get("direction")
    
    # Map to trade direction
    if displacement_direction == "bullish":
        direction = "long"
    elif displacement_direction == "bearish":
        direction = "short"
    else:
        direction = "unknown"
    
    return {
        "mss_location": mss_location,
        "displacement_direction": displacement_direction,
        "trade_direction_from_displacement": direction,
        "confidence": "high" if displacement_direction in ["bullish", "bearish"] else "low"
    }

def analyze_trade_zones(vision_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the trade zones (boxes/rectangles) to determine trade direction.
    This doesn't rely on specific colors but on the structural relationship
    between risk and profit zones.
    """
    # Look for zone descriptions in the vision analysis
    fvg_description = vision_json.get("fvg_analysis", {}).get("description", "").lower()
    mss_description = vision_json.get("mss_location_description", "").lower()
    
    # Keywords that indicate zone relationships
    profit_keywords = ["profit", "target", "tp", "take profit", "reward"]
    risk_keywords = ["risk", "stop", "sl", "stop loss"]
    
    # Zone position indicators
    zone_above = any(word in fvg_description + " " + mss_description for word in [
        "above", "higher", "upper", "top"
    ])
    zone_below = any(word in fvg_description + " " + mss_description for word in [
        "below", "lower", "bottom", "underneath" 
    ])
    
    # Relationship between zones
    risk_above_profit = vision_json.get("is_risk_above_entry_suggestion", None)
    
    # Attempt to determine direction based on zone relationship
    direction = "unknown"
    confidence = "low"
    
    # If we know risk is above entry, it's likely a short trade
    if risk_above_profit is True:
        direction = "short"
        confidence = "medium"
    # If we know risk is below entry, it's likely a long trade  
    elif risk_above_profit is False:
        direction = "long"
        confidence = "medium"
    # Otherwise try to infer from zone descriptions
    elif zone_above and any(k in fvg_description for k in profit_keywords):
        direction = "long"  # Profit zone above = long
        confidence = "low"
    elif zone_below and any(k in fvg_description for k in profit_keywords):
        direction = "short"  # Profit zone below = short
        confidence = "low"
    
    return {
        "trade_direction_from_zones": direction,
        "zone_confidence": confidence,
        "risk_above_entry": risk_above_profit
    }

def analyze_price_path(vision_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the actual price movement pattern visible in the chart to determine direction.
    This uses the candle patterns after MSS to determine if price is moving up or down.
    """
    # We need to analyze if price is moving up or down after MSS
    # Look for candle pattern descriptions
    candle_description = vision_json.get("candle_colors", "")
    
    # Extract all text fields for analysis
    all_text = " ".join([
        str(vision_json.get("mss_location_description", "")),
        str(vision_json.get("displacement_analysis", {}).get("description", "")),
        str(vision_json.get("fvg_analysis", {}).get("description", ""))
    ]).lower()
    
    # Look for descriptions of price movement
    price_increases = any(phrase in all_text for phrase in [
        "price increases", "price rises", "moving upward", "moving higher",
        "higher than before", "price goes up", "bullish movement"
    ])
    
    price_decreases = any(phrase in all_text for phrase in [
        "price decreases", "price falls", "moving downward", "moving lower",
        "lower than before", "price goes down", "bearish movement"
    ])
    
    # If movement description is inconclusive, check for price levels
    if not price_increases and not price_decreases:
        # Look for numeric patterns like "1.3245 to 1.3260" or "41,050 → 41,030"
        price_pattern = r'(\d+[.,]\d+).*?(\d+[.,]\d+)'
        matches = re.findall(price_pattern, all_text)
        
        if matches:
            for match in matches:
                try:
                    price1 = float(match[0].replace(',', ''))
                    price2 = float(match[1].replace(',', ''))
                    
                    if price2 > price1:
                        price_increases = True
                    elif price2 < price1:
                        price_decreases = True
                except:
                    continue
    
    # Determine direction based on price movement
    if price_increases:
        return {"price_path_direction": "long", "confidence": "medium"}
    elif price_decreases:
        return {"price_path_direction": "short", "confidence": "medium"}
    else:
        return {"price_path_direction": "unknown", "confidence": "low"}

def analyze_text_labels(vision_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze text labels on the chart for directional clues.
    """
    # Get all visible labels
    labels = vision_json.get("visible_labels_on_chart", [])
    
    # Look for direction-specific labels
    long_indicators = ["LONG", "BUY", "BULLISH"]
    short_indicators = ["SHORT", "SELL", "BEARISH"]
    
    # Check if BE (Break Even) or TP (Take Profit) positions indicate direction
    be_position = None
    tp_position = None
    mss_position = None
    
    # Extract positions if available, otherwise just note presence
    for label in labels:
        label_text = label if isinstance(label, str) else label.get("text", "")
        label_text = label_text.upper()
        
        if "BE" in label_text:
            be_position = True
        elif "TP" in label_text or "TARGET" in label_text:
            tp_position = True
        elif "MSS" in label_text:
            mss_position = True
    
    # Check OCR text for additional directional clues
    ocr_text = vision_json.get("extracted_text", "").upper()
    
    direction = "unknown"
    confidence = "low"
    
    # Look for explicit direction indicators in labels
    if any(indicator in " ".join(str(label).upper() for label in labels) for indicator in long_indicators):
        direction = "long"
        confidence = "high"
    elif any(indicator in " ".join(str(label).upper() for label in labels) for indicator in short_indicators):
        direction = "short"
        confidence = "high"
    # Look in OCR text if we have it
    elif any(indicator in ocr_text for indicator in long_indicators):
        direction = "long"
        confidence = "medium"
    elif any(indicator in ocr_text for indicator in short_indicators):
        direction = "short"
        confidence = "medium"
    
    return {
        "label_suggested_direction": direction,
        "label_confidence": confidence,
        "has_be_label": be_position is not None,
        "has_tp_label": tp_position is not None,
        "has_mss_label": mss_position is not None
    }

def determine_trade_direction_from_pivot(vision_json: Dict[str, Any]) -> str:
    """Determine trade direction based on the type of pivot being broken"""
    
    # Extract pivot description
    mss_pivot_data = vision_json.get("mss_pivot_analysis", {})
    pivot_description = mss_pivot_data.get("description", "").lower()
    
    # Check for pivot type keywords
    is_lower_high = any(term in pivot_description for term in ["lower high", "lower peak", "decreasing peak", "descending peak"])
    is_higher_low = any(term in pivot_description for term in ["higher low", "higher trough", "increasing trough", "ascending trough"])
    
    # Apply the market structure rule
    if is_lower_high:
        return "long"  # Breaking a lower high = LONG trade
    elif is_higher_low:
        return "short"  # Breaking a higher low = SHORT trade
    else:
        return "unknown"  # Pivot type not clearly identified


def determine_final_trade_direction(vision_json: Dict[str, Any], cv_analysis: Dict[str, Any]) -> str:
    """
    Integrate all analysis methods to make a final trade direction decision.
    Prioritizes displacement+MSS alignment but uses multiple evidence sources.
    """
    # Gather evidence from all analysis methods
    mss_displacement = analyze_mss_and_displacement(vision_json)
    zone_analysis = analyze_trade_zones(vision_json)
    price_path = analyze_price_path(vision_json)
    label_analysis = analyze_text_labels(vision_json)
    
    # Create a weighted voting system
    direction_votes = {
        "long": 0,
        "short": 0,
        "unknown": 0
    }
    
    # MSS and displacement get highest weight
    if mss_displacement["trade_direction_from_displacement"] != "unknown":
        direction_votes[mss_displacement["trade_direction_from_displacement"]] += 3
    
    # Zone analysis gets good weight
    if zone_analysis["trade_direction_from_zones"] != "unknown":
        direction_votes[zone_analysis["trade_direction_from_zones"]] += 2
    
    # Price path gets medium weight
    if price_path["price_path_direction"] != "unknown":
        direction_votes[price_path["price_path_direction"]] += 2
    
    # Label analysis gets medium weight if high confidence
    if label_analysis["label_suggested_direction"] != "unknown":
        weight = 3 if label_analysis["label_confidence"] == "high" else 1
        direction_votes[label_analysis["label_suggested_direction"]] += weight
    
    # Determine direction based on voting
    if direction_votes["long"] > direction_votes["short"]:
        return "long"
    elif direction_votes["short"] > direction_votes["long"]:
        return "short"
    else:
        # If tied, prioritize MSS+displacement
        return mss_displacement["trade_direction_from_displacement"]

def validate_trade_direction(vision_data: Dict[str, Any]) -> tuple:
    """Validate trade direction using multiple data points"""
    
    # Get key data points
    break_direction = vision_data.get("break_direction_suggestion", "unknown")
    displacement_direction = vision_data.get("displacement_analysis", {}).get("direction", "unknown")
    mss_description = vision_data.get("mss_location_description", "").lower()
    
    # Map break direction to expected trade direction
    if break_direction == "upward":
        mss_direction = "bullish"
    elif break_direction == "downward":
        mss_direction = "bearish"
    else:
        mss_direction = "unknown"

    # Check if displacement matches expected direction
    direction_aligned = False
    if mss_direction == "bullish" and displacement_direction == "bullish":
        direction_aligned = True
        trade_direction = "long"
    elif mss_direction == "bearish" and displacement_direction == "bearish":
        direction_aligned = True
        trade_direction = "short"

     # Return immediately if we have high confidence alignment
    if direction_aligned:
        return trade_direction, 3  # High confidence

    # Otherwise, look for additional clues in the descriptions
    signals = []
    
    # Look for direct mentions of trade direction in descriptions
    if "long" in mss_description or "buy" in mss_description:
        signals.append("long")
    elif "short" in mss_description or "sell" in mss_description:
        signals.append("short")
    
    # Look for aligned movement descriptors
    if (break_direction == "upward" and 
        any(phrase in mss_description for phrase in ["bullish", "upward", "higher", "increased"])):
        signals.append("long")
    elif (break_direction == "downward" and 
          any(phrase in mss_description for phrase in ["bearish", "downward", "lower", "decreased"])):
        signals.append("short")
    
    # Count signal frequencies
    short_count = signals.count("short")
    long_count = signals.count("long")
    
    if short_count > long_count:
        return "short", min(short_count, 2)  # Cap confidence at 2 (medium)
    elif long_count > short_count:
        return "long", min(long_count, 2)  # Cap confidence at 2 (medium)
    else:
        return "unknown", 0  # Equal signals or no signals      

def determine_setup_type(fvg_count: int, fvg_description: str = "", is_second_leg: bool = False) -> str:
    """
    Determine the trading setup type based on program definitions.
    
    Args:
        fvg_count: Number of FVGs (Fair Value Gaps)
        fvg_description: Description of FVGs from vision model (helps detect consecutive gaps)
        is_second_leg: Whether this is a second leg setup
        
    Returns:
        String with the setup type according to Trading Institutional terminology
    """
    # Check for second leg setup first (special case)
    if is_second_leg:
        return "SLG"  # Second Leg Gap Setup
        
    # Classify based on FVG count
    if fvg_count == 1:
        return "OG"   # One Gap Setup
    elif fvg_count == 2:
        # Check if consecutive based on description
        if "consecutive" in fvg_description.lower() or "consecutive" in fvg_description.lower():
            return "TCG"  # Two Consecutive Gap Setup
        else:
            return "TG"   # Two Gap Setup
    elif fvg_count == 3:
        # Check if consecutive based on description
        if "consecutive" in fvg_description.lower() or "consecutive" in fvg_description.lower():
            return "3CG"  # Three Consecutive Gap Setup
        else:
            return "3G"   # Three Gap Setup
    elif fvg_count > 3:
        return "MG"   # Multiple Gap Setup
    
    # Default if can't determine
    return "unknown"

def apply_rule_engine(vision_json: Dict[str, Any], cv_findings: Dict[str, Any], query_type: str = "trade_evaluation_image_query") -> Dict[str, Any]:
    """
    Apply rule engine with awareness of query type
    to make deterministic trading decisions based on the rules.

    Returns an enriched analysis with rule-based decisions.
    """
    final_analysis = copy.deepcopy(vision_json)  # Start with vision model output
    final_analysis["_rule_engine_notes"] = f"Rule engine applied for {query_type}"
    
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

    # --- INTEGRATED DIRECTION DETERMINATION ---
    # FIRST: Try to determine direction based on pivot type (highest priority)
    pivot_based_direction = determine_trade_direction_from_pivot(vision_json)
    final_analysis["pivot_based_direction"] = pivot_based_direction

    # Collect additional evidence from all subsystems
    mss_displacement = analyze_mss_and_displacement(vision_json)
    zone_analysis = analyze_trade_zones(vision_json)
    price_path = analyze_price_path(vision_json)
    label_analysis = analyze_text_labels(vision_json)

    # Record all the evidence for transparency
    final_analysis["direction_evidence"] = {
        "pivot_based_direction": pivot_based_direction,
        "mss_displacement_analysis": mss_displacement,
        "zone_analysis": zone_analysis,
        "price_path_analysis": price_path,
        "label_analysis": label_analysis
    }

    # PRIMARY RULE: Use pivot-based direction if available
    if pivot_based_direction != "unknown":
        final_analysis["final_trade_direction"] = pivot_based_direction
        final_analysis["_rule_engine_notes"] += f", Direction determined from pivot type: {pivot_based_direction}"
    else:
        # Fall back to the integrated approach only if pivot type is unclear
        final_analysis["final_trade_direction"] = determine_final_trade_direction(vision_json, cv_findings)
        final_analysis["_rule_engine_notes"] += ", Direction determined from integrated analysis (pivot type unclear)"
    
    # Determine confidence based on agreement level
    evidence_directions = [
        mss_displacement["trade_direction_from_displacement"],
        zone_analysis["trade_direction_from_zones"],
        price_path["price_path_direction"],
        label_analysis["label_suggested_direction"]
    ]

    # Filter out unknown directions
    known_directions = [d for d in evidence_directions if d != "unknown"]

     # Check if all known directions agree
    if known_directions and all(d == known_directions[0] for d in known_directions):
        final_analysis["direction_confidence"] = "high"
    # If we have limited but consistent evidence
    elif known_directions and len(set(known_directions)) == 1:
        final_analysis["direction_confidence"] = "medium"
    # If we have conflicting evidence
    else:
        final_analysis["direction_confidence"] = "low"
        final_analysis["_rule_engine_notes"] += ", Note: Conflicting direction evidence, using weighted decision"

    # Record if we have alignment between MSS break and displacement
    break_direction = final_analysis.get("break_direction_suggestion", "unknown")
    displacement_direction = mss_displacement.get("displacement_direction", "unknown")

    # First determine MSS direction from break direction
    if break_direction == "upward":
        mss_direction = "bullish"
    elif break_direction == "downward":
        mss_direction = "bearish"
    else:
        mss_direction = "unclear"
    final_analysis["mss_direction"] = mss_direction

    # Then check for alignment between MSS and displacement
    if break_direction != "unknown" and displacement_direction != "unknown":
        if (break_direction == "upward" and displacement_direction == "bullish") or \
            (break_direction == "downward" and displacement_direction == "bearish"):
            final_analysis["direction_alignment"] = True
        else:
            final_analysis["direction_alignment"] = False
            final_analysis["_rule_engine_notes"] += ", Warning: MSS and displacement directions don't align"
        
    # Enrich the analysis with direction alignment information
    if final_analysis.get("direction_alignment") is True:
        # If directions align, increase confidence
        final_analysis["direction_confidence"] = "high"
        final_analysis["_rule_engine_notes"] += f", Direction confirmed by MSS and displacement alignment"
    else:
        # When not aligned, record the conflict but keep the weighted decision
        final_analysis["_rule_engine_notes"] += ", Warning: MSS and displacement directions don't align"
        
        # Record what each analysis suggests for diagnostics
        if displacement_direction in ["bullish", "bearish"]:
            final_analysis["displacement_suggests"] = "long" if displacement_direction == "bullish" else "short"
        if mss_direction in ["bullish", "bearish"]:
            final_analysis["mss_suggests"] = "long" if mss_direction == "bullish" else "short"
        
        # Lower confidence since we have conflicting signals
        if final_analysis.get("direction_confidence") != "none":
            final_analysis["direction_confidence"] = "low"

    # Add validation check as a safety net
    if final_analysis["final_trade_direction"] == "unknown":
        validated_direction, confidence_score = validate_trade_direction(final_analysis)
        if validated_direction != "unknown" and confidence_score >= 2:
            final_analysis["final_trade_direction"] = validated_direction
            final_analysis["direction_confidence"] = "low"  # Low confidence because we had to rely on validation
            final_analysis["_rule_engine_notes"] += f", Direction determined from validation: {validated_direction}"

    # --- Process CV findings ---
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

    # --- For concept verification, apply only the relevant rules and return early ---
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

    # --- Continue with full analysis for trade evaluation ---
    
    # --- Determine trade outcome ---
    # Use vision model's trade outcome suggestion as primary input
    outcome_suggestion = final_analysis.get("trade_outcome_suggestion", "unknown")

    # Check if we can determine outcome based on price movement relative to risk levels
    risk_detected = False
    trade_direction = final_analysis.get("final_trade_direction", "unknown")

    # If we have risk box info from CV, we can try to determine if price has broken through stop loss levels
    risk_box_info = final_analysis.get("risk_box_info_cv", {})
    if risk_box_info and "coordinates" in risk_box_info and trade_direction != "unknown":
        # For a SHORT trade, if there's strong bullish displacement AFTER the trade entry, that suggests a loss
        if trade_direction == "short" and displacement_direction == "bullish" and displacement.get("strength") in ["moderate", "strong"]:
            risk_detected = True
            final_analysis["_rule_engine_notes"] += ", Loss detected: Bullish price movement against short position"
    
        # For a LONG trade, if there's strong bearish displacement AFTER the trade entry, that suggests a loss
        elif trade_direction == "long" and displacement_direction == "bearish" and displacement.get("strength") in ["moderate", "strong"]:
            risk_detected = True
            final_analysis["_rule_engine_notes"] += ", Loss detected: Bearish price movement against long position"

    # Now set the final outcome, giving priority to our risk detection if it's conclusive
    if risk_detected:
        final_analysis["final_trade_outcome"] = "loss"
    elif outcome_suggestion in ["win", "breakeven", "loss", "potential_setup"]:
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
    # This information is already captured during direction determination, but setting it explicitly for compatibility
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

    # --- Determine setup type based on FVG analysis ---
    fvg_description = fvg_analysis.get("description", "")
    # Try to detect second leg setup from description or vision model analysis
    is_second_leg = False
    if "second leg" in fvg_description.lower() or "al doilea picior" in fvg_description.lower():
        is_second_leg = True
        
    # Determine the setup type
    setup_type = determine_setup_type(fvg_count, fvg_description, is_second_leg)
    final_analysis["setup_type"] = setup_type
    final_analysis["_rule_engine_notes"] += f", Setup classified as {setup_type}"

    # --- Calculate final validity score ---
    validity_score = 50 # Base score

    # Add points for important criteria
    if final_analysis.get("final_mss_type") != "not_identified": validity_score += 10
    if final_analysis.get("final_trade_direction") != "unknown": validity_score += 10
    if final_analysis.get("liquidity_status_suggestion") == "swept": validity_score += 10
    if final_analysis.get("has_valid_fvg") is True: validity_score += 10 # Check for explicit True
    if final_analysis.get("displacement_matches_trade") is True: validity_score += 10 # Check for explicit True

    # CRITICAL: Add points ONLY if direction alignment is confirmed
    if final_analysis.get("direction_alignment") is True: validity_score += 20 # Higher weight for alignment

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
            "Utilizatorul a trimis un grafic pentru analiza unei tranzacții. "
            "Analizeză setup-ul de tranzacționare prezentat în grafic conform regulilor Trading Instituțional. "
            "Bazează-ți analiza în principal pe Raportul de Analiză Tehnică, care conține rezultatele analizei CV, "
            "ale modelului de viziune și determinări ale motorului de reguli."
        )
        full_prompt += "\n\n" + (
            "REGULI IMPORTANTE DE APLICAT ÎN CONVERSAȚIE:\n"
            "1. Răspunde într-un stil conversațional, natural, ca un trader experimentat care explică unui coleg.\n"
            "2. Evită listele cu numere/bullets în favoarea unor paragrafe scurte și naturale.\n"
            "3. Asigură-te că discuți despre direcția REALĂ a prețului, nu doar despre direcția ruperii de structură.\n"
            "4. Menționează explicit dacă tranzacția este LONG (cumpărare) sau SHORT (vânzare).\n"
            "5. Nu face niciodată presupuneri despre continuarea mișcării prețului dacă nu sunt evidente în grafic.\n"
            "6. Folosește un ton încrezător dar nu prea formal - ca un coleg care explică cu respect.\n"
            "7. Fii natural și direct, dar totuși profesionist.\n"
            "8. NU menționa niciodată scoruri numerice de validitate (de exemplu, 'scor de 100') - exprimă calitatea setup-ului în termeni naturali.\n"
            "9. NU folosi fraze tehnice precum 'nivel de încredere' sau 'indicând că toate condițiile au fost îndeplinite'.\n"
            "10. NU enumera criteriile tehnice îndeplinite, ci discută despre setup în mod natural.\n"
            "11. Discută despre setup ca și cum ai vorbi cu un coleg trader, nu ca un raport tehnic automat."
            "12. Verifică întotdeauna câmpul 'setup_type' din raport pentru a identifica corect tipul de setup:\n"
            "   - 'OG' = One Gap Setup (un singur gap FVG)\n"
            "   - 'TG' = Two Gap Setup (2 gap-uri FVG)\n"
            "   - 'TCG' = Two Consecutive Gap Setup (2 gap-uri consecutive)\n"
            "   - '3G' = Three Gap Setup (3 gap-uri)\n"
            "   - '3CG' = Three Consecutive Gap Setup (3 gap-uri consecutive)\n"
            "   - 'SLG' = Second Leg Setup (al doilea picior)\n"
            "   - 'MG' = Multiple Gap Setup (mai mult de 3 gap-uri)\n"
            "   - Menționează întotdeauna tipul corect de setup folosind terminologia din cadrul programului.\n"
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

    # Trade evaluation specific prompt
    if query_type == "trade_evaluation_image_query" or query_type == "general_image_query":
        vision_system_prompt_template = base_prompt + """
Provide a comprehensive analysis of the trading chart, focusing on these key aspects:
- Market Structure Shifts (MSS) - both Normal and Aggressive types
- Candle and color patterns
- Direction of the break (upward/downward)
- CRITICAL: Displacement after MSS - carefully describe the ACTUAL price movement AFTER the MSS

EXTREMELY IMPORTANT FOR DISPLACEMENT ANALYSIS:
1. Look at where the MSS is marked on the chart (usually with "MSS" label or arrows)
2. CAREFULLY observe what happens to price AFTER this MSS point:
   - If price MOVES HIGHER after MSS = "bullish" displacement
   - If price MOVES LOWER after MSS = "bearish" displacement
3. Don't just look at the break direction - analyze what ACTUALLY HAPPENS after the break
4. In your displacement_analysis, include a detailed description of the actual price path

Definitions:
- MSS (Market Structure Shift): A break of market structure that signals a potential trend change.
- Normal MSS: The pivot (higher low or lower high) that is broken must have at least 2 bearish AND 2 bullish candles.
- Aggressive MSS: The pivot has fewer than 2 bearish OR fewer than 2 bullish candles.
- FVG (Fair Value Gap): An unfilled gap created when price makes an impulsive move.
- Liquidity: Price levels where stop losses or take profits are clustered.

CRITICAL DIRECTION ANALYSIS INSTRUCTIONS:
1. MOST IMPORTANT: Identify the TYPE of pivot structure being broken:
   - Breaking a LOWER HIGH = LONG trade (regardless of break direction)
   - Breaking a HIGHER LOW = SHORT trade (regardless of break direction)
2. A LOWER HIGH is a peak that's lower than the previous peak
3. A HIGHER LOW is a trough/valley that's higher than the previous trough/valley
4. In your pivot analysis, clearly state whether the pivot is a "lower high" or "higher low"

AFTER IDENTIFYING PIVOT TYPE, ALSO ANALYZE:
1. Direction of the break (upward/downward)
2. Direction of displacement (bullish/bearish)
3. Alignment between break and displacement directions
4. The best setups have all factors aligned:
   - Breaking a lower high downward with bullish displacement = strong LONG
   - Breaking a higher low upward with bearish displacement = strong SHORT

PRICE MOVEMENT AFTER MSS:
1. After breaking a LOWER HIGH, price should move HIGHER (bullish displacement)
2. After breaking a HIGHER LOW, price should move LOWER (bearish displacement)
3. Pay special attention to displacement strength and direction
""" + base_json_structure + """
Your JSON response MUST include the following full structure:
{
  "analysis_possible": true/false,
  "candle_colors": "description of bullish/bearish candle colors or 'unknown'",
  "is_risk_above_entry_suggestion": true/false/null,
  "mss_location_description": "description of where the MSS is located",
  "mss_pivot_analysis": {
    "description": "description of the pivot structure (clearly state if it's a 'lower high' or 'higher low')",
    "pivot_type": "lower_high"/"higher_low"/"unknown",
    "pivot_bearish_count": number,
    "pivot_bullish_count": number
  },
  "break_direction_suggestion": "upward"/"downward"/"unclear",
  "displacement_analysis": { 
    "direction": "bullish"/"bearish"/"unclear", 
    "strength": "weak"/"moderate"/"strong",
    "description": "DETAILED description of how price actually moves after the MSS point, including whether it goes higher or lower"
  },
  "fvg_analysis": { "count": number, "description": "description of FVGs present" },
  "liquidity_zones_description": "description of liquidity zones and if they were swept",
  "liquidity_status_suggestion": "swept"/"not_swept"/"unclear",
  "trade_outcome_suggestion": "win"/"loss"/"breakeven"/"potential_setup"/"unknown",
  "visible_labels_on_chart": ["MSS", "BE", etc.],
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
    "description": "description of the pivot structure (clearly state if it's a 'lower high' or 'higher low')",
    "pivot_type": "lower_high"/"higher_low"/"unknown",
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
- Displacement after MSS - CRITICAL for determining trade direction (bearish displacement = SHORT, bullish displacement = LONG)
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

IMPORTANT FOR DIRECTION ANALYSIS:
1. Carefully analyze the ACTUAL price movement after the MSS, not just the break direction
2. "Bullish displacement" means price is moving UP after MSS (creating bullish FVGs) = LONG trade
3. "Bearish displacement" means price is moving DOWN after MSS (creating bearish FVGs) = SHORT trade
4. Count candles in pivots very carefully - each actual candle body (not wicks) counts
5. Don't assume direction based on the MSS break alone - confirm with the subsequent movement

PRICE LEVEL ANALYSIS FOR DIRECTION:
1. Compare price LEVELS before and after the MSS:
   - If price is LOWER after MSS than before = BEARISH = SHORT
   - If price is HIGHER after MSS than before = BULLISH = LONG
2. Look at the NUMBER values on the price axis:
   - If numbers are DECREASING after MSS = BEARISH = SHORT (e.g. 41,050 → 41,030 → 41,010)
   - If numbers are INCREASING after MSS = BULLISH = LONG (e.g. 41,010 → 41,030 → 41,050)
3. The actual trade direction is determined by where price GOES after the MSS, not by what the MSS breaks.

IMPORTANT FOR TRADE OUTCOME ANALYSIS:
1. Look carefully for evidence of trade outcome (win, loss, breakeven)
2. A trade is a LOSS if price has clearly moved beyond the stop loss level
3. Check for labels like "SL", "TP", or visually inspect if price broke significant levels
4. For active trades, check if price is still within acceptable ranges
5. If you see price has moved against the trade beyond risk limits, classify as "loss"

IMPORTANT FOR FVG ANALYSIS:
1. Look carefully for ALL Fair Value Gaps in the chart - there may be multiple FVGs
2. Count each distinct gap between candles where price hasn't traded yet
3. Pay special attention to areas marked with boxes or rectangles, which often indicate FVGs
4. Check both above and below the current price for potential FVGs
""" + base_json_structure + """
Your JSON response MUST include the following full structure:
{
  "analysis_possible": true/false,
  "candle_colors": "description of bullish/bearish candle colors or 'unknown'",
  "is_risk_above_entry_suggestion": true/false/null,
  "mss_location_description": "description of where the MSS is located",
  "mss_pivot_analysis": {
    "description": "description of the pivot structure (clearly state if it's a 'lower high' or 'higher low')",
    "pivot_type": "lower_high"/"higher_low"/"unknown",
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
@app.post("/ask", response_model=Dict[str, Any])
async def ask_question(query: TextQuery):
    """
    Handles user text questions using improved semantic retrieval.
    """
    question = query.question
    chapter = query.chapter if hasattr(query, 'chapter') else None
    lesson = query.lesson if hasattr(query, 'lesson') else None
    try:
        # Use improved retrieval logic
        results = await asyncio.to_thread(
            retrieve_lesson_content,
            question,
            chapter,
            lesson,
            TOP_K
        )
        if results:
            # Combine top results for context
            context_text = "\n\n".join([r["text"] for r in results])
            logging.info(f"Retrieved {len(results)} relevant context chunks for text query.")
        else:
            context_text = ""
            logging.info("No relevant context found for text query.")
    except Exception as e:
        logging.error(f"Improved retrieval error: {e}")
        context_text = "Am întâmpinat o problemă la accesarea materialului de curs din baza de date Pinecone."
    # ... rest of the endpoint logic ...
    # Return context_text as part of the response as before
    return {"context": context_text}

@app.post("/ask-image-hybrid", response_model=Dict[str, Any])
async def ask_image_hybrid(payload: ImageHybridQuery) -> Dict[str, Any]:
    """
    Handles hybrid image/text questions using improved semantic retrieval for the text part.
    """
    question = payload.question
    chapter = getattr(payload, 'chapter', None)
    lesson = getattr(payload, 'lesson', None)
    try:
        # Use improved retrieval logic for text context
        results = await asyncio.to_thread(
            retrieve_lesson_content,
            question,
            chapter,
            lesson,
            TOP_K
        )
        if results:
            context_text = "\n\n".join([r["text"] for r in results])
            logging.info(f"Retrieved {len(results)} relevant context chunks for image hybrid query.")
        else:
            context_text = ""
            logging.info("No relevant context found for image hybrid query.")
    except Exception as e:
        logging.error(f"Improved retrieval error (image hybrid): {e}")
        context_text = "Nu am putut prelua informații suplimentare din baza de date vectorială."
    # ... rest of the endpoint logic ...
    # Return context_text as part of the response as before
    return {"context": context_text}

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