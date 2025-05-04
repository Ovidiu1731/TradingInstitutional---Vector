# code
import os
import re
import json
import logging
import time
from io import BytesIO
from typing import Dict, Any, Optional, List, Union

import requests
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI, RateLimitError, APIError
from pinecone import Pinecone, PineconeException

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# ENVIRONMENT & GLOBALS
# ---------------------------------------------------------------------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trading-lessons")
FEEDBACK_LOG = os.getenv("FEEDBACK_LOG", "feedback_log.jsonl")  # New env variable for feedback log

# Model selection - easier to update as new models are released
EMBEDDING_MODEL = "text-embedding-ada-002"  # Update when new models are available
VISION_MODEL = "gpt-4.1"  # Update to newer models like gpt-4.5/5 when available
COMPLETION_MODEL = "gpt-4.1"  # Update when new models are available
TEXT_MODEL = "gpt-3.5-turbo"  # Could be updated to newer model

if not (OPENAI_API_KEY and PINECONE_API_KEY):
    logging.error("Missing OpenAI or Pinecone API key(s) in environment variables.")
    raise ValueError("Missing OpenAI or Pinecone API key(s)")

# Load core system prompt
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT_CORE = f.read().strip()
except FileNotFoundError:
    logging.warning("system_prompt.txt not found. Using fallback system prompt.")
    SYSTEM_PROMPT_CORE = (
        "You are an AI assistant trained by Rareș for the Trading Instituțional community. "
        "Answer questions strictly based on the provided course material and visual analysis (if available). "
        "Emulate Rareș's direct, concise teaching style. Be helpful and accurate according to the course rules."
    )

# Define the core structural definitions
MSS_AGRESIV_STRUCTURAL_DEFINITION = "Definiție Structurală MSS Agresiv: Este o rupere de structură formată dintr-o singură lumânare care face low/high."
MSS_NORMAL_STRUCTURAL_DEFINITION = "Definiție Structurală MSS Normal: Este o rupere de structură formată din două sau mai multe lumânări care fac low/high."
FVG_STRUCTURAL_DEFINITION = "Definiție Structurală FVG (Fair Value Gap): Este un gap (spațiu gol) între lumânări creat în momentul în care prețul face o mișcare impulsivă, lăsând o zonă netranzacționată."
DISPLACEMENT_DEFINITION = "Definiție Displacement: Este o mișcare continuă a prețului în aceeași direcție, după o structură invalidată, creând FVG-uri (Fair Value Gaps)."

# SDK clients
try:
    openai = OpenAI(api_key=OPENAI_API_KEY)
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY) # Renamed to avoid conflict
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

app = FastAPI(title="Trading Instituțional AI Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Consider restricting in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# FEEDBACK MECHANISM
# ---------------------------------------------------------------------------
# New functionality for capturing user feedback

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
        
        # Include selective analysis data if available
        if analysis_data:
            # Only include relevant fields to keep log size manageable
            relevant_fields = [
                "trade_direction", "mss_type", "breaking_candle_count",
                "trend_direction", "direction_consistency_warning", "mss_consistency_warning"
            ]
            analysis_extract = {k: v for k, v in analysis_data.items() if k in relevant_fields}
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
    success = log_feedback(
        feedback_data.session_id,
        feedback_data.question,
        feedback_data.answer,
        feedback_data.feedback,
        feedback_data.query_type,
        feedback_data.analysis_data
    )
    
    if success:
        return {"status": "success", "message": "Feedback înregistrat cu succes. Mulțumim!"}
    else:
        raise HTTPException(
            status_code=500,
            detail="Nu am putut înregistra feedback-ul. Te rugăm să încerci din nou mai târziu."
        )

# ---------------------------------------------------------------------------
# QUERY TYPE IDENTIFICATION
# ---------------------------------------------------------------------------
# Helper for determining query type to guide analysis

def identify_query_type(question: str) -> Dict[str, Any]:
    """
    Identifies the type of query to guide appropriate analysis.
    Returns a dictionary with query type flags.
    """
    question_lower = question.lower().strip()
    
    # Patterns for identification
    liquidity_patterns = [
        "liq", "lichid", "lichidit", "sunt corect notate", "marchează", "marchea", "marcate"
    ]
    
    trend_patterns = [
        "trend", "trendul", "tendință", "tendinta"
    ]
    
    mss_classification_patterns = [
        "mss normal sau", "mss agresiv sau", "mss normal sau agresiv", 
        "este un mss normal", "este un mss agresiv", "ce fel de mss",
        "este agresiv sau normal", "este normal sau agresiv"
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
    
    specific_element_patterns = {
        "liquidity": [p for p in liquidity_patterns],
        "trend": [p for p in trend_patterns],
        "mss_classification": [p for p in mss_classification_patterns],
        "displacement": [p for p in displacement_patterns],
        "fvg": [p for p in fvg_patterns],
    }
    
    # Check for specific question types first
    for element_type, patterns in specific_element_patterns.items():
        if any(p in question_lower for p in patterns):
            return {
                "type": element_type,
                "requires_full_analysis": False,
                "requires_mss_analysis": element_type == "mss_classification",
                "requires_direction_analysis": element_type in ["trend", "displacement"],
                "requires_color_analysis": True,  # Always analyze colors to improve accuracy
                "requires_fvg_analysis": element_type == "fvg"
            }
    
    # If not a specific element question, check if it's a trade evaluation
    is_trade_evaluation = any(p in question_lower for p in trade_evaluation_patterns)
    
    # Default to general question that needs visual details but possibly not full analysis
    return {
        "type": "trade_evaluation" if is_trade_evaluation else "general",
        "requires_full_analysis": is_trade_evaluation,
        "requires_mss_analysis": is_trade_evaluation,
        "requires_direction_analysis": True,
        "requires_color_analysis": True,
        "requires_fvg_analysis": is_trade_evaluation
    }

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def extract_text_from_image(image_url: str) -> str:
    """Download an image and return ASCII-cleaned OCR text, or empty string on failure."""
    try:
        logging.info(f"Attempting OCR for image URL: {image_url}")
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        # Try to determine image format for PIL
        content_type = resp.headers.get('Content-Type', '').lower()
        img = Image.open(BytesIO(resp.content))
        text = pytesseract.image_to_string(img, lang="eng") # Consider adding 'ron' if helpful? lang="eng+ron"
        cleaned_text = "".join(ch for ch in text if ord(ch) < 128).strip() # Basic ASCII cleaning
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text) # Consolidate whitespace
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


def extract_json_from_text(text: str) -> Optional[str]: # Return Optional[str]
    """Extract JSON string from text that might contain markdown code blocks or other text."""
    logging.debug(f"Attempting to extract JSON from text: {text[:200]}...")
    # Prioritize JSON within markdown code blocks
    json_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
    match = re.search(json_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        logging.info("JSON extracted from markdown code block.")
        return extracted
    # Look for JSON directly in the text as a fallback
    # Make this more robust: look for balanced braces starting from the first {
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        potential_json = brace_match.group(0).strip()
        # Basic check: does it start/end with braces and contain quotes?
        if potential_json.startswith("{") and potential_json.endswith("}") and '"' in potential_json:
             # Attempt to parse to validate
             try:
                 json.loads(potential_json)
                 logging.info("Potential JSON object found directly in text and seems valid.")
                 return potential_json
             except json.JSONDecodeError:
                 logging.warning("Found brace-enclosed text, but it's not valid JSON.")
                 pass # Continue searching if validation fails

    logging.warning("Could not extract valid-looking JSON object from text.")
    return None # Return None if no valid JSON found

def generate_session_id() -> str:
    """Generate a unique session ID for tracking feedback"""
    timestamp = int(time.time())
    random_part = os.urandom(4).hex()
    return f"{timestamp}-{random_part}"

# ---------------------------------------------------------------------------
# ROUTES – TEXT ONLY
# ---------------------------------------------------------------------------
@app.post("/ask", response_model=Dict[str, str])
async def ask_question(request: Request) -> Dict[str, str]:
    """Handles text-only questions answered strictly from course material."""
    try:
        body = await request.json()
        question = body.get("question", "").strip()
        session_id = body.get("session_id", generate_session_id())
        
        if not question:
            logging.warning("Received empty question in /ask request.")
            return {"answer": "Te rog să specifici o întrebare.", "session_id": session_id}

        logging.info(f"Received /ask request. Question: '{question[:100]}...', Session ID: {session_id}")
        question_lower = question.lower() # Check lowercase once
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
            logging.exception(f"Unexpected error during embedding generation: {e}") # Use logging.exception
            raise HTTPException(status_code=500, detail="A apărut o eroare la procesarea întrebării.")

        # 2. Query Pinecone
        context = "" # Initialize context
        try:
            results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
            matches = results.get("matches", [])
            context = "\n\n---\n\n".join(m["metadata"].get("text", "") for m in matches if m["metadata"].get("text")).strip()
            logging.info(f"Pinecone query returned {len(matches)} matches. Context length: {len(context)}")
            logging.debug(f"DEBUG TXT - Retrieved Course Context Content:\n---\n{context[:1000]}...\n---")

            if not context:
                logging.warning("Pinecone query returned no relevant context.")
                # Handle specific definition questions
                if is_mss_agresiv_text_q:
                    logging.info("Specific question 'ce este un mss agresiv' detected, providing hardcoded definition as context was empty.")
                    return {"answer": MSS_AGRESIV_STRUCTURAL_DEFINITION.replace("Definiție Structurală MSS Agresiv: ", "") + " Dacă ești la început, este recomandat în program să nu-l folosești încă.", "session_id": session_id}
                if is_mss_normal_text_q:
                    logging.info("Specific question 'ce este un mss normal' detected, providing hardcoded definition as context was empty.")
                    return {"answer": MSS_NORMAL_STRUCTURAL_DEFINITION.replace("Definiție Structurală MSS Normal: ", ""), "session_id": session_id}
                if is_fvg_text_q:
                    logging.info("Specific question about FVG detected, providing hardcoded definition as context was empty.")
                    return {"answer": FVG_STRUCTURAL_DEFINITION.replace("Definiție Structurală FVG (Fair Value Gap): ", ""), "session_id": session_id}
                if is_displacement_text_q:
                    logging.info("Specific question about displacement detected, providing hardcoded definition as context was empty.")
                    return {"answer": DISPLACEMENT_DEFINITION.replace("Definiție Displacement: ", ""), "session_id": session_id}
                    
                return {"answer": "Nu am găsit informații relevante în materialele de curs pentru a răspunde la această întrebare.", "session_id": session_id}
                
            # Inject structural definition if question is exactly about specific concepts and context might be missing it
            definitions_to_inject = []
            if is_mss_agresiv_text_q and MSS_AGRESIV_STRUCTURAL_DEFINITION.lower() not in context.lower():
                definitions_to_inject.append(MSS_AGRESIV_STRUCTURAL_DEFINITION)
            if is_mss_normal_text_q and MSS_NORMAL_STRUCTURAL_DEFINITION.lower() not in context.lower():
                definitions_to_inject.append(MSS_NORMAL_STRUCTURAL_DEFINITION)
            if is_fvg_text_q and FVG_STRUCTURAL_DEFINITION.lower() not in context.lower():
                definitions_to_inject.append(FVG_STRUCTURAL_DEFINITION)
            if is_displacement_text_q and DISPLACEMENT_DEFINITION.lower() not in context.lower():
                definitions_to_inject.append(DISPLACEMENT_DEFINITION)
                
            if definitions_to_inject:
                definition_block = "\n\n".join(definitions_to_inject)
                context = f"{definition_block}\n\n---\n\n{context}"
                logging.info(f"Injected {len(definitions_to_inject)} definitions into context.")

        except PineconeException as e:
            logging.error(f"Pinecone query error: {e}")
            # Handle specific definition questions even if Pinecone fails
            if is_mss_agresiv_text_q:
                logging.info("Pinecone failed for 'ce este un mss agresiv', providing hardcoded definition.")
                return {"answer": MSS_AGRESIV_STRUCTURAL_DEFINITION.replace("Definiție Structurală MSS Agresiv: ", "") + " Dacă ești la început, este recomandat în program să nu-l folosești încă.", "session_id": session_id}
            if is_mss_normal_text_q:
                logging.info("Pinecone failed for 'ce este un mss normal', providing hardcoded definition.")
                return {"answer": MSS_NORMAL_STRUCTURAL_DEFINITION.replace("Definiție Structurală MSS Normal: ", ""), "session_id": session_id}
            if is_fvg_text_q:
                logging.info("Pinecone failed for FVG question, providing hardcoded definition.")
                return {"answer": FVG_STRUCTURAL_DEFINITION.replace("Definiție Structurală FVG (Fair Value Gap): ", ""), "session_id": session_id}
            if is_displacement_text_q:
                logging.info("Pinecone failed for displacement question, providing hardcoded definition.")
                return {"answer": DISPLACEMENT_DEFINITION.replace("Definiție Displacement: ", ""), "session_id": session_id}
                
            raise HTTPException(status_code=503, detail="Serviciul de căutare (Pinecone) nu este disponibil momentan.")
        except Exception as e:
            logging.exception(f"Unexpected error during Pinecone query: {e}") # Use logging.exception
            # Same handling for specific definitions on unexpected errors
            if is_mss_agresiv_text_q:
                logging.info("Unexpected error during Pinecone query for 'ce este un mss agresiv', providing hardcoded definition.")
                return {"answer": MSS_AGRESIV_STRUCTURAL_DEFINITION.replace("Definiție Structurală MSS Agresiv: ", "") + " Dacă ești la început, este recomandat în program să nu-l folosești încă.", "session_id": session_id}
            if is_mss_normal_text_q:
                logging.info("Unexpected error during Pinecone query for 'ce este un mss normal', providing hardcoded definition.")
                return {"answer": MSS_NORMAL_STRUCTURAL_DEFINITION.replace("Definiție Structurală MSS Normal: ", ""), "session_id": session_id}
            if is_fvg_text_q:
                logging.info("Unexpected error during Pinecone query for FVG question, providing hardcoded definition.")
                return {"answer": FVG_STRUCTURAL_DEFINITION.replace("Definiție Structurală FVG (Fair Value Gap): ", ""), "session_id": session_id}
            if is_displacement_text_q:
                logging.info("Unexpected error during Pinecone query for displacement question, providing hardcoded definition.")
                return {"answer": DISPLACEMENT_DEFINITION.replace("Definiție Displacement: ", ""), "session_id": session_id}
                
            raise HTTPException(status_code=500, detail="A apărut o eroare la căutarea informațiilor.")

        # 3. Generate Answer
        try:
            system_message = SYSTEM_PROMPT_CORE + "\n\nAnswer ONLY based on the provided Context."
            user_message = f"Question: {question}\n\nContext:\n{context}"

            logging.debug(f"Sending to {TEXT_MODEL}. System: {system_message[:200]}... User: {user_message[:200]}...")
            response = openai.chat.completions.create(
                model=TEXT_MODEL,
                messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
                temperature=0.3,
                max_tokens=300
            )
            answer = response.choices[0].message.content.strip()

            # Ensure specific advice is included for certain questions
            if is_mss_agresiv_text_q and "dacă ești la început" not in answer.lower():
                answer += " Dacă ești la început, este recomandat în program să nu-l folosești încă."

            logging.info(f"Successfully generated answer using {TEXT_MODEL}.")
            logging.debug(f"Generated Answer (raw): {answer[:200]}...")
            return {
                "answer": answer,
                "session_id": session_id
            }

        except (APIError, RateLimitError) as e:
            logging.error(f"OpenAI Chat API error ({TEXT_MODEL}): {e}")
            raise HTTPException(status_code=503, detail=f"Serviciul OpenAI (Chat) nu este disponibil momentan.")
        except Exception as e:
            logging.exception(f"Unexpected error during {TEXT_MODEL} answer generation: {e}") # Use logging.exception
            raise HTTPException(status_code=500, detail="A apărut o eroare la generarea răspunsului.")

    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Unhandled exception in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail="A apărut o eroare internă neașteptată.")

# ---------------------------------------------------------------------------
# ROUTES – IMAGE HYBRID (REVISED WITH IMPROVED VISUAL ANALYSIS)
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

    # Initialize variables for storing results from different stages
    detailed_vision_analysis: Dict[str, Any] = {"error": "Vision analysis not performed"} # Store detailed analysis JSON/dict
    ocr_text: str = ""
    course_context: str = ""

    # --- Identify query type for tailored analysis ---
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
            if query_info["type"] == "liquidity":
                # For liquidity specific queries
                detailed_vision_system_prompt = (
                    "You are an expert Trading Instituțional chart analyst specializing in liquidity identification. Analyze this chart "
                    "and focus ONLY on the liquidity zones marked. Output a structured JSON with these fields:"
                    "\n1. 'analysis_possible': boolean"
                    "\n2. 'visible_liquidity_zones': List their positions and whether they appear to be major or minor"
                    "\n3. 'liquidity_quality': Assess the quality of marked liquidity zones based on price action around them"
                    "\n4. 'overall_trend_direction': ONLY 'bullish', 'bearish', or 'sideways' - but focus on marked liquidity"
                    "\n5. 'candle_colors': Specifically identify what colors represent bullish vs bearish candles in THIS chart"
                    "\nDO NOT analyze trade setups, MSS structures, or displacement unless specifically marked as liquidity areas."
                    "\nOnly analyze what's clearly visible and relevant to LIQUIDITY in the image."
                )
            elif query_info["type"] == "trend":
                # For trend specific queries
                detailed_vision_system_prompt = (
                    "You are an expert Trading Instituțional chart analyst specializing in trend identification. Analyze this chart "
                    "focusing PRIMARILY on the visible trend. Output a structured JSON with these fields:"
                    "\n1. 'analysis_possible': boolean"
                    "\n2. 'trend_direction': MUST be 'bullish', 'bearish', or 'sideways'"
                    "\n3. 'trend_strength': Assess the strength and clarity of the trend"
                    "\n4. 'trend_structure': Brief description of what makes this a trend (higher highs/lows or lower highs/lows)"
                    "\n5. 'candle_colors': SPECIFICALLY identify what colors represent bullish vs bearish candles in THIS chart"
                    "\n6. 'visible_trendlines': Describe any visible trendlines or support/resistance levels"
                    "\nDO NOT analyze MSS structures or specific trade setups unless directly related to the trend."
                    "\nOnly analyze what's clearly visible in the image related to trend direction and strength."
                )
            elif query_info["type"] == "mss_classification":
                # For MSS classification specific queries
                detailed_vision_system_prompt = (
                    "You are an expert Trading Instituțional chart analyst specializing in MSS classification. Your task is to count "
                    "the candles involved in breaking structure where 'MSS' is labeled on the chart and determine if it's MSS Agresiv (ONE candle) "
                    "or MSS Normal (TWO OR MORE candles). Output a structured JSON with these fields:"
                    "\n1. 'analysis_possible': boolean"
                    "\n2. 'mss_location': Where is MSS labeled on the chart"
                    "\n3. 'breaking_candle_count': EXACT INTEGER count of candles breaking structure - this is CRITICAL"
                    "\n4. 'mss_type': MUST be EXACTLY 'agresiv' (ONE candle) or 'normal' (TWO OR MORE candles)"
                    "\n5. 'break_type': 'high' or 'low'"
                    "\n6. 'candle_direction': 'bullish' (typically green/blue) or 'bearish' (typically red)"
                    "\n7. 'candle_colors': SPECIFICALLY identify what colors represent bullish vs bearish candles in THIS chart"
                    "\nCOUNT CANDLES CAREFULLY - This is the MOST important part of your analysis."
                    "\nONLY analyze the MSS structure, not the entire trade setup."
                )
            elif query_info["type"] == "displacement":
                # For displacement specific queries
                detailed_vision_system_prompt = (
                    "You are an expert Trading Instituțional chart analyst specializing in displacement analysis. Your task is to analyze "
                    "the displacement visible in the chart. Output a structured JSON with these fields:"
                    "\n1. 'analysis_possible': boolean"
                    "\n2. 'displacement_direction': 'bullish' (price moving up) or 'bearish' (price moving down)"
                    "\n3. 'displacement_strength': Assess whether the displacement is strong, moderate, or weak"
                    "\n4. 'fvg_presence': Identify if Fair Value Gaps (FVGs) are created by the displacement"
                    "\n5. 'candle_colors': SPECIFICALLY identify what colors represent bullish vs bearish candles in THIS chart"
                    "\n6. 'trade_direction': Based on displacement, is this likely a 'short' or 'long' trade"
                    "\nFocus ONLY on the displacement aspect - the impulsive price movement creating gaps/imbalances."
                    "\nDisplacement should match trade direction: bearish displacement for short trades, bullish for long trades."
                )
            elif query_info["type"] == "fvg":
                # For FVG specific queries
                detailed_vision_system_prompt = (
                    "You are an expert Trading Instituțional chart analyst specializing in Fair Value Gap (FVG) identification. Your task is to "
                    "analyze the FVGs visible in the chart. Output a structured JSON with these fields:"
                    "\n1. 'analysis_possible': boolean"
                    "\n2. 'fvg_locations': Identify and describe where FVGs are located in the chart"
                    "\n3. 'fvg_types': For each FVG, indicate if it's bullish (created by upward movement) or bearish (created by downward movement)"
                    "\n4. 'fvg_quality': Assess the quality and clarity of the identified FVGs"
                    "\n5. 'candle_colors': SPECIFICALLY identify what colors represent bullish vs bearish candles in THIS chart"
                    "\n6. 'trade_implication': How these FVGs might affect trading decisions"
                    "\nFocus ONLY on the FVG aspects - the gaps/imbalances created by impulsive price movements."
                    "\nRemember: FVGs are created when price moves impulsively, leaving an area of no trading activity."
                )
            else:
                # For general trade evaluation queries - most comprehensive
                detailed_vision_system_prompt = (
                    "You are an expert Trading Instituțional chart analyst. Analyze this trading chart comprehensively and "
                    "output a structured JSON with your detailed findings. Follow these specific guidelines:"
                    
                    "\n1. **COLOR INTERPRETATION FIRST:**"
                    "   - SPECIFICALLY identify what colors represent bullish vs bearish candles in THIS EXACT chart"
                    "   - Identify what colors are used for different zones (red/gray often = resistance, blue/cyan/green often = support)"
                    "   - DO NOT rely solely on conventional color meanings - observe how THIS SPECIFIC chart uses colors"
                    "   - Different platforms use different color schemes - adapt to what you actually see"
                    
                    "\n2. **TRADE DIRECTION DETERMINATION:**"
                    "   - Look for labeled arrows, text ('SHORT', 'LONG'), colored zones and their arrangement"
                    "   - For SHORT trades: Entry typically at top of red zone with targets below"
                    "   - For LONG trades: Entry typically at bottom of blue/green zone with targets above"
                    "   - Look for explicit direction labels or text in the screenshot"
                    "   - VALIDATE your direction guess by checking if breaks and displacement match direction"
                    
                    "\n3. **MSS CLASSIFICATION - CRITICAL CANDLE COUNTING:**"
                    "   - Look for where 'MSS' is labeled on the chart"
                    "   - COUNT EXACTLY how many candles break structure at that point"
                    "   - ONE candle breaking = MSS Agresiv"
                    "   - TWO OR MORE candles breaking = MSS Normal"
                    "   - Identify if it's breaking a high (price moving up through resistance) or low (price moving down through support)"
                    "   - Determine if breaking candle(s) are bullish or bearish based on THIS chart's color scheme"
                    
                    "\n4. **DISPLACEMENT & FVG ANALYSIS:**"
                    "   - Identify any displacement (impulsive movement) and its direction"
                    "   - Look for any marked FVGs (Fair Value Gaps) or visible gaps in price"
                    "   - Note if FVGs align with the overall trade direction"
                    "   - For SHORT trades: Expect bearish displacement and bearish FVGs"
                    "   - For LONG trades: Expect bullish displacement and bullish FVGs"
                    
                    "\n5. **ZONES, LIQUIDITY & OUTCOME:**"
                    "   - Identify any marked liquidity zones or areas"
                    "   - Note if price has already moved after entry (for outcome assessment)"
                    "   - For SHORT trades: Winning would show price moving DOWN after entry"
                    "   - For LONG trades: Winning would show price moving UP after entry"
                    
                    "\n6. **ESSENTIAL JSON FIELDS:**"
                    "   - 'analysis_possible': boolean"
                    "   - 'candle_colors': MUST describe colors used in THIS chart for bullish/bearish candles"
                    "   - 'trade_direction': MUST be 'short', 'long', or 'undetermined' - with your confidence level"
                    "   - 'structure_break': Describe what's broken (high or low) if visible"
                    "   - 'mss_analysis': {" 
                    "       'location': Where is MSS marked in the chart,"
                    "       'breaking_candle_count': EXACT INTEGER count of candles breaking structure,"
                    "       'break_type': 'high' or 'low',"
                    "       'candle_direction': 'bullish' or 'bearish' based on THIS chart's color scheme"
                    "     }"
                    "   - 'mss_type': 'agresiv' (ONE candle) or 'normal' (TWO OR MORE candles)"
                    "   - 'displacement_analysis': Include direction ('bullish' or 'bearish') and mention any FVGs"
                    "   - 'fvg_analysis': Describe any visible Fair Value Gaps"
                    "   - 'trade_outcome': 'win', 'loss', or 'undetermined'"
                    "   - 'visible_labels': List any text labels visible on chart (like 'MSS', 'FVG', etc.)"
                )

            # Craft user prompt based on query type
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
                    f"Analyze this trading chart FOCUSING ONLY ON MSS CLASSIFICATION. "
                    f"The user is asking: '{payload.question}'. "
                    f"Be sure to identify SPECIFICALLY what colors represent bullish vs bearish candles in THIS chart. "
                    f"Your PRIMARY task is to COUNT THE EXACT NUMBER OF CANDLES breaking structure where 'MSS' is marked. "
                    f"Remember: ONE candle breaking = MSS Agresiv, TWO OR MORE candles breaking = MSS Normal. "
                    f"Also identify if it's breaking a high or low, and if the breaking candles are bullish or bearish. "
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
            else:
                detailed_vision_user_prompt = (
                    f"Analyze this trading chart according to Trading Instituțional methodology. "
                    f"The user is asking: '{payload.question}'. "
                    f"First, identify SPECIFICALLY what colors represent bullish vs bearish candles in THIS chart. "
                    f"Then identify the trade direction (SHORT vs LONG) using all available visual cues. "
                    f"If there's a marked MSS, COUNT THE EXACT NUMBER OF CANDLES breaking structure: "
                    f"ONE candle = MSS Agresiv, TWO OR MORE candles = MSS Normal. "
                    f"Look for displacement and FVGs (Fair Value Gaps) created by impulsive moves. "
                    f"Ensure directional consistency in your analysis. "
                    f"Provide your comprehensive structured analysis as JSON."
                )

            vision_resp = openai.chat.completions.create(
                model=VISION_MODEL, 
                messages=[
                    {"role": "system", "content": detailed_vision_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": payload.image_url}},
                            {"type": "text", "text": detailed_vision_user_prompt},
                        ],
                    },
                ],
                max_tokens=1500,
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            # --- Process the response (expecting JSON directly) ---
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
                     
                     # Extract color scheme information for better interpretation
                     candle_colors = detailed_vision_analysis.get("candle_colors", "")
                     if candle_colors:
                         logging.info(f"Detected candle colors: {candle_colors}")
                         # Store candle color information prominently for downstream use
                         detailed_vision_analysis["_detected_color_scheme"] = candle_colors
                     
                     # Validate MSS classification if it's provided
                     if "mss_analysis" in detailed_vision_analysis and isinstance(detailed_vision_analysis["mss_analysis"], dict):
                         breaking_candle_count = detailed_vision_analysis["mss_analysis"].get("breaking_candle_count", 0)
                         mss_type = detailed_vision_analysis.get("mss_type", "")
                         
                         # Add double-check field to help final analysis
                         if breaking_candle_count == 1 and mss_type != "agresiv":
                             logging.warning("⚠️ Inconsistency: 1 breaking candle but not classified as MSS Agresiv")
                             detailed_vision_analysis["mss_consistency_warning"] = "candle_count_1_but_not_agresiv"
                         elif breaking_candle_count > 1 and mss_type != "normal":
                             logging.warning(f"⚠️ Inconsistency: {breaking_candle_count} breaking candles but not classified as MSS Normal")
                             detailed_vision_analysis["mss_consistency_warning"] = "multiple_candles_but_not_normal"
                     
                     # Add additional validation for trade direction consistency
                     if query_info["requires_direction_analysis"]:
                         trade_direction = detailed_vision_analysis.get("trade_direction", "").lower()
                         structure_break = detailed_vision_analysis.get("structure_break", "").lower()
                         mss_analysis = detailed_vision_analysis.get("mss_analysis", {})
                         break_type = mss_analysis.get("break_type", "").lower() if isinstance(mss_analysis, dict) else ""
                         
                         # Check for direction consistency
                         if trade_direction == "long" and ("low" in break_type or "low" in structure_break):
                             logging.warning("⚠️ Direction inconsistency: LONG trade with break of LOW")
                             detailed_vision_analysis["direction_consistency_warning"] = "long_with_break_of_low"
                         elif trade_direction == "short" and ("high" in break_type or "high" in structure_break):
                             logging.warning("⚠️ Direction inconsistency: SHORT trade with break of HIGH") 
                             detailed_vision_analysis["direction_consistency_warning"] = "short_with_break_of_high"
                         
                         # Check displacement consistency (if available)
                         displacement_analysis = detailed_vision_analysis.get("displacement_analysis", {})
                         if isinstance(displacement_analysis, dict):
                             displacement_direction = displacement_analysis.get("direction", "").lower()
                             if trade_direction == "long" and displacement_direction == "bearish":
                                 logging.warning("⚠️ Direction inconsistency: LONG trade with BEARISH displacement")
                                 detailed_vision_analysis["direction_consistency_warning"] = (
                                     detailed_vision_analysis.get("direction_consistency_warning", "") + 
                                     " long_with_bearish_displacement"
                                 ).strip()
                             elif trade_direction == "short" and displacement_direction == "bullish":
                                 logging.warning("⚠️ Direction inconsistency: SHORT trade with BULLISH displacement")
                                 detailed_vision_analysis["direction_consistency_warning"] = (
                                     detailed_vision_analysis.get("direction_consistency_warning", "") + 
                                     " short_with_bullish_displacement"
                                 ).strip()

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

        except (APIError, RateLimitError) as e:
            logging.error(f"OpenAI Vision API error: {e}")
            detailed_vision_analysis = {"error": f"Vision API error: {str(e)}"}
        except Exception as e:
            logging.exception(f"Unexpected error during Vision processing: {e}")
            detailed_vision_analysis = {"error": "Unexpected vision processing error"}

        # --- Run OCR (keep separate for now) ---
        ocr_text = extract_text_from_image(payload.image_url)

    except HTTPException:
        raise # Re-raise validation/network errors related to image access
    except Exception as e:
        logging.exception(f"Unhandled exception during Vision/OCR stage: {e}")
        if "error" not in detailed_vision_analysis:
             detailed_vision_analysis = {"error": "Unhandled exception in Vision/OCR stage"}

    # --- 2️⃣ Vector Search ---
    try:
        query_parts = [f"Question: {payload.question}"]
        if len(ocr_text) > 10: query_parts.append(f"OCR Text Snippet: {ocr_text[:200]}") # Use OCR text if significant
        
        # Add identified query type to help context retrieval
        query_parts.append(f"Query type: {query_info['type']}")
        
        # Add detected color scheme if available
        if "_detected_color_scheme" in detailed_vision_analysis:
            query_parts.append(f"Chart color scheme: {detailed_vision_analysis['_detected_color_scheme']}")
        
        # Add direction if available for context
        if detailed_vision_analysis.get("trade_direction") in ["long", "short"]:
            query_parts.append(f"Trade direction: {detailed_vision_analysis.get('trade_direction')}")
        
        # Add trend if available
        if detailed_vision_analysis.get("trend_direction") in ["bullish", "bearish", "sideways"]:
            query_parts.append(f"Trend direction: {detailed_vision_analysis.get('trend_direction')}")
            
        # Add MSS type if available
        if detailed_vision_analysis.get("mss_type") in ["agresiv", "normal"]:
            query_parts.append(f"MSS type: {detailed_vision_analysis.get('mss_type')}")
        
        # Add FVG information if available
        if query_info["requires_fvg_analysis"] and "fvg_analysis" in detailed_vision_analysis:
            fvg_info = detailed_vision_analysis.get("fvg_analysis", "")
            if isinstance(fvg_info, dict):
                fvg_info = json.dumps(fvg_info)
            query_parts.append(f"FVG analysis: {str(fvg_info)[:100]}")
            
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

        # Handle definitions for related questions based on query type
        definitions_to_add = []
        
        if query_info["type"] == "mss_classification" or "mss" in payload.question.lower():
            # Include both MSS definitions for MSS-related queries
            if MSS_AGRESIV_STRUCTURAL_DEFINITION.lower() not in course_context.lower():
                definitions_to_add.append(MSS_AGRESIV_STRUCTURAL_DEFINITION)
            if MSS_NORMAL_STRUCTURAL_DEFINITION.lower() not in course_context.lower():
                definitions_to_add.append(MSS_NORMAL_STRUCTURAL_DEFINITION)
        
        if query_info["type"] == "fvg" or "fvg" in payload.question.lower() or "fair value gap" in payload.question.lower():
            if FVG_STRUCTURAL_DEFINITION.lower() not in course_context.lower():
                definitions_to_add.append(FVG_STRUCTURAL_DEFINITION)
        
        if query_info["type"] == "displacement" or "displacement" in payload.question.lower():
            if DISPLACEMENT_DEFINITION.lower() not in course_context.lower():
                definitions_to_add.append(DISPLACEMENT_DEFINITION)
            
        if definitions_to_add:
            definition_block = "\n\n".join(definitions_to_add)
            course_context = f"{definition_block}\n\n---\n\n{course_context}"
            logging.info(f"Injected {len(definitions_to_add)} definitions into context.")

        if not course_context:
            logging.warning("Pinecone query returned no relevant context for the hybrid query.")
            course_context = "[Eroare: Niciun context specific din curs nu a fost găsit pentru această combinație.]"
            
            # Add minimal context based on query type if no context was found
            minimal_definitions = []
            if query_info["type"] == "mss_classification" or "mss" in payload.question.lower():
                minimal_definitions.extend([MSS_AGRESIV_STRUCTURAL_DEFINITION, MSS_NORMAL_STRUCTURAL_DEFINITION])
            if query_info["type"] == "fvg" or "fvg" in payload.question.lower():
                minimal_definitions.append(FVG_STRUCTURAL_DEFINITION)
            if query_info["type"] == "displacement" or "displacement" in payload.question.lower():
                minimal_definitions.append(DISPLACEMENT_DEFINITION)
            
            if minimal_definitions:
                course_context += "\n\n---\n\n" + "\n\n".join(minimal_definitions)

    except (APIError, RateLimitError) as e:
        logging.error(f"OpenAI Embedding API error during hybrid search: {e}")
        course_context = "[Eroare: Nu s-a putut genera embedding pentru căutare context]"
        # Add minimal context based on query type if embedding failed
        minimal_definitions = []
        if query_info["type"] == "mss_classification" or "mss" in payload.question.lower():
            minimal_definitions.extend([MSS_AGRESIV_STRUCTURAL_DEFINITION, MSS_NORMAL_STRUCTURAL_DEFINITION])
        if query_info["type"] == "fvg" or "fvg" in payload.question.lower():
            minimal_definitions.append(FVG_STRUCTURAL_DEFINITION)
        if query_info["type"] == "displacement" or "displacement" in payload.question.lower():
            minimal_definitions.append(DISPLACEMENT_DEFINITION)
        
        if minimal_definitions:
            course_context += "\n\n---\n\n" + "\n\n".join(minimal_definitions)
    except PineconeException as e:
        logging.error(f"Pinecone query error during hybrid search: {e}")
        course_context = "[Eroare: Nu s-a putut căuta în materialele de curs]"
        # Same fallback with minimal definitions
        minimal_definitions = []
        if query_info["type"] == "mss_classification" or "mss" in payload.question.lower():
            minimal_definitions.extend([MSS_AGRESIV_STRUCTURAL_DEFINITION, MSS_NORMAL_STRUCTURAL_DEFINITION])
        if query_info["type"] == "fvg" or "fvg" in payload.question.lower():
            minimal_definitions.append(FVG_STRUCTURAL_DEFINITION)
        if query_info["type"] == "displacement" or "displacement" in payload.question.lower():
            minimal_definitions.append(DISPLACEMENT_DEFINITION)
        
        if minimal_definitions:
            course_context += "\n\n---\n\n" + "\n\n".join(minimal_definitions)
    except Exception as e:
        logging.exception(f"Unexpected error during vector search stage: {e}")
        course_context = "[Eroare: Problemă neașteptată la căutarea contextului]"
        # Same fallback with minimal definitions
        minimal_definitions = []
        if query_info["type"] == "mss_classification" or "mss" in payload.question.lower():
            minimal_definitions.extend([MSS_AGRESIV_STRUCTURAL_DEFINITION, MSS_NORMAL_STRUCTURAL_DEFINITION])
        if query_info["type"] == "fvg" or "fvg" in payload.question.lower():
            minimal_definitions.append(FVG_STRUCTURAL_DEFINITION)
        if query_info["type"] == "displacement" or "displacement" in payload.question.lower():
            minimal_definitions.append(DISPLACEMENT_DEFINITION)
        
        if minimal_definitions:
            course_context += "\n\n---\n\n" + "\n\n".join(minimal_definitions)

    # --- 3️⃣ Final Answer Generation ---
    try:
        # --- Prepare the visual analysis report string ---
        visual_analysis_report_str = "[Eroare la formatarea raportului vizual]" # Default error
        try:
             visual_analysis_report_str = json.dumps(detailed_vision_analysis, indent=2, ensure_ascii=False)
        except Exception:
             visual_analysis_report_str = str(detailed_vision_analysis) # Fallback
        logging.debug(f"Visual Analysis Report string for prompt:\n{visual_analysis_report_str}")


        # --- Define the system prompt based on query type ---
        if query_info["type"] == "liquidity":
            final_system_prompt = SYSTEM_PROMPT_CORE + (
                "\n\n--- Instructions for Liquidity Zone Analysis ---"
                "\n1. You are provided with a Visual Analysis Report (JSON) focused on LIQUIDITY ZONES visible in the user's chart."
                "\n2. You also have Course Material Context providing rules about liquidity in trading."
                "\n3. Your task is to ONLY evaluate the liquidity zones marked in the chart and answer the user's specific question."
                "\n4. DO NOT analyze or mention MSS, displacement, or trade setups unless directly relevant to liquidity."
                "\n5. Focus on confirming if the liquidity zones are correctly identified and how they relate to the Trading Instituțional methodology."
                "\n6. For liquidity questions, explain: Liquidity is where stop orders accumulate, creating potential targets for price to move toward."
                "\n7. Mention that high-quality liquidity zones are those most visible on the chart, where many stop orders may be gathered."
                "\n8. Pay attention to the SPECIFIC COLOR SCHEME used in this chart as identified in the Visual Analysis Report."
                "\n9. Be concise, direct, and focus ONLY on the liquidity aspects of the chart."
                "\n10. Expected response for liquidity questions: Confirm if zones are valid, mention quality criteria, give brief guidance."
            )
        elif query_info["type"] == "trend":
            final_system_prompt = SYSTEM_PROMPT_CORE + (
                "\n\n--- Instructions for Trend Analysis ---"
                "\n1. You are provided with a Visual Analysis Report (JSON) focused on TREND DIRECTION visible in the user's chart."
                "\n2. You also have Course Material Context about trend following strategies."
                "\n3. Your task is to ONLY evaluate the trend characteristics and answer the user's specific question."
                "\n4. DO NOT analyze or mention MSS, displacement, or specific trade setups unless directly relevant to trend."
                "\n5. Focus on confirming if there is a clear trend, its direction (bullish/bearish), and strength."
                "\n6. For trend questions with minor liquidity mentioned, explain how minor liquidity helps sustain trends."
                "\n7. Pay attention to the SPECIFIC COLOR SCHEME used in this chart as identified in the Visual Analysis Report."
                "\n8. Be concise, direct, and focus ONLY on the trend aspects of the chart."
                "\n9. Expected response for trend questions: Confirm trend direction (bullish/bearish/sideways), mention strength, give brief guidance."
            )
        elif query_info["type"] == "mss_classification":
            final_system_prompt = SYSTEM_PROMPT_CORE + (
                "\n\n--- Instructions for MSS Classification ---"
                "\n1. You are provided with a Visual Analysis Report (JSON) focused on MSS CLASSIFICATION in the user's chart."
                "\n2. You also have Course Material Context about MSS types and definitions."
                "\n3. Your task is to DETERMINE if the MSS shown is AGRESIV or NORMAL based STRICTLY on CANDLE COUNT:"
                "   - MSS Agresiv = EXACTLY ONE candle breaking structure"
                "   - MSS Normal = TWO OR MORE candles breaking structure"
                "\n4. Also state if it's a break of HIGH or LOW, and whether breaking candle(s) are BULLISH or BEARISH."
                "\n5. Pay attention to the SPECIFIC COLOR SCHEME used in this chart as identified in the Visual Analysis Report."
                "\n6. If there's an inconsistency in the analysis, prioritize the actual candle count:"
                "   - If report shows 1 candle but calls it 'normal' - it's actually AGRESIV"
                "   - If report shows 2+ candles but calls it 'agresiv' - it's actually NORMAL"
                "\n7. Be concise, direct, and ONLY classify the MSS without analyzing the entire trade setup."
                "\n8. Always clearly state the EXACT classification and the reason (candle count)."
                "\n9. Expected response for MSS classification: State MSS type, explain why (candle count), mention break direction (high/low)."
            )
        elif query_info["type"] == "displacement":
            final_system_prompt = SYSTEM_PROMPT_CORE + (
                "\n\n--- Instructions for Displacement Analysis ---"
                "\n1. You are provided with a Visual Analysis Report (JSON) focused on DISPLACEMENT visible in the user's chart."
                "\n2. You also have Course Material Context about displacement in trading."
                "\n3. Your task is to ONLY evaluate the displacement characteristics and answer the user's specific question."
                "\n4. Focus on the direction of displacement (bullish/bearish) and its strength."
                "\n5. Mention any FVGs (Fair Value Gaps) created by the displacement if visible."
                "\n6. Pay attention to the SPECIFIC COLOR SCHEME used in this chart as identified in the Visual Analysis Report."
                "\n7. Explain how displacement relates to trade direction: bearish displacement for SHORT trades, bullish for LONG trades."
                "\n8. Be concise, direct, and focus ONLY on the displacement aspects of the chart."
                "\n9. Expected response for displacement questions: Confirm displacement direction and strength, mention FVGs if visible."
            )
                elif query_info["type"] == "fvg":
            final_system_prompt = SYSTEM_PROMPT_CORE + (
                "\n\n--- Instructions for FVG Analysis ---"
                "\n1. You are provided with a Visual Analysis Report (JSON) focused on FVGs (Fair Value Gaps) visible in the user's chart."
                "\n2. You also have Course Material Context about FVGs in trading."
                "\n3. Your task is to ONLY evaluate the FVG characteristics and answer the user's specific question."
                "\n4. Focus on identifying FVGs, their direction (bullish/bearish), and quality."
                "\n5. Explain that FVGs are created when price moves impulsively, leaving areas where no trading has occurred."
                "\n6. Pay attention to the SPECIFIC COLOR SCHEME used in this chart as identified in the Visual Analysis Report."
                "\n7. Relate FVGs to the overall trade direction: bearish FVGs for SHORT trades, bullish FVGs for LONG trades."
                "\n8. Be concise, direct, and focus ONLY on the FVG aspects of the chart."
                "\n9. Expected response for FVG questions: Identify FVGs, their direction, quality, and implications for the trade."
            )
        else:
            # Default for general or trade evaluation queries
            final_system_prompt = SYSTEM_PROMPT_CORE + (
                "\n\n--- Instructions for Trade Setup Evaluation ---"
                "\n1. You are provided with a Visual Analysis Report (JSON) of the user's trading chart."
                "\n2. You also have Course Material Context from Trading Instituțional program."
                "\n3. Your task is to evaluate the chart based STRICTLY on the Trading Instituțional methodology."
                "\n4. Pay special attention to these elements:"
                "   - MSS Type (agresiv vs normal) based on CANDLE COUNT"
                "   - Direction consistency (break/displacement should match trade direction)"
                "   - FVGs (Fair Value Gaps) quality and alignment with trade direction"
                "   - Liquidity zones and their quality"
                "\n5. Pay attention to the SPECIFIC COLOR SCHEME used in this chart as identified in the Visual Analysis Report."
                "\n6. Be direct, concise, and focus ONLY on what's visible in the chart."
                "\n7. Do not make predictions or provide 'would be better if' advice."
                "\n8. If there are inconsistencies in the setup (e.g., direction mismatch), point them out."
                "\n9. Expected response for trade evaluations: Assess structure validity, mention direction consistency, evaluate overall quality."
            )

        # --- User prompt for final answer generation ---
        try:
            final_user_prompt = (
                f"Question: {payload.question}\n\n"
                f"Visual Analysis Report:\n{visual_analysis_report_str}\n\n"
                f"Course Context:\n{course_context}"
            )

            logging.debug(f"Final system prompt length: {len(final_system_prompt)}")
            logging.debug(f"Final user prompt length: {len(final_user_prompt)}")

            # Send to OpenAI for final response
            chat_completion = openai.chat.completions.create(
                model=COMPLETION_MODEL,
                messages=[
                    {"role": "system", "content": final_system_prompt},
                    {"role": "user", "content": final_user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            final_answer = chat_completion.choices[0].message.content.strip()
            
            # Respond to the user
            return {
                "answer": final_answer,
                "session_id": session_id
            }
            
        except (APIError, RateLimitError) as e:
            logging.error(f"OpenAI Chat API error during final generation: {e}")
            error_msg = "Nu am putut genera un răspuns final. Serviciul OpenAI nu este disponibil momentan."
            return {"answer": error_msg, "session_id": session_id}
        except Exception as e:
            logging.exception(f"Unexpected error during final answer generation: {e}")
            error_msg = "A apărut o eroare la generarea răspunsului final. Te rugăm să încerci din nou."
            return {"answer": error_msg, "session_id": session_id}
            
    except Exception as e:
        logging.exception(f"Unhandled exception in image-hybrid response generation: {e}")
        error_msg = "A apărut o eroare neașteptată la procesarea răspunsului. Te rugăm să încerci din nou."
        return {"answer": error_msg, "session_id": session_id}

# --- Health check endpoint ---
@app.get("/health")
async def health_check():
    """Simple health check endpoint to verify the service is running."""
    try:
        # Check OpenAI API connection
        openai.embeddings.create(model=EMBEDDING_MODEL, input=["test"])
        # Check Pinecone connection (simple query)
        test_vector = [0.0] * 1536  # Empty vector for test
        index.query(vector=test_vector, top_k=1)
        
        return {
            "status": "healthy",
            "openai": "connected",
            "pinecone": "connected",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
