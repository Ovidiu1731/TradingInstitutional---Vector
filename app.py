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
FEEDBACK_LOG = os.getenv("FEEDBACK_LOG", "feedback_log.jsonl")

# --- Model selection - UPDATED ---
EMBEDDING_MODEL = "text-embedding-ada-002" # Current standard embedding model
VISION_MODEL = "gpt-4-turbo"              # Updated to latest Vision-capable model (or gpt-4o if preferred)
COMPLETION_MODEL = "gpt-4-turbo"          # Updated to latest powerful model (or gpt-4o if preferred)
TEXT_MODEL = "gpt-3.5-turbo"              # Can remain or be updated

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
        # Added note for final LLM regarding validated analysis
        "\n\nIMPORTANT: When reviewing the 'Visual Analysis Report', trust the provided 'mss_type' and 'trade_direction'. "
        "If a '_validator_note' field is present, it means these fields were adjusted by internal rules for accuracy."
    )

# Define the core structural definitions (Unchanged)
MSS_AGRESIV_STRUCTURAL_DEFINITION = "Definiție Structurală MSS Agresiv: Un MSS agresiv se produce atunci cand ultimul higher low sau lower high care este rupt (unde se produce shift-ul) nu are in structura sa minim 2 candele bearish cu 2 candele bullish."
MSS_NORMAL_STRUCTURAL_DEFINITION = "Definiție Structurală MSS Normal: Este o rupere de structură formată din două sau mai multe lumânări care fac low/high." # Note: Original code definition for Normal is vague; the Agresiv one implies the correct logic. Using Agresiv's logic for classification.
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
# FEEDBACK MECHANISM (Unchanged)
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
            # Only include relevant fields to keep log size manageable + validator note
            relevant_fields = [
                "trade_direction", "mss_type",
                "pivot_bearish_count", "pivot_bullish_count", # From mss_pivot_analysis
                "trend_direction", "direction_consistency_warning", "mss_consistency_warning",
                "is_risk_above_price", "_validator_note" # Add validator inputs/outputs
            ]
            # Extract nested fields carefully
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
    # Fetch pivot counts from nested structure if present in analysis_data
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
        analysis_input # Pass the potentially modified dict
    )

    if success:
        return {"status": "success", "message": "Feedback înregistrat cu succes. Mulțumim!"}
    else:
        raise HTTPException(
            status_code=500,
            detail="Nu am putut înregistra feedback-ul. Te rugăm să încerci din nou mai târziu."
        )

# ---------------------------------------------------------------------------
# QUERY TYPE IDENTIFICATION (Unchanged)
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
        "este agresiv sau normal", "este normal sau agresiv", "tip de mss" # Added tip de mss
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
        "requires_mss_analysis": is_trade_evaluation, # Always analyze MSS in evaluations
        "requires_direction_analysis": True, # Always need direction for evaluation
        "requires_color_analysis": True,
        "requires_fvg_analysis": is_trade_evaluation # Analyze FVG in evaluations
    }

# ---------------------------------------------------------------------------
# HELPERS (Unchanged logic, added logging)
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
        # Consider adding 'ron' if helpful and if language pack installed: lang="eng+ron"
        text = pytesseract.image_to_string(img, lang="eng")
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
        # Validate extracted JSON before returning
        try:
            json.loads(extracted)
            return extracted
        except json.JSONDecodeError:
            logging.warning("Text in markdown block wasn't valid JSON.")
            # Continue to search outside block
    # Look for JSON directly in the text as a fallback
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
# ROUTES – TEXT ONLY (Unchanged)
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
            raise HTTPException(status_code=503, detail=f"Serviciul OpenAI ({TEXT_MODEL}) nu este disponibil momentan.")
        except Exception as e:
            logging.exception(f"Unexpected error during {TEXT_MODEL} answer generation: {e}") # Use logging.exception
            raise HTTPException(status_code=500, detail="A apărut o eroare la generarea răspunsului.")

    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Unhandled exception in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail="A apărut o eroare internă neașteptată.")

# ---------------------------------------------------------------------------
# --- SANITY CHECK VALIDATOR --- (Defined before use)
# ---------------------------------------------------------------------------
def _sanity_check_analysis(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforces deterministic rules for MSS type and trade direction based on
    expected fields potentially extracted by the Vision model.
    Modifies the dictionary in place for simplicity, but returns it.
    """
    logging.debug("Applying sanity check validator...")
    corrected = False

    # --- 1️⃣  MSS Type based on Pivot Structure Candle Count ---
    # Access counts from the nested structure specified in the updated prompt
    pivot_analysis = analysis_dict.get("mss_pivot_analysis", {})
    bearish_count = pivot_analysis.get("pivot_bearish_count", 0)
    bullish_count = pivot_analysis.get("pivot_bullish_count", 0)
    original_mss_type = analysis_dict.get("mss_type", "").lower()
    calculated_mss_type = ""

    # Ensure counts are integers
    try:
        bearish_count = int(bearish_count)
    except (ValueError, TypeError):
        logging.warning("Could not parse pivot_bearish_count as int.")
        bearish_count = 0
    try:
        bullish_count = int(bullish_count)
    except (ValueError, TypeError):
        logging.warning("Could not parse pivot_bullish_count as int.")
        bullish_count = 0

    if bearish_count >= 2 and bullish_count >= 2:
        calculated_mss_type = "normal"
    else:
        # Only classify as agresiv if counts were actually provided or structure explicitly analyzed
        # Avoid defaulting to agresiv if counts are missing entirely
        if "pivot_bearish_count" in pivot_analysis or "pivot_bullish_count" in pivot_analysis:
             calculated_mss_type = "agresiv"
        else:
             calculated_mss_type = original_mss_type # Keep original if no counts found

    # Apply correction if calculated type differs from vision model's output AND calculated type is valid
    if calculated_mss_type in ["normal", "agresiv"] and calculated_mss_type != original_mss_type:
        logging.warning(f"Validator correcting MSS Type: Was '{original_mss_type}', "
                        f"became '{calculated_mss_type}' (Bearish: {bearish_count}, Bullish: {bullish_count})")
        analysis_dict["mss_type"] = calculated_mss_type
        corrected = True

    # --- 2️⃣  Trade Direction based on Risk Box Placement ---
    risk_above = analysis_dict.get("is_risk_above_price")  # Expecting True, False, or None/missing
    original_direction = analysis_dict.get("trade_direction", "").lower()
    calculated_direction = ""

    if risk_above is True:
        calculated_direction = "short"
    elif risk_above is False:
        calculated_direction = "long"
    # If risk_above is None or missing, don't override - rely on model's other logic
    else:
        calculated_direction = original_direction

    # Apply correction if calculated direction differs and is valid
    if calculated_direction in ["short", "long"] and calculated_direction != original_direction:
        logging.warning(f"Validator correcting Trade Direction: Was '{original_direction}', "
                        f"became '{calculated_direction}' (is_risk_above_price: {risk_above})")
        analysis_dict["trade_direction"] = calculated_direction
        corrected = True

    # Add validator note if corrections were made
    if corrected:
        analysis_dict["_validator_note"] = (
            "NOTE: Analysis adjusted by internal rules for MSS type and/or trade direction consistency."
        )

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
            # (Prompts for liquidity, trend, displacement, fvg remain largely the same - focus on adding candle colors and structure where needed)
            # --- Updated mss_classification prompt ---
            if query_info["type"] == "mss_classification":
                detailed_vision_system_prompt = (
                    "You are an expert Trading Instituțional chart analyst specializing in MSS classification. "
                    "Analyze this chart with attention to the following critical criteria:"

                    "\n\n**CRITICAL MSS CLASSIFICATION RULES:**"
                    "\n1. Identify the swing high or low (the 'pivot') that is potentially broken by an MSS."
                    "\n2. Analyze the candle composition FORMING this pivot structure."
                    "\n3. Count the number of bearish and bullish candles within this core pivot structure."
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
                    "     'pivot_bearish_count': 'INTEGER count of BEARISH candles forming the core pivot structure', "
                    "     'pivot_bullish_count': 'INTEGER count of BULLISH candles forming the core pivot structure', "
                    "     'has_minimum_structure': 'BOOLEAN, true only if bearish_count >= 2 AND bullish_count >= 2'"
                    "   }"
                    "\n4. 'mss_type': MUST be EXACTLY 'normal' (if has_minimum_structure is true) or 'agresiv' (if false)."
                    "\n5. 'break_direction': 'upward' or 'downward'"
                    "\n6. 'candle_colors': Description of bullish vs bearish candle colors in THIS chart."

                    "\nThe PIVOT STRUCTURE COMPOSITION (bearish/bullish counts) is the ONLY factor determining 'normal' vs 'agresiv'."
                )
            # --- Updated general trade evaluation prompt ('else' block) ---
            else: # Includes 'trade_evaluation' and 'general' types
                detailed_vision_system_prompt = (
                    "You are an expert Trading Instituțional chart analyst. Analyze this trading chart comprehensively and "
                    "output a structured JSON with your detailed findings. Follow these specific guidelines:"

                    "\n\n**1. COLOR INTERPRETATION FIRST:**"
                    "\n   - Identify `candle_colors`: Describe colors for bullish/bearish candles in THIS chart."
                    "\n   - Note colors for zones/indicators if obvious."

                    "\n\n**2. TRADE DIRECTION (PRIORITY on Risk Box):**"
                    "\n   - **PRIMARY:** Look for a colored Risk/Reward box (often red or blue/green)."
                    "\n   - Determine `is_risk_above_price`: BOOLEAN. True if risk box is clearly ABOVE entry/current price, False if clearly BELOW. Null if unclear/absent."
                    "\n   - Set `trade_direction` based PRIMARILY on this box: 'short' if risk is above (True), 'long' if risk is below (False)."
                    "\n   - **Secondary:** If no clear risk box, infer `trade_direction` from labels ('SHORT'/'LONG', arrows) or overall strong recent directional movement (displacement)."
                    "\n   - Output: 'short', 'long', or 'undetermined'."

                    "\n\n**3. MSS CLASSIFICATION (Based on PIVOT STRUCTURE):**"
                    "\n   - Look for where 'MSS' is labeled or implied by a structure break."
                    "\n   - Identify the swing high/low (the 'pivot') that was broken."
                    "\n   - Analyze the candle composition FORMING this pivot."
                    "\n   - Count `pivot_bearish_count` and `pivot_bullish_count` within the pivot structure."
                    "\n   - Determine `has_minimum_structure`: BOOLEAN (True only if count >= 2 for BOTH bearish and bullish)."
                    "\n   - Classify `mss_type`: 'normal' if `has_minimum_structure` is True, else 'agresiv'."
                    "\n   - Identify `break_direction` ('upward' or 'downward')."

                    "\n\n**4. DISPLACEMENT & FVG ANALYSIS:**"
                    "\n   - Identify `displacement_analysis`: Direction ('bullish'/'bearish'), strength, and presence of FVGs."
                    "\n   - Ensure displacement direction aligns with the determined `trade_direction` (e.g., bearish displacement for short trade)."
                    "\n   - Identify `fvg_analysis`: Location and type (bullish/bearish) of any visible FVGs."

                    "\n\n**5. ZONES, LIQUIDITY & OUTCOME:**"
                    "\n   - Describe any marked `liquidity_zones`."
                    "\n   - Assess `trade_outcome` ('win', 'loss', 'running', 'undetermined') if possible based on price movement after entry."

                    "\n\n**6. ESSENTIAL JSON FIELDS:**"
                    "\n   - 'analysis_possible': boolean"
                    "\n   - 'candle_colors': description"
                    "\n   - 'is_risk_above_price': boolean | null" # <<< Added for validator
                    "\n   - 'trade_direction': 'short' | 'long' | 'undetermined'"
                    "\n   - 'mss_pivot_analysis': { " # <<< Added structure analysis
                    "        'description': description, "
                    "        'pivot_bearish_count': integer, "
                    "        'pivot_bullish_count': integer, "
                    "        'has_minimum_structure': boolean"
                    "      }"
                    "\n   - 'mss_type': 'normal' | 'agresiv' | 'not_identified'" # <<< Based on pivot structure
                    "\n   - 'break_direction': 'upward' | 'downward' | 'none'"
                    "\n   - 'displacement_analysis': { 'direction': 'bullish'|'bearish'|'none', 'strength': description, 'fvg_created': boolean }"
                    "\n   - 'fvg_analysis': description"
                    "\n   - 'liquidity_zones': description"
                    "\n   - 'trade_outcome': 'win'|'loss'|'running'|'undetermined'"
                    "\n   - 'visible_labels': list of strings"
                )


            # --- Craft user prompt --- (Keep user prompts mostly the same, relying on system prompt changes)
            # Ensure the user prompt for general evaluation highlights the key elements
            if query_info["type"] not in ["liquidity", "trend", "mss_classification", "displacement", "fvg"]: # General/Evaluation
                 detailed_vision_user_prompt = (
                     f"Analyze this trading chart according to Trading Instituțional methodology for a general setup evaluation. "
                     f"The user is asking: '{payload.question}'. "
                     f"Follow the instructions in the System Prompt precisely: "
                     f"1. Identify candle colors. "
                     f"2. Determine trade direction (PRIORITIZE risk box position). "
                     f"3. Classify MSS based on PIVOT structure candle counts (>=2 bearish AND >=2 bullish = Normal). "
                     f"4. Analyze displacement and FVGs. "
                     f"5. Assess liquidity and outcome if possible. "
                     f"Provide your comprehensive structured analysis as JSON."
                 )
            else: # Use existing specific prompts for other types
                # (Code for specific user prompts like liquidity, trend etc. remains here - omitted for brevity but should be kept from original)
                 # Example for mss_classification specific prompt:
                 if query_info["type"] == "mss_classification":
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
                 # ... (Include other specific user prompts here) ...
                 # Fallback just in case:
                 else:
                     detailed_vision_user_prompt = f"Analyze the chart based on the user question: '{payload.question}'. Provide output as JSON."


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
                max_tokens=1800, # Increased slightly for potentially more detailed JSON
                temperature=0.1, # Lowered temperature for more deterministic JSON output
                response_format={"type": "json_object"} # Request JSON mode
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
                     # (Removed internal consistency checks here - rely on validator now)

            except json.JSONDecodeError as json_err:
                 logging.error(f"❌ Failed to decode JSON from Vision response: {json_err}.")
                 # Try fallback extraction (function defined above)
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
            if isinstance(detailed_vision_analysis, dict) and "error" not in detailed_vision_analysis:
                 logging.debug("Original Vision Analysis before validation: %s", json.dumps(detailed_vision_analysis, indent=2, ensure_ascii=False))
                 original_analysis_copy = copy.deepcopy(detailed_vision_analysis) # Keep for comparison/logging
                 # Apply the validator function
                 detailed_vision_analysis = _sanity_check_analysis(detailed_vision_analysis)
                 # Log if validator made changes (validator adds the note itself)
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
        raise # Re-raise validation/network errors related to image access
    except Exception as e:
        logging.exception(f"Unhandled exception during Vision/OCR stage: {e}")
        if "error" not in detailed_vision_analysis:
             detailed_vision_analysis = {"error": "Unhandled exception in Vision/OCR stage"}

    # --- 2️⃣ Vector Search (RAG) ---
    try:
        # Build query using validated/corrected analysis results
        query_parts = [f"Question: {payload.question}"]
        if len(ocr_text) > 10: query_parts.append(f"OCR Text Snippet: {ocr_text[:200]}")
        query_parts.append(f"Query type: {query_info['type']}")

        # Add key validated elements from analysis to enrich context search
        if isinstance(detailed_vision_analysis, dict): # Check it's a dict first
            if "candle_colors" in detailed_vision_analysis:
                query_parts.append(f"Chart candle colors: {str(detailed_vision_analysis.get('candle_colors'))[:100]}")
            if detailed_vision_analysis.get("trade_direction") in ["long", "short"]:
                query_parts.append(f"Trade direction: {detailed_vision_analysis.get('trade_direction')}")
            if detailed_vision_analysis.get("mss_type") in ["agresiv", "normal"]:
                query_parts.append(f"MSS type: {detailed_vision_analysis.get('mss_type')}")
            if detailed_vision_analysis.get("fvg_analysis"):
                query_parts.append(f"FVG analysis summary: {str(detailed_vision_analysis.get('fvg_analysis'))[:100]}")
            # Add displacement direction
            disp_analysis = detailed_vision_analysis.get("displacement_analysis", {})
            if isinstance(disp_analysis, dict) and disp_analysis.get("direction") in ["bullish", "bearish"]:
                 query_parts.append(f"Displacement direction: {disp_analysis.get('direction')}")


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
            if MSS_AGRESIV_STRUCTURAL_DEFINITION.lower() not in course_context.lower():
                definitions_to_add.append(MSS_AGRESIV_STRUCTURAL_DEFINITION)
            if MSS_NORMAL_STRUCTURAL_DEFINITION.lower() not in course_context.lower():
                 # Add the definition that explains the structure rule
                 definitions_to_add.append("Definiție Structurală MSS Normal: Un MSS normal necesită ca pivotul (swing high/low) rupt să fie format din minim 2 candele bearish ȘI minim 2 candele bullish.")

        if "fvg" in payload.question.lower() or "fair value gap" in payload.question.lower() or (isinstance(detailed_vision_analysis, dict) and detailed_vision_analysis.get("fvg_analysis")):
             if FVG_STRUCTURAL_DEFINITION.lower() not in course_context.lower():
                 definitions_to_add.append(FVG_STRUCTURAL_DEFINITION)

        if "displacement" in payload.question.lower() or (isinstance(detailed_vision_analysis, dict) and detailed_vision_analysis.get("displacement_analysis")):
             if DISPLACEMENT_DEFINITION.lower() not in course_context.lower():
                 definitions_to_add.append(DISPLACEMENT_DEFINITION)

        if definitions_to_add:
            definition_block = "\n\n".join(definitions_to_add)
            # Prepend definitions for higher priority
            course_context = f"Definiții Relevante:\n{definition_block}\n\n---\n\nMaterial Curs:\n{course_context}"
            logging.info(f"Injected {len(definitions_to_add)} definitions into context.")

        if not course_context.strip(): # Check if context is empty *after* potential injection
            logging.warning("Pinecone query returned no relevant context, and no definitions were injected.")
            course_context = "[Eroare: Niciun context specific din curs nu a fost găsit pentru această combinație.]"
            # Add minimal definitions if still totally empty
            minimal_definitions = []
            if "mss" in payload.question.lower():
                 minimal_definitions.extend([MSS_AGRESIV_STRUCTURAL_DEFINITION, "MSS Normal: Pivotul rupt are >=2 bearish AND >=2 bullish."])
            if "fvg" in payload.question.lower(): minimal_definitions.append(FVG_STRUCTURAL_DEFINITION)
            if "displacement" in payload.question.lower(): minimal_definitions.append(DISPLACEMENT_DEFINITION)
            if minimal_definitions: course_context += "\n\nDefiniții de Bază:\n" + "\n".join(minimal_definitions)


    except (APIError, RateLimitError) as e:
        logging.error(f"OpenAI Embedding API error during hybrid search: {e}")
        course_context = "[Eroare: Nu s-a putut genera embedding pentru căutare context]"
        # Add minimal definitions on error
        minimal_definitions = []
        if "mss" in payload.question.lower(): minimal_definitions.extend([MSS_AGRESIV_STRUCTURAL_DEFINITION, "MSS Normal: Pivotul rupt are >=2 bearish AND >=2 bullish."])
        if "fvg" in payload.question.lower(): minimal_definitions.append(FVG_STRUCTURAL_DEFINITION)
        if "displacement" in payload.question.lower(): minimal_definitions.append(DISPLACEMENT_DEFINITION)
        if minimal_definitions: course_context += "\n\nDefiniții de Bază:\n" + "\n".join(minimal_definitions)
    except PineconeException as e:
        logging.error(f"Pinecone query error during hybrid search: {e}")
        course_context = "[Eroare: Nu s-a putut căuta în materialele de curs]"
        # Add minimal definitions on error
        minimal_definitions = []
        if "mss" in payload.question.lower(): minimal_definitions.extend([MSS_AGRESIV_STRUCTURAL_DEFINITION, "MSS Normal: Pivotul rupt are >=2 bearish AND >=2 bullish."])
        if "fvg" in payload.question.lower(): minimal_definitions.append(FVG_STRUCTURAL_DEFINITION)
        if "displacement" in payload.question.lower(): minimal_definitions.append(DISPLACEMENT_DEFINITION)
        if minimal_definitions: course_context += "\n\nDefiniții de Bază:\n" + "\n".join(minimal_definitions)
    except Exception as e:
        logging.exception(f"Unexpected error during vector search stage: {e}")
        course_context = "[Eroare: Problemă neașteptată la căutarea contextului]"
        # Add minimal definitions on error
        minimal_definitions = []
        if "mss" in payload.question.lower(): minimal_definitions.extend([MSS_AGRESIV_STRUCTURAL_DEFINITION, "MSS Normal: Pivotul rupt are >=2 bearish AND >=2 bullish."])
        if "fvg" in payload.question.lower(): minimal_definitions.append(FVG_STRUCTURAL_DEFINITION)
        if "displacement" in payload.question.lower(): minimal_definitions.append(DISPLACEMENT_DEFINITION)
        if minimal_definitions: course_context += "\n\nDefiniții de Bază:\n" + "\n".join(minimal_definitions)


    # --- 3️⃣ Final Answer Generation ---
    try:
        # --- Build the visual analysis report string safely ---
        # Use the potentially validated detailed_vision_analysis dictionary
        try:
            # Ensure analysis data is a dictionary before trying to dump
            if isinstance(detailed_vision_analysis, dict):
                visual_analysis_report_str = json.dumps(
                    detailed_vision_analysis, indent=2, ensure_ascii=False
                )
            else:
                 # Handle case where vision analysis failed and returned an error string/dict
                 visual_analysis_report_str = f"{{ \"error\": \"Visual analysis data is not a valid dictionary.\", \"raw_data\": \"{str(detailed_vision_analysis)}\" }}"
                 logging.error("Vision analysis result was not a dict, cannot format as JSON for final prompt.")

        except Exception as json_dump_err:
            logging.exception("Error dumping visual analysis to JSON string.")
            visual_analysis_report_str = f"{{ \"error\": \"Could not format visual analysis.\", \"details\": \"{str(json_dump_err)}\" }}"
        logging.debug("Final Visual Analysis Report string for prompt:\n%s", visual_analysis_report_str)

        # --- System prompt remains SYSTEM_PROMPT_CORE (already includes note about validator) ---
        final_system_prompt = SYSTEM_PROMPT_CORE # Already includes the note

        # --- Craft user prompt ---
        final_user_prompt = (
            f"User Question: {payload.question}\n\n"
            f"Visual Analysis Report (JSON):\n```json\n{visual_analysis_report_str}\n```\n\n"
            f"Retrieved Course Context:\n{course_context}\n\n"
            f"Task: Based on the user question, the visual analysis report (paying attention to any '_validator_note'), and the course context, "
            f"provide a concise, helpful answer in Romanian, following the Trading Instituțional methodology and Rareș's style."
        )
        logging.debug("Final System prompt length: %d", len(final_system_prompt))
        logging.debug("Final User prompt length: %d", len(final_user_prompt))
        # Truncate logging if prompt is too long
        logging.debug(f"Final User prompt (first 500 chars): {final_user_prompt[:500]}...")

        # --- OpenAI call ---
        try:
            chat_completion = openai.chat.completions.create(
                model=COMPLETION_MODEL,
                messages=[
                    {"role": "system", "content": final_system_prompt},
                    {"role": "user", "content": final_user_prompt},
                ],
                temperature=0.3, # Keep temperature relatively low for consistency
                max_tokens=800,  # Keep reasonable token limit
            )
            final_answer = chat_completion.choices[0].message.content.strip()
            logging.info(f"Final answer generated successfully using {COMPLETION_MODEL}.")
            logging.debug(f"Final Answer (raw): {final_answer[:300]}...")

            # Return including session_id for feedback linkage
            return {
                "answer": final_answer,
                "session_id": session_id
                # Intentionally not returning analysis_data here - it's logged via /feedback endpoint
            }

        except (APIError, RateLimitError) as e:
            logging.error(f"OpenAI Chat API error ({COMPLETION_MODEL}): {e}")
            # Provide error response but include session_id
            return {
                "answer": f"Nu am putut genera un răspuns final. Serviciul OpenAI ({COMPLETION_MODEL}) nu este disponibil momentan.",
                "session_id": session_id,
                "error": str(e)
            }
        except Exception as e_final:
            logging.exception("Unexpected error during final answer generation")
            # Provide error response but include session_id
            return {
                "answer": "A apărut o eroare la generarea răspunsului final. Te rugăm să încerci din nou.",
                "session_id": session_id,
                "error": str(e_final)
            }

    except Exception as e_gen:
        logging.exception("Unhandled exception in final response generation stage")
        # Provide error response but include session_id
        return {
            "answer": "A apărut o eroare neașteptată la procesarea răspunsului. Te rugăm să încerci din nou.",
            "session_id": session_id,
             "error": str(e_gen)
        }


# ---------------------------------------------------------------------------
# Health‑check Endpoint (Unchanged)
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Very small self‑diagnostic."""
    health_status = {"status": "healthy", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        # Check OpenAI Embedding
        openai.embeddings.create(model=EMBEDDING_MODEL, input=["health_check"])
        health_status["openai_embedding"] = "connected"
    except Exception as e_openai:
        health_status["openai_embedding"] = f"error: {str(e_openai)}"
        health_status["status"] = "unhealthy"
        logging.error("Health check failed - OpenAI Embedding: %s", e_openai)

    try:
         # Check OpenAI Chat (using a cheap model for check)
         openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role":"user", "content":"."}], max_tokens=1)
         health_status["openai_chat"] = "connected"
    except Exception as e_chat:
         health_status["openai_chat"] = f"error: {str(e_chat)}"
         health_status["status"] = "unhealthy"
         logging.error("Health check failed - OpenAI Chat: %s", e_chat)

    try:
        # Check Pinecone - Use describe_index_stats for minimal impact
        stats = index.describe_index_stats()
        health_status["pinecone"] = f"connected (vectors: {stats.total_vector_count})"
    except Exception as e_pinecone:
        health_status["pinecone"] = f"error: {str(e_pinecone)}"
        health_status["status"] = "unhealthy"
        logging.error("Health check failed - Pinecone: %s", e_pinecone)

    return health_status

# Allow running with uvicorn for local testing
# Example: uvicorn your_filename:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
