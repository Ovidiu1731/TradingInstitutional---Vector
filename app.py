# code
import os
import re
import json
import logging
from io import BytesIO
from typing import Dict, Any, Optional # Added Optional

import requests
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
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

# Define the core structural definition here for injection workaround
MSS_AGRESIV_STRUCTURAL_DEFINITION = "Definiție Structurală MSS Agresiv: Este o rupere de structură formată dintr-o singură lumânare care face low/high."

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
# HELPERS
# ---------------------------------------------------------------------------
# Keep extract_text_from_image and extract_json_from_text as they were in the previous version

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

# ---------------------------------------------------------------------------
# ROUTES – TEXT ONLY
# ---------------------------------------------------------------------------
# (Keep the /ask endpoint exactly as it was in your original code)
@app.post("/ask", response_model=Dict[str, str])
async def ask_question(request: Request) -> Dict[str, str]:
    """Handles text-only questions answered strictly from course material."""
    try:
        body = await request.json()
        question = body.get("question", "").strip()
        if not question:
            logging.warning("Received empty question in /ask request.")
            return {"answer": "Te rog să specifici o întrebare."}

        logging.info(f"Received /ask request. Question: '{question[:100]}...'")
        question_lower = question.lower() # Check lowercase once
        is_mss_agresiv_text_q = question_lower == "ce este un mss agresiv"

        # 1. Get Embedding
        try:
            emb_response = openai.embeddings.create(model="text-embedding-ada-002", input=[question])
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
                if is_mss_agresiv_text_q:
                    logging.info("Specific question 'ce este un mss agresiv' detected, providing hardcoded definition as context was empty.")
                    # Provide definition and advice directly
                    return {"answer": MSS_AGRESIV_STRUCTURAL_DEFINITION.replace("Definiție Structurală MSS Agresiv: ", "") + " Dacă ești la început, este recomandat în program să nu-l folosești încă."}
                return {"answer": "Nu am găsit informații relevante în materialele de curs pentru a răspunde la această întrebare."}
            # Inject structural definition if question is exactly "ce este un mss agresiv" and context might be mixed/missing it
            if is_mss_agresiv_text_q and MSS_AGRESIV_STRUCTURAL_DEFINITION.lower() not in context.lower():
                logging.info("Injecting core MSS Agresiv structural definition into context for specific text query as it seems missing.")
                context = f"{MSS_AGRESIV_STRUCTURAL_DEFINITION}\n\n---\n\n{context}" # Prepend

        except PineconeException as e:
            logging.error(f"Pinecone query error: {e}")
            if is_mss_agresiv_text_q: # Still provide definition if query failed for this specific Q
                logging.info("Pinecone failed for 'ce este un mss agresiv', providing hardcoded definition.")
                return {"answer": MSS_AGRESIV_STRUCTURAL_DEFINITION.replace("Definiție Structurală MSS Agresiv: ", "") + " Dacă ești la început, este recomandat în program să nu-l folosești încă."}
            raise HTTPException(status_code=503, detail="Serviciul de căutare (Pinecone) nu este disponibil momentan.")
        except Exception as e:
            logging.exception(f"Unexpected error during Pinecone query: {e}") # Use logging.exception
            if is_mss_agresiv_text_q: # Still provide definition if query failed for this specific Q
                logging.info("Unexpected error during Pinecone query for 'ce este un mss agresiv', providing hardcoded definition.")
                return {"answer": MSS_AGRESIV_STRUCTURAL_DEFINITION.replace("Definiție Structurală MSS Agresiv: ", "") + " Dacă ești la început, este recomandat în program să nu-l folosești încă."}
            raise HTTPException(status_code=500, detail="A apărut o eroare la căutarea informațiilor.")

        # 3. Generate Answer
        try:
            system_message = SYSTEM_PROMPT_CORE + "\n\nAnswer ONLY based on the provided Context."
            # For the specific question, ensure the user message emphasizes the core definition first
            if is_mss_agresiv_text_q:
                # We prepend the definition to the context itself now, so just pass the modified context
                user_message = f"Question: {question}\n\nContext:\n{context}"
                # Add specific advice to the end if not already in answer? Or rely on context containing it.
                # Let's rely on the system prompt / context having the advice.
            else:
                user_message = f"Question: {question}\n\nContext:\n{context}"

            logging.debug(f"Sending to GPT-3.5. System: {system_message[:200]}... User: {user_message[:200]}...")
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
                temperature=0.3,
                max_tokens=300
            )
            answer = response.choices[0].message.content.strip()

            # Ensure the specific advice for MSS Agresiv is included if it's that question
            if is_mss_agresiv_text_q and "dacă ești la început" not in answer.lower():
                answer += " Dacă ești la început, este recomandat în program să nu-l folosești încă."

            logging.info("Successfully generated answer using GPT-3.5-turbo.")
            logging.debug(f"Generated Answer (raw): {answer[:200]}...")
            return {"answer": answer}

        except (APIError, RateLimitError) as e:
            logging.error(f"OpenAI Chat API error (GPT-3.5): {e}")
            raise HTTPException(status_code=503, detail="Serviciul OpenAI (Chat) nu este disponibil momentan.")
        except Exception as e:
            logging.exception(f"Unexpected error during GPT-3.5 answer generation: {e}") # Use logging.exception
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

@app.post("/ask-image-hybrid", response_model=Dict[str, str])
async def ask_image_hybrid(payload: ImageHybridQuery) -> Dict[str, str]:
    """Handles questions with chart screenshots, aiming for detailed visual analysis."""
    logging.info(f"Received /ask-image-hybrid request. Question: '{payload.question[:100]}...', Image URL: {payload.image_url}")

    # Initialize variables for storing results from different stages
    detailed_vision_analysis: Dict[str, Any] = {"error": "Vision analysis not performed"} # Store detailed analysis JSON/dict
    ocr_text: str = ""
    course_context: str = ""

    # --- Keywords for logic branching ---
    trade_evaluation_keywords = ["trade", "tranzacție", "tranzactie", "setup", "intrare", "ce parere", "ce părere", "cum arata", "valid", "corect", "evalua", "rezultat"] # Added "rezultat"
    is_trade_evaluation_or_result_q = any(keyword in payload.question.lower() for keyword in trade_evaluation_keywords)
    logging.info(f"Is trade evaluation/result request: {is_trade_evaluation_or_result_q}")
    question_lower = payload.question.lower()
    is_mss_type_question = "mss" in question_lower and ("normal" in question_lower or "agresiv" in question_lower or "agresivă" in question_lower)
    logging.info(f"Is MSS type classification question: {is_mss_type_question}")


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

        # --- Call GPT-4 Vision for DETAILED analysis (including direction and outcome) ---
        try:
            logging.info("Starting DETAILED GPT-4 Vision analysis...")
            # --- UPDATED STAGE 1 PROMPT WITH ENHANCED INSTRUCTIONS ---
            detailed_vision_system_prompt = (
                "You are an expert Trading Instituțional chart analyst. Your task is to meticulously analyze the provided candlestick chart image "
                "and output a structured JSON object containing your detailed observations. Focus ONLY on the visual elements present."
                "\nGuidelines:"
                
                "\n1. **Candlestick Direction Fundamentals FIRST:**"
                "   - BLUE/GREEN candles are BULLISH (price closed HIGHER than it opened)"
                "   - RED candles are BEARISH (price closed LOWER than it opened)"
                "   - Look at the COLORS of the candles in the chart to determine their direction"
                "   - Note: in some charts, hollow candles are bullish and filled candles are bearish"
                
                "\n2. **High vs Low Break Identification:**"
                "   - A break of HIGH occurs when price moves UPWARD through a previous resistance level"
                "   - A break of LOW occurs when price moves DOWNWARD through a previous support level"
                "   - ALWAYS verify if the break is of HIGH or LOW by looking at the DIRECTION of the movement"
                "   - For a break of HIGH: newer candles are ABOVE older candles forming resistance"
                "   - For a break of LOW: newer candles are BELOW older candles forming support"
                
                "\n3. **MSS Analysis & Classification:**"
                "   - MSS Agresiv DEFINITION: Structure break formed by a SINGLE candle making a new high or low"
                "   - MSS Normal DEFINITION: Structure break formed by MULTIPLE candles making a new high or low"
                "   - For EACH potential MSS, count the EXACT number of candles involved in breaking the structure"
                "   - If only ONE candle breaks the structure -> MSS Agresiv"
                "   - If TWO OR MORE candles break the structure -> MSS Normal"
                
                "\n4. **Trade Direction Analysis:** Based on the MSS type and other context:"
                "   - SHORT trades involve breaks of LOW (price expected to move DOWN)"
                "   - LONG trades involve breaks of HIGH (price expected to move UP)"
                "   - Use labeled arrows, zones, and MSS context to confirm direction"
                
                "\n5. **Displacement Analysis:** Ensure consistency with candle direction:"
                "   - BEARISH displacement shows price moving DOWN, creating gaps (should align with SHORT trades)"
                "   - BULLISH displacement shows price moving UP, creating gaps (should align with LONG trades)"
                
                "\n6. **Critical Color Interpretation:**"
                "   - Do NOT rely solely on red/green colored zones for direction"
                "   - FOCUS on the ACTUAL PRICE MOVEMENT direction and structure breaks"
                "   - EXPLICITLY state if candles breaking structure are BULLISH (blue/green) or BEARISH (red)"
                
                "\n7. **MSS Label Analysis:**"
                "   - Look for 'MSS' labels or arrows in the chart"
                "   - Note the direction the MSS arrow points (UP for breaks of high, DOWN for breaks of low)"
                "   - Match the arrow direction with your identified break type"
                
                "\n8. **Final Cross-Check:**"
                "   - After your analysis, VERIFY that:"
                "   - Your identified 'break_type' (high/low) is consistent with the candle direction (bullish/bearish)"
                "   - For break of HIGH: expect predominantly BULLISH candles (blue/green)"
                "   - For break of LOW: expect predominantly BEARISH candles (red)"
                "   - If you see inconsistency, re-evaluate your analysis"
                
                "\n9. **Output Format:** Return a valid JSON with these MANDATORY fields:"
                "   - 'analysis_possible': boolean"
                "   - 'candle_colors': Description of candle colors in the chart (e.g., 'blue/green for bullish, red for bearish')"
                "   - 'break_identification': Must include the SPECIFIC direction ('break of high' or 'break of low')"
                "   - 'candle_direction': Must specify if the breaking candle is 'bullish' or 'bearish'"
                "   - 'mss_analysis': Include 'break_type' ('high' or 'low'), 'breaking_candle_count' (number), and 'structure_description'"
                "   - 'mss_type': Must be either 'agresiv' (1 candle) or 'normal' (2+ candles)"
                "   - 'displacement_analysis': Include 'direction' ('bullish' or 'bearish')"
                "   - 'trade_direction': Should match break direction ('long' for break of high, 'short' for break of low)"
                "   - Additional fields for price movement, liquidity, etc. as appropriate"
            )

            detailed_vision_user_prompt = (
                "Analyze this trading chart image according to the Trading Instituțional methodology. "
                "Pay special attention to candle colors and direction of movement when determining if it's a break of HIGH or break of LOW. "
                "For MSS classification, count the EXACT number of candles involved in breaking structure - ONE candle means MSS Agresiv, "
                "TWO OR MORE candles means MSS Normal. "
                "Ensure your analysis is directionally consistent: breaks of HIGH should involve BULLISH candles and upward movement, "
                "while breaks of LOW should involve BEARISH candles and downward movement. "
                "Provide your complete analysis as a structured JSON object with ALL required fields."
            )

            vision_resp = openai.chat.completions.create(
                model="gpt-4.1", # Using recommended gpt-4.1
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
                max_tokens=1500, # Increased for more detailed analysis including outcome
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            # --- Process the response (expecting JSON directly) ---
            raw_response_content = vision_resp.choices[0].message.content.strip()
            logging.info("Detailed GPT-4 Vision analysis completed.")
            logging.debug(f"Raw Detailed Vision JSON Response: {raw_response_content}")

            try:
                detailed_vision_analysis = json.loads(raw_response_content)
                if not isinstance(detailed_vision_analysis, dict) or 'analysis_possible' not in detailed_vision_analysis:
                     logging.warning("Vision JSON structure might be invalid. Setting error.")
                     detailed_vision_analysis = {"error": "Invalid JSON structure received from vision model", "raw_response": raw_response_content}
                else:
                     logging.info(f"Successfully parsed detailed Vision JSON.")

                     # Check for consistency between fields
                     if detailed_vision_analysis.get("analysis_possible", False):
                         break_identification = detailed_vision_analysis.get("break_identification", "").lower()
                         candle_direction = detailed_vision_analysis.get("candle_direction", "").lower()
                         trade_direction = detailed_vision_analysis.get("trade_direction", "").lower()
                         mss_analysis = detailed_vision_analysis.get("mss_analysis", {})
                         displacement_analysis = detailed_vision_analysis.get("displacement_analysis", {})
                         
                         # Extract key information
                         break_type = mss_analysis.get("break_type", "").lower() if isinstance(mss_analysis, dict) else ""
                         if not break_type and "high" in break_identification:
                             break_type = "high"
                         elif not break_type and "low" in break_identification:
                             break_type = "low"
                         
                         displacement_direction = displacement_analysis.get("direction", "").lower() if isinstance(displacement_analysis, dict) else ""
                         
                         # Log possible inconsistencies
                         if "high" in break_type and "bearish" in candle_direction:
                             logging.warning(f"❌ DIRECTIONAL INCONSISTENCY: Break of HIGH with BEARISH candle")
                             detailed_vision_analysis["direction_inconsistency"] = "high_break_bearish_candle"
                         
                         if "low" in break_type and "bullish" in candle_direction:
                             logging.warning(f"❌ DIRECTIONAL INCONSISTENCY: Break of LOW with BULLISH candle")
                             detailed_vision_analysis["direction_inconsistency"] = "low_break_bullish_candle"
                         
                         if "high" in break_type and trade_direction == "short":
                             logging.warning(f"❌ DIRECTIONAL INCONSISTENCY: Break of HIGH with SHORT trade direction")
                             detailed_vision_analysis["direction_inconsistency"] = "high_break_short_trade"
                         
                         if "low" in break_type and trade_direction == "long":
                             logging.warning(f"❌ DIRECTIONAL INCONSISTENCY: Break of LOW with LONG trade direction")
                             detailed_vision_analysis["direction_inconsistency"] = "low_break_long_trade"
                         
                         if "high" in break_type and "bearish" in displacement_direction:
                             logging.warning(f"❌ DIRECTIONAL INCONSISTENCY: Break of HIGH with BEARISH displacement")
                             detailed_vision_analysis["direction_inconsistency"] = "high_break_bearish_displacement"
                         
                         if "low" in break_type and "bullish" in displacement_direction:
                             logging.warning(f"❌ DIRECTIONAL INCONSISTENCY: Break of LOW with BULLISH displacement")
                             detailed_vision_analysis["direction_inconsistency"] = "low_break_bullish_displacement"
                         
                         # Set flag for inconsistencies
                         if detailed_vision_analysis.get("direction_inconsistency"):
                             logging.warning(f"⚠️ Direction inconsistencies found in vision analysis: {detailed_vision_analysis.get('direction_inconsistency')}")

            except json.JSONDecodeError as json_err:
                 logging.error(f"❌ Failed to decode JSON from Vision response: {json_err}.")
                 fallback_json_string = extract_json_from_text(raw_response_content)
                 if fallback_json_string:
                     try:
                         detailed_vision_analysis = json.loads(fallback_json_string)
                         logging.info("Successfully parsed detailed Vision JSON using fallback extractor.")
                     except json.JSONDecodeError as fallback_err:
                          logging.error(f"❌ Fallback JSON extraction also failed: {fallback_err}. Raw string: '{fallback_json_string}'")
                          detailed_vision_analysis = {"error": "Invalid JSON structure from vision model (fallback failed)", "raw_response": raw_response_content}
                 else:
                      detailed_vision_analysis = {"error": "No valid JSON found in vision response", "raw_response": raw_response_content}

        except (APIError, RateLimitError) as e:
            logging.error(f"OpenAI Vision API error: {e}")
            detailed_vision_analysis = {"error": f"Vision API error: {str(e)}"}
        except Exception as e:
            logging.exception(f"Unexpected error during detailed Vision processing: {e}")
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
        # Add identified direction to improve context retrieval
        if detailed_vision_analysis.get("trade_direction") in ["long", "short"]:
            query_parts.append(f"Trade direction: {detailed_vision_analysis.get('trade_direction')}")
        
        # Add MSS type if available
        if detailed_vision_analysis.get("mss_type") in ["agresiv", "normal"]:
            query_parts.append(f"MSS type: {detailed_vision_analysis.get('mss_type')}")
            
        combo_query = " ".join(query_parts)

        logging.info(f"Constructed embedding query (first 200 chars): {combo_query[:200]}...")
        emb_response = openai.embeddings.create(model="text-embedding-ada-002", input=[combo_query])
        query_embedding = emb_response.data[0].embedding
        logging.info("Generated embedding for combined query.")
        matches = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        retrieved_matches = matches.get("matches", [])
        course_context = "\n\n---\n\n".join(m["metadata"].get("text", "") for m in retrieved_matches if m["metadata"].get("text")).strip()
        logging.info(f"Pinecone query returned {len(retrieved_matches)} matches. Context length: {len(course_context)}")
        logging.debug(f"DEBUG - Retrieved Course Context Content:\n---\n{course_context}\n---")

        # Inject definition workaround (keep for specific MSS Agresiv questions)
        is_mss_agresiv_definition_needed = is_mss_type_question or question_lower == "ce este un mss agresiv"
        if is_mss_agresiv_definition_needed:
            if MSS_AGRESIV_STRUCTURAL_DEFINITION.lower() not in course_context.lower():
                logging.info("Injecting core MSS Agresiv structural definition into context as it seems missing.")
                course_context = f"{MSS_AGRESIV_STRUCTURAL_DEFINITION}\n\n---\n\n{course_context}"
            else:
                logging.debug("Core MSS Agresiv structural definition already found in retrieved context. No injection needed.")

        if not course_context:
            logging.warning("Pinecone query returned no relevant context for the hybrid query.")
            course_context = "[Eroare: Niciun context specific din curs nu a fost găsit pentru această combinație.]"
            if is_mss_agresiv_definition_needed:
                 course_context += f"\n\n---\n\n{MSS_AGRESIV_STRUCTURAL_DEFINITION}"

    except (APIError, RateLimitError) as e:
        logging.error(f"OpenAI Embedding API error during hybrid search: {e}")
        course_context = "[Eroare: Nu s-a putut genera embedding pentru căutare context]"
        if is_mss_agresiv_definition_needed: course_context += f"\n\n---\n\n{MSS_AGRESIV_STRUCTURAL_DEFINITION}"
    except PineconeException as e:
        logging.error(f"Pinecone query error during hybrid search: {e}")
        course_context = "[Eroare: Nu s-a putut căuta în materialele de curs]"
        if is_mss_agresiv_definition_needed: course_context += f"\n\n---\n\n{MSS_AGRESIV_STRUCTURAL_DEFINITION}"
    except Exception as e:
        logging.exception(f"Unexpected error during vector search stage: {e}")
        course_context = "[Eroare: Problemă neașteptată la căutarea contextului]"
        if is_mss_agresiv_definition_needed: course_context += f"\n\n---\n\n{MSS_AGRESIV_STRUCTURAL_DEFINITION}"

    # --- 3️⃣ Final Answer Generation (GPT-4.1 - ENHANCED PROMPT WITH OUTCOME ANALYSIS) ---
    try:
        # --- Prepare the visual analysis report string ---
        visual_analysis_report_str = "[Eroare la formatarea raportului vizual]" # Default error
        try:
             visual_analysis_report_str = json.dumps(detailed_vision_analysis, indent=2, ensure_ascii=False)
        except Exception:
             visual_analysis_report_str = str(detailed_vision_analysis) # Fallback
        logging.debug(f"Detailed Visual Analysis Report string for prompt:\n{visual_analysis_report_str}")


        # --- Define the UPDATED system prompt instructions for final synthesis ---
        final_system_prompt = SYSTEM_PROMPT_CORE + (
            "\n\n--- Additional Instructions for Image Analysis ---\n"
            "1. You are provided with a **Detailed Visual Analysis Report** from the user's chart image.\n"
            "2. You are also given **Relevant Course Material Context** and possibly **OCR text**.\n"
            
            "\n3. **CRITICAL DIRECTIONAL VERIFICATION:** Before answering:"
            "   - Check if the vision analysis shows a break of HIGH or break of LOW"
            "   - Verify if breaking candle(s) are described as BULLISH or BEARISH"
            "   - For breaks of HIGH: candles should generally be BULLISH (blue/green)"
            "   - For breaks of LOW: candles should generally be BEARISH (red)"
            "   - If you detect inconsistency (e.g., break of high with bearish candle), CORRECT it in your response"
            
            "\n4. **MSS Type Determination:**"
            "   - MSS Agresiv: ALWAYS defined by a SINGLE candle breaking structure"
            "   - MSS Normal: ALWAYS defined by MULTIPLE (2+) candles breaking structure"
            "   - Count the candles involved in breaking the structure from the analysis"
            
            "\n5. **When Asked About MSS Classification:**"
            "   - For MSS Agresiv: Emphasize it's formed by ONE candle breaking structure"
            "   - For MSS Normal: Emphasize it's formed by MULTIPLE candles breaking structure"
            "   - Always specify if it's a break of HIGH (upward) or LOW (downward)"
            "   - Always specify if the breaking candle is BULLISH (blue/green) or BEARISH (red)"
            
            "\n6. **Direction Consistency Check:** Be EXTREMELY careful about direction consistency in your analysis:"
            "   - SHORT trades should have breaks of LOW, BEARISH displacement, and price moving DOWN for a win"
            "   - LONG trades should have breaks of HIGH, BULLISH displacement, and price moving UP for a win"
            "   - If the Visual Analysis Report contains a 'direction_inconsistency' flag, be extra cautious and CORRECT the inconsistency"
            
            "\n7. **Handling MSS Type Questions:** When answering about MSS type:"
            "   - Only consider the NUMBER OF CANDLES breaking structure (not their color or direction)"
            "   - ONE candle breaking = MSS Agresiv"
            "   - TWO OR MORE candles breaking = MSS Normal"
            "   - Make sure to also state the correct direction (break of high or low) and candle color (bullish or bearish)"
            
            "\n8. **Critical Corrections for Common Errors:**"
            "   - If you see 'high_break_bearish_candle' flag: Correct by noting that a break of high typically involves bullish candles"
            "   - If you see 'low_break_bullish_candle' flag: Correct by noting that a break of low typically involves bearish candles"
            "   - If you see 'high_break_short_trade' flag: Correct by noting that breaks of high typically align with long trades"
            "   - If you see 'low_break_long_trade' flag: Correct by noting that breaks of low typically align with short trades"
            
            "\n9. **Evaluating Trade Setups:** When asked to evaluate a setup, consider ALL of these factors while maintaining direction consistency:"
            "   - Whether the setup follows the course trading rules (as determined by comparing visual elements to course material)"
            "   - The quality of the MSS structure (normal vs. aggressive, and how well it matches the definition)"
            "   - The quality of the displacement (if visible and analyzed in report)"
            "   - The position relative to liquidity (if visible and analyzed in report)"
            
            "\n10. **Trade Outcome Analysis:** If the image shows price action after entry:"
            "    - For SHORT trades: Price should move DOWN for a win, UP for a loss"
            "    - For LONG trades: Price should move UP for a win, DOWN for a loss"
            "    - Report whether the trade appears to have won (hit TP) or lost (hit SL) according to the visual analysis"
            "    - IMPORTANT: Even if a trade appears to have lost, evaluate if the setup was valid according to course rules"
            
            "\n11. **Always use MSS, not BOS:** Never mention 'BOS'... Use only 'MSS'\n"
            "\n12. Maintain Rareș's direct, helpful, and concise tone\n"
        )

        # --- Construct the final user message ---
        user_message_parts = [
            f"User Question: {payload.question}\n",
            f"--- Detailed Visual Analysis Report (from image scan): ---\n{visual_analysis_report_str}\n",
            f"--- Relevant Course Material Context (Definition possibly injected): ---",
            f"{course_context}\n"
        ]
        if len(ocr_text) > 5:
            user_message_parts.extend([
                f"--- Text from Image (OCR - may contain errors): ---",
                f"{ocr_text}\n"
            ])
        
        # Add special instructions for MSS type questions
        if is_mss_type_question:
            user_message_parts.append(
                f"--- Special Instructions for MSS Type Question ---\n"
                f"This question is specifically about MSS classification. Remember:\n"
                f"1. MSS Agresiv = EXACTLY ONE candle breaking structure\n"
                f"2. MSS Normal = TWO OR MORE candles breaking structure\n"
                f"3. Also confirm if it's a break of HIGH (upward) or LOW (downward)\n"
                f"4. Also note if breaking candle(s) are BULLISH (blue/green) or BEARISH (red)\n"
            )
        
        user_message_parts.append(
            f"--- Task ---\nAnswer the User Question by carefully integrating the Detailed Visual Analysis Report with the Course Material Context, following all instructions in the System Prompt."
            "ENSURE DIRECTION CONSISTENCY in your analysis. If the vision analysis has directional inconsistencies, correct them in your answer."
            "Be especially careful with MSS classification based on candle count, and matching breaking candle color with break direction."
            "Explain your reasoning by linking visual observations to course rules. Be concise and direct."
        )
        user_msg = "\n".join(user_message_parts)

        logging.debug(f"Sending to GPT-4.1. System Prompt (start): {final_system_prompt[:200]}... User Message (start): {user_msg[:300]}...")

        # Determine model and temperature
        model = "gpt-4.1" # Using recommended gpt-4.1
        temp = 0.3 # Keep temperature relatively low for factual application of rules to visuals

        gpt_resp = openai.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": final_system_prompt}, {"role": "user", "content": user_msg}],
            temperature=temp,
            max_tokens=450 # Increased slightly for potentially more detailed explanations
        )

        answer = gpt_resp.choices[0].message.content.strip()
        logging.info("Successfully generated final answer using GPT-4.1.")
        logging.debug(f"Raw GPT-4.1 Answer: {answer}")

        # --- Post-processing (same as before) ---
        answer = re.sub(r"^(Analizând|Pe baza|Conform|Based on)[^.]*\.\s*", "", answer, flags=re.IGNORECASE).strip()
        answer = re.sub(r"\bBOS\b|\bBreak of Structure\b", "MSS", answer, flags=re.IGNORECASE)
        answer = re.sub(r"\n{2,}", "\n", answer).strip()

        # --- Add directional correction if needed ---
        mss_analysis = detailed_vision_analysis.get("mss_analysis", {})
        break_type = mss_analysis.get("break_type", "").lower() if isinstance(mss_analysis, dict) else ""
        if not break_type and "break_identification" in detailed_vision_analysis:
            if "high" in detailed_vision_analysis.get("break_identification", "").lower():
                break_type = "high"
            elif "low" in detailed_vision_analysis.get("break_identification", "").lower():
                break_type = "low"
                
        candle_direction = detailed_vision_analysis.get("candle_direction", "").lower()
        direction_inconsistency = detailed_vision_analysis.get("direction_inconsistency", "")

        # Check for mismatched candle and break directions
        if direction_inconsistency and any(word in answer.lower() for word in ["agresiv", "normal", "mss"]):
            correction = ""
            if "high_break_bearish_candle" in direction_inconsistency:
                correction = "\n\nNotă importantă: În mod normal, o ruptură de HIGH (nivel superior) este realizată de lumânări predominant BULLISH (verzi/albastre), nu bearish."
            elif "low_break_bullish_candle" in direction_inconsistency:
                correction = "\n\nNotă importantă: În mod normal, o ruptură de LOW (nivel inferior) este realizată de lumânări predominant BEARISH (roșii), nu bullish."
            elif "high_break_short_trade" in direction_inconsistency:
                correction = "\n\nNotă importantă: De obicei, o ruptură de HIGH (nivel superior) este asociată cu tranzacții LONG (cumpărare), nu short."
            elif "low_break_long_trade" in direction_inconsistency:
                correction = "\n\nNotă importantă: De obicei, o ruptură de LOW (nivel inferior) este asociată cu tranzacții SHORT (vânzare), nu long."
            
            # Only add correction if it doesn't already exist in the answer
            if correction and not any(c.lower() in answer.lower() for c in ["notă importantă", "corectie", "corecție", "trebuie menționat"]):
                answer += correction
                logging.warning(f"Added directional correction to answer due to detected mismatch: {direction_inconsistency}")

        # --- ENHANCED Fallback Logic with Outcome Handling ---
        if not answer or len(answer) < 20 or "nu pot oferi" in answer.lower() or "nu am informații" in answer.lower():
            if detailed_vision_analysis.get("error"):
                answer = f"Nu am putut analiza imaginea din cauza unei erori ({detailed_vision_analysis.get('error', 'necunoscută')}). Te rog verifică imaginea sau încearcă din nou."
                logging.info(f"Applied fallback answer due to vision error: {answer}")
            else:
                # If user asked about result/outcome
                if any(term in question_lower for term in ["rezultat", "câștigat", "castigat", "pierdut", "outcome", "win", "loss"]):
                    outcome = detailed_vision_analysis.get("trade_outcome", "undetermined")
                    trade_direction = detailed_vision_analysis.get("trade_direction", "undetermined")
                    
                    if outcome == "undetermined":
                        answer = "Nu pot determina cu certitudine rezultatul final al acestei tranzacții din imaginea furnizată. Pentru a evalua corect rezultatul, ar trebui să văd întreaga mișcare a prețului până la atingerea SL sau TP. Pot însă evalua dacă setup-ul respectă regulile din curs, indiferent de rezultatul final."
                    else:
                        # We have a determined outcome, but answer was still rejected - create a better one
                        direction_text = "short (vânzare)" if trade_direction == "short" else "long (cumpărare)" if trade_direction == "long" else trade_direction
                        result = "câștigătoare" if outcome == "win" else "pierzătoare"
                        answer = f"Din analiza vizuală, această tranzacție {direction_text} pare să fie {result}, deoarece prețul a atins {'nivelul de Take Profit' if outcome == 'win' else 'nivelul de Stop Loss'}. "
                        answer += "Totuși, evaluarea unui setup nu depinde doar de rezultat, ci de respectarea regulilor din curs la momentul intrării în piață. Chiar și un trade valid conform regulilor poate fi pierzător din cauza naturii probabilistice a tradingului."
                    logging.info(f"Applied specific fallback for outcome question, using detected outcome: {outcome}")
                # Special handling for MSS type questions
                elif is_mss_type_question:
                    mss_type = detailed_vision_analysis.get("mss_type", "").lower()
                    breaking_candle_count = mss_analysis.get("breaking_candle_count", 0) if isinstance(mss_analysis, dict) else 0
                    
                    if mss_type == "agresiv" or breaking_candle_count == 1:
                        answer = "Acesta este un MSS agresiv, deoarece structura este ruptă de o singură lumânare. " 
                        answer += f"Se poate observa o ruptură de {'HIGH (nivel superior)' if break_type == 'high' else 'LOW (nivel inferior)'} "
                        answer += f"realizată de o lumânare {'bullish (verde/albastră)' if 'bullish' in candle_direction else 'bearish (roșie)'}, "
                        answer += "exact cum este definit MSS agresiv în program."
                    elif mss_type == "normal" or breaking_candle_count > 1:
                        answer = "Acesta este un MSS normal, deoarece structura este ruptă de mai multe lumânări. "
                        answer += f"Se poate observa o ruptură de {'HIGH (nivel superior)' if break_type == 'high' else 'LOW (nivel inferior)'} "
                        answer += f"realizată de {breaking_candle_count} lumânări {'predominant bullish (verzi/albastre)' if 'bullish' in candle_direction else 'predominant bearish (roșii)'}."
                    else:
                        answer = "Nu pot determina cu certitudine dacă este un MSS agresiv sau normal din imaginea furnizată. Pentru o clasificare corectă, ar trebui să pot identifica clar numărul de lumânări care rup structura."
                    
                    logging.info(f"Applied specific fallback for MSS type question, using detected type: {mss_type}, candle count: {breaking_candle_count}")
                else:
                    answer = "Nu am putut genera un răspuns specific bazat pe informațiile disponibile. Ai putea te rog să reformulezi întrebarea?"
                    logging.warning(f"Applying generic fallback as generated answer was short/uninformative. Vision analysis did not report error. Analysis dump: {detailed_vision_analysis}")

        if not answer: # Final check if still empty
            logging.error("Generated answer was empty even after potential fallback.")
            answer = "Nu am putut genera un răspuns specific. Te rog reformulează sau verifică imaginea."

        logging.info(f"Final Answer Prepared: {answer[:200]}...")
        return {"answer": answer}

    except (APIError, RateLimitError) as e:
        logging.error(f"OpenAI Chat API error (GPT-4.1): {e}")
        raise HTTPException(status_code=503, detail="Serviciul OpenAI (Chat) nu este disponibil momentan pentru generarea răspunsului final.")
    except Exception as e:
        logging.exception(f"Unexpected error during final GPT-4.1 answer generation stage: {e}")
        raise HTTPException(status_code=500, detail="A apărut o eroare la generarea răspunsului final.")

# Optional: Add a root endpoint for health checks
@app.get("/", status_code=200)
def health_check():
    return {"status": "ok", "message": "Trading Instituțional AI Assistant is running"}

# Allow running with uvicorn directly (for local testing)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
