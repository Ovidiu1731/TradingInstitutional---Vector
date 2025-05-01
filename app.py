import os
import re
import json
import logging
from io import BytesIO
from typing import Dict, Any

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# SDK clients
try:
    openai = OpenAI(api_key=OPENAI_API_KEY)
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY) # Renamed to avoid conflict
    index = pinecone_client.Index(PINECONE_INDEX_NAME)
    logging.info(f"Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")
    # Optional: Check index stats
    # index_stats = index.describe_index_stats()
    # logging.info(f"Pinecone Index Stats: {index_stats}")
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

def extract_text_from_image(image_url: str) -> str:
    """Download an image and return ASCII-cleaned OCR text, or empty string on failure."""
    try:
        logging.info(f"Attempting OCR for image URL: {image_url}")
        resp = requests.get(image_url, timeout=15) # Increased timeout
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        # Consider adding image preprocessing here if needed (grayscale, thresholding)
        text = pytesseract.image_to_string(img, lang="eng") # Ensure tesseract + eng lang pack installed
        cleaned_text = "".join(ch for ch in text if ord(ch) < 128).strip()
        logging.info(f"OCR successful. Extracted text length: {len(cleaned_text)}")
        # Log first 100 chars for context
        logging.debug(f"OCR Text (first 100 chars): {cleaned_text[:100]}")
        return cleaned_text
    except requests.exceptions.RequestException as err:
        logging.error(f"❌ OCR failed: Network error accessing image URL {image_url}: {err}")
        return ""
    except pytesseract.TesseractNotFoundError:
        logging.error("❌ OCR failed: pytesseract executable not found. Ensure it's installed and in PATH.")
        return ""
    except Exception as err:
        logging.error(f"❌ OCR failed: Unexpected error processing image {image_url}: {err}")
        return ""


def extract_json_from_text(text: str) -> str:
    """Extract JSON string from text that might contain markdown code blocks or other text."""
    logging.debug(f"Attempting to extract JSON from text: {text[:200]}...") # Log beginning of text
    # Try to find JSON inside markdown code blocks first
    json_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
    match = re.search(json_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        logging.info("JSON extracted from markdown code block.")
        return extracted

    # If no code blocks, try to find anything that looks like a valid JSON object
    # Be more careful to avoid grabbing random {}
    brace_match = re.search(r"(\{[\s\S]*?\})", text)
    if brace_match:
        potential_json = brace_match.group(1).strip()
        # Basic validation: does it look like JSON?
        if potential_json.startswith("{") and potential_json.endswith("}") and '"' in potential_json:
             logging.info("Potential JSON object found directly in text.")
             return potential_json

    logging.warning("Could not extract valid-looking JSON object from text.")
    # Return default structure as a string
    return '{"error": "No JSON found in vision response", "MSS": false, "imbalance": false, "liquidity": false}'

# REMOVED summarize_vision_data and _flag functions - logic integrated into route

# ---------------------------------------------------------------------------
# ROUTES – TEXT ONLY
# ---------------------------------------------------------------------------

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

        # 1. Get Embedding for the question
        try:
            emb_response = openai.embeddings.create(
                model="text-embedding-ada-002", input=[question]
            )
            query_embedding = emb_response.data[0].embedding
            logging.info("Successfully generated embedding for the question.")
        except (APIError, RateLimitError) as e:
            logging.error(f"OpenAI Embedding API error: {e}")
            raise HTTPException(status_code=503, detail="Serviciul OpenAI (Embeddings) nu este disponibil momentan.")
        except Exception as e:
            logging.error(f"Unexpected error during embedding generation: {e}")
            raise HTTPException(status_code=500, detail="A apărut o eroare la procesarea întrebării.")

        # 2. Query Pinecone
        try:
            results = index.query(vector=query_embedding, top_k=5, include_metadata=True) # Reduced top_k slightly
            matches = results.get("matches", [])
            context = "\n\n---\n\n".join(
                m["metadata"].get("text", "") for m in matches if m["metadata"].get("text")
            ).strip()
            logging.info(f"Pinecone query returned {len(matches)} matches. Context length: {len(context)}")
            if not context:
                logging.warning("Pinecone query returned no relevant context.")
                return {"answer": "Nu am găsit informații relevante în materialele de curs pentru a răspunde la această întrebare."}
        except PineconeException as e:
            logging.error(f"Pinecone query error: {e}")
            raise HTTPException(status_code=503, detail="Serviciul de căutare (Pinecone) nu este disponibil momentan.")
        except Exception as e:
            logging.error(f"Unexpected error during Pinecone query: {e}")
            raise HTTPException(status_code=500, detail="A apărut o eroare la căutarea informațiilor.")

        # 3. Generate Answer with GPT-3.5-turbo
        try:
            system_message = SYSTEM_PROMPT_CORE + "\n\nAnswer ONLY based on the provided Context."
            user_message = f"Question: {question}\n\nContext:\n{context}"

            logging.debug(f"Sending to GPT-3.5. System: {system_message[:200]}... User: {user_message[:200]}...")

            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3, # Slightly lower temp for fact-based retrieval
                max_tokens=300
            )
            answer = response.choices[0].message.content.strip()
            logging.info("Successfully generated answer using GPT-3.5-turbo.")
            logging.debug(f"Generated Answer (raw): {answer[:200]}...")
            return {"answer": answer}

        except (APIError, RateLimitError) as e:
            logging.error(f"OpenAI Chat API error (GPT-3.5): {e}")
            raise HTTPException(status_code=503, detail="Serviciul OpenAI (Chat) nu este disponibil momentan.")
        except Exception as e:
            logging.error(f"Unexpected error during GPT-3.5 answer generation: {e}")
            raise HTTPException(status_code=500, detail="A apărut o eroare la generarea răspunsului.")

    except HTTPException:
        # Re-raise HTTPExceptions to let FastAPI handle them
        raise
    except Exception as e:
        # Catch-all for any other unexpected errors during the request handling
        logging.exception(f"Unhandled exception in /ask endpoint: {e}") # Use logging.exception to include traceback
        raise HTTPException(status_code=500, detail="A apărut o eroare internă neașteptată.")


# ---------------------------------------------------------------------------
# ROUTES – IMAGE HYBRID
# ---------------------------------------------------------------------------

class ImageHybridQuery(BaseModel):
    question: str
    image_url: str

@app.post("/ask-image-hybrid", response_model=Dict[str, str])
async def ask_image_hybrid(payload: ImageHybridQuery) -> Dict[str, str]:
    """Handles questions with chart screenshots, using Vision, OCR, and course context."""
    logging.info(f"Received /ask-image-hybrid request. Question: '{payload.question[:100]}...', Image URL: {payload.image_url}")

    # Initialize variables
    vision_dict: Dict[str, Any] = {"error": "Vision analysis not performed", "MSS": False, "imbalance": False, "liquidity": False}
    ocr_text: str = ""
    course_context: str = ""
    visual_keywords: str = "" # For embedding query

    # Determine if it's a trade evaluation request (influences prompt & temp later)
    trade_evaluation_keywords = ["trade", "tranzacție", "tranzactie", "setup", "intrare", "ce parere", "ce părere", "cum arata", "valid", "corect", "evalua"]
    is_trade_evaluation = any(keyword in payload.question.lower() for keyword in trade_evaluation_keywords)
    logging.info(f"Is trade evaluation request: {is_trade_evaluation}")

    # --- 1️⃣ Vision Analysis & OCR ---
    try:
        # Verify image URL accessibility first
        try:
            logging.debug(f"Checking image URL accessibility: {payload.image_url}")
            img_response = requests.head(payload.image_url, timeout=10, allow_redirects=True)
            img_response.raise_for_status()
            # Check content type if possible
            content_type = img_response.headers.get('Content-Type', '').lower()
            if not content_type.startswith('image/'):
                 logging.warning(f"URL {payload.image_url} does not appear to be an image (Content-Type: {content_type}). Proceeding anyway.")
            logging.info("Image URL is accessible.")
        except requests.exceptions.RequestException as img_err:
            logging.error(f"❌ Image URL access error: {img_err}")
            raise HTTPException(status_code=400, detail="Nu am putut accesa imaginea furnizată. Verifică URL-ul.")

        # Call GPT-4 Vision for analysis
        try:
            logging.info("Starting GPT-4 Vision analysis...")
            vision_system_prompt = (
                "You are an expert chart parser specialized in the Trading Instituțional methodology. "
                "Analyze the visual patterns in the provided chart image. Focus ONLY on identifying the presence of: "
                "1. MSS (Market Structure Shift): Look for clear price structure breaks, often marked by labels or horizontal lines. "
                "2. Imbalance/FVG (Fair Value Gap): Identify ANY distinctly colored zones OR highlighted rectangular areas between candles that contrast sharply with the chart background, OR visible price gaps. What matters is COLOR/HIGHLIGHT CONTRAST indicating a specific zone, not the specific color itself. "
                "3. Liquidity: Identify zones marked (e.g., horizontal lines/zones) as potential price targets or areas where price accumulated/reversed. "
                "Output *ONLY* a valid JSON object containing boolean flags for each element: 'MSS', 'imbalance', 'liquidity'. Example: {\"MSS\": true, \"imbalance\": false, \"liquidity\": true}. "
                "Do NOT include explanations, confidence scores, or any other keys in the JSON."
            )
            vision_user_prompt = (
                "Analyze the provided trading chart image based on the Trading Instituțional methodology. "
                "Identify the presence (true/false) of MSS, Imbalance/FVG (any distinct colored/highlighted zone or gap), and Liquidity zones. "
                "Output *only* the JSON object with boolean flags: {\"MSS\": ..., \"imbalance\": ..., \"liquidity\": ...}"
            )

            vision_resp = openai.chat.completions.create(
                model="gpt-4-turbo", # Ensure you are using the latest vision-capable model
                messages=[
                    {"role": "system", "content": vision_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": payload.image_url}},
                            {"type": "text", "text": vision_user_prompt},
                        ],
                    },
                ],
                max_tokens=150, # Reduced tokens as only JSON is expected
                temperature=0.1 # Low temp for factual JSON extraction
            )

            raw_response_content = vision_resp.choices[0].message.content.strip()
            logging.info("GPT-4 Vision analysis completed.")
            logging.debug(f"Raw Vision Response: {raw_response_content}")

            # Extract and validate JSON
            json_string = extract_json_from_text(raw_response_content)
            try:
                vision_dict = json.loads(json_string)
                # Basic validation of expected keys and types
                expected_keys = {"MSS", "imbalance", "liquidity"}
                if not all(key in vision_dict and isinstance(vision_dict[key], bool) for key in expected_keys):
                    logging.warning(f"Vision JSON has unexpected structure: {vision_dict}. Attempting to use anyway.")
                    # Ensure default keys exist even if structure is odd
                    vision_dict = {k: vision_dict.get(k, False) for k in expected_keys}

                logging.info(f"Successfully parsed Vision JSON: {vision_dict}")
                # Create keywords for embedding query
                visual_keywords = " ".join([k for k, v in vision_dict.items() if isinstance(v, bool) and v])
                logging.debug(f"Visual keywords extracted: '{visual_keywords}'")

            except json.JSONDecodeError as json_err:
                logging.error(f"❌ Failed to decode JSON from Vision response: {json_err}. Raw string: '{json_string}'")
                vision_dict = {"error": "Invalid JSON structure from vision model", "MSS": False, "imbalance": False, "liquidity": False}

        except (APIError, RateLimitError) as e:
            logging.error(f"OpenAI Vision API error: {e}")
            # Don't raise HTTPException here, try to proceed without vision data or with defaults
            vision_dict = {"error": "Vision API unavailable", "MSS": False, "imbalance": False, "liquidity": False}
        except Exception as e:
            logging.error(f"Unexpected error during Vision processing: {e}")
            vision_dict = {"error": "Unexpected vision processing error", "MSS": False, "imbalance": False, "liquidity": False}

        # Run OCR (runs even if vision fails, might still be useful)
        ocr_text = extract_text_from_image(payload.image_url)
        # Optional: Basic OCR cleaning
        # ocr_text = re.sub(r'\n+', '\n', ocr_text).strip() # Remove excessive newlines

    except HTTPException:
         raise # Re-raise validation/network errors related to image access
    except Exception as e:
        # Catch-all for stage 1
        logging.exception(f"Unhandled exception during Vision/OCR stage: {e}")
        # Allow proceeding with default/error values for vision_dict and ocr_text

    # --- 2️⃣ Vector Search ---
    try:
        # Strategy 1: Question + Visual Keywords + OCR (Current Implementation)
        # Adjust relevance based on content length/presence
        query_parts = [f"Question: {payload.question}"]
        if visual_keywords: query_parts.append(f"Key Visual Elements: {visual_keywords}")
        if len(ocr_text) > 10: query_parts.append(f"OCR Text Snippet: {ocr_text[:200]}") # Limit OCR length
        combo_query = " ".join(query_parts)

        # Strategy 2: Question + OCR Only (Alternative - uncomment to test)
        # combo_query = f"Question: {payload.question} OCR: {ocr_text}"

        # Strategy 3: Question Only (Alternative - uncomment to test)
        # combo_query = payload.question

        logging.info(f"Constructed embedding query (first 200 chars): {combo_query[:200]}...")

        emb_response = openai.embeddings.create(
            model="text-embedding-ada-002", input=[combo_query]
        )
        query_embedding = emb_response.data[0].embedding
        logging.info("Generated embedding for combined query.")

        matches = index.query(vector=query_embedding, top_k=5, include_metadata=True) # Reduced top_k
        retrieved_matches = matches.get("matches", [])
        course_context = "\n\n---\n\n".join(
             m["metadata"].get("text", "") for m in retrieved_matches if m["metadata"].get("text")
        ).strip()
        logging.info(f"Pinecone query returned {len(retrieved_matches)} matches. Context length: {len(course_context)}")
        if not course_context:
             logging.warning("Pinecone query returned no relevant context for the hybrid query.")
             # Don't return yet, let the final LLM try without specific context

    except (APIError, RateLimitError) as e:
        logging.error(f"OpenAI Embedding API error during hybrid search: {e}")
        # Proceed without context, maybe notify user?
        course_context = "[Eroare: Nu s-a putut genera embedding pentru căutare]"
    except PineconeException as e:
        logging.error(f"Pinecone query error during hybrid search: {e}")
        course_context = "[Eroare: Nu s-a putut căuta în materialele de curs]"
    except Exception as e:
        logging.exception(f"Unexpected error during vector search stage: {e}")
        course_context = "[Eroare: Problemă neașteptată la căutarea contextului]"


    # --- 3️⃣ Final Answer Generation (GPT-4) ---
    try:
        # Construct the visual evidence string for the prompt
        visual_evidence_parts = []
        if vision_dict.get("error"):
            visual_evidence_parts.append(f"Analiza vizuală nu a putut fi completată ({vision_dict['error']}).")
        else:
            if vision_dict.get("MSS"): visual_evidence_parts.append("MSS este prezent")
            else: visual_evidence_parts.append("MSS NU este prezent") # Explicitly state absence
            if vision_dict.get("imbalance"): visual_evidence_parts.append("Imbalance/FVG este prezent")
            else: visual_evidence_parts.append("Imbalance/FVG NU este prezent")
            if vision_dict.get("liquidity"): visual_evidence_parts.append("Lichiditate este vizibilă")
            else: visual_evidence_parts.append("Lichiditate NU este vizibilă")

        visual_evidence_str = ". ".join(visual_evidence_parts)
        logging.debug(f"Visual evidence string for prompt: {visual_evidence_str}")

        # Refine system prompt with clearer instructions
        final_system_prompt = SYSTEM_PROMPT_CORE + (
            "\n\n--- Additional Instructions for Image Analysis ---\n"
            "1. You are provided with Visual Analysis Results derived directly from the user's chart image. **Treat these results as the ground truth** for what is visible in the image.\n"
            "2. You are also given OCR text (potentially noisy) and relevant Course Material context.\n"
            "3. Answer the User's Question concisely (2-3 sentences max). Synthesize information from the Visual Analysis, Course Material, and the Question.\n"
            "4. **Prioritize the Visual Analysis Results** when confirming the presence/absence of elements like MSS, Imbalance, or Liquidity in the *specific chart provided*.\n"
            "5. Refer to Course Material for definitions and rules, but confirm specifics based on the Visual Analysis.\n"
            "6. **Crucially: NEVER mention 'BOS' or 'Break of Structure'.** Use only 'MSS' (Market Structure Shift) as per Trading Instituțional rules.\n"
            "7. If asked for an opinion on a trade/setup ('ce parere', 'e corect?', etc.), provide a direct evaluation based on the Visual Analysis and Course rules. Do NOT refuse.\n"
            "8. Maintain Rareș's direct, helpful, and concise tone. Avoid filler phrases like 'Based on the analysis...'. Be direct."
            # Removed overly strict sentence/word count for now, focusing on concise instruction
        )

        # Construct user message clearly separating inputs
        user_message_parts = [
            f"User Question: {payload.question}\n",
            f"--- Visual Analysis Results (from the image): ---\n{visual_evidence_str}\n",
            f"--- Course Material Context: ---",
            f"{course_context if course_context else 'N/A'}\n"
        ]
        if len(ocr_text) > 5: # Only include OCR if it's non-trivial
             user_message_parts.extend([
                 f"--- Text from Image (OCR - may contain errors): ---",
                 f"{ocr_text}\n"
             ])
        user_message_parts.append(
            f"--- Task ---\nAnswer the User Question based *primarily* on the Visual Analysis Results and Course Material Context provided. Be concise and direct (2-3 sentences max)."
        )
        user_msg = "\n".join(user_message_parts)

        logging.debug(f"Sending to GPT-4. System Prompt (start): {final_system_prompt[:200]}... User Message (start): {user_msg[:300]}...")

        model = "gpt-4-turbo" # Use the best model for this complex task
        temp = 0.5 if is_trade_evaluation else 0.3 # Slightly higher temp for evaluation/opinion

        gpt_resp = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=temp,
            max_tokens=250 # Allow slightly more room than text-only
        )

        answer = gpt_resp.choices[0].message.content.strip()
        logging.info("Successfully generated final answer using GPT-4.")
        logging.debug(f"Raw GPT-4 Answer: {answer}")

        # --- Minimal Post-Filtering ---
        # Attempt to remove generic intros if prompt fails
        answer = re.sub(r"^(Analizând|Pe baza|Conform|Based on)[^.]*\.\s*", "", answer, flags=re.IGNORECASE).strip()
        # Remove BOS mentions if they slip through (should be rare with strong prompt)
        answer = re.sub(r"\bBOS\b|\bBreak of Structure\b", "MSS", answer, flags=re.IGNORECASE)
        # Basic cleanup
        answer = re.sub(r"\n{2,}", "\n", answer).strip()

        # --- Fallback for Trade Evaluations ---
        # If it's an evaluation, and the answer is still too short/generic after filtering
        if is_trade_evaluation and (not answer or len(answer) < 30 or "nu pot oferi" in answer.lower() or "nu am informații" in answer.lower()):
            logging.warning("GPT-4 answer for trade evaluation was too short or generic. Applying fallback.")
            # Base fallback on the *validated* vision_dict
            mss_ok = vision_dict.get("MSS", False)
            imb_ok = vision_dict.get("imbalance", False)
            liq_ok = vision_dict.get("liquidity", False) # Use if needed

            if mss_ok and imb_ok:
                answer = "Confirm că în chart se văd MSS și Imbalance/FVG conform analizei vizuale. Din punct de vedere tehnic și al regulilor Trading Instituțional, setup-ul pare corect."
            elif mss_ok and not imb_ok:
                answer = "Confirm prezența MSS conform analizei vizuale, dar Imbalance/FVG nu este clar vizibil sau lipsește. Verifică dacă acesta este prezent conform regulilor înainte de a considera intrarea."
            elif not mss_ok:
                answer = "Conform analizei vizuale, MSS (Market Structure Shift) esențial pentru intrare nu este (încă) prezent în acest chart. Așteaptă confirmarea MSS conform regulilor."
            else: # Default fallback if vision analysis had errors or didn't find key elements
                 answer = "Analiza vizuală nu a confirmat clar elementele cheie (MSS, Imbalance). Asigură-te că respecți toate regulile Trading Instituțional înainte de a intra într-o tranzacție."
            logging.info(f"Applied fallback answer: {answer}")

        logging.info(f"Final Answer Prepared: {answer[:200]}...")
        return {"answer": answer}

    except (APIError, RateLimitError) as e:
        logging.error(f"OpenAI Chat API error (GPT-4): {e}")
        raise HTTPException(status_code=503, detail="Serviciul OpenAI (Chat) nu este disponibil momentan pentru generarea răspunsului final.")
    except Exception as e:
        logging.exception(f"Unexpected error during final GPT-4 answer generation stage: {e}")
        raise HTTPException(status_code=500, detail="A apărut o eroare la generarea răspunsului final.")

# Optional: Add a root endpoint for health checks
@app.get("/", status_code=200)
def health_check():
    return {"status": "ok", "message": "Trading Instituțional AI Assistant is running"}

# Note: Consider adding authentication/authorization middleware for production use.
# Note: Ensure pytesseract is installed and the tesseract executable is in your system's PATH or configured.
