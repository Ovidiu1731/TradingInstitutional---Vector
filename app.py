# code
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
# Keep level DEBUG for now
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

# Define the core structural definition here to avoid repetition
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

def extract_text_from_image(image_url: str) -> str:
    """Download an image and return ASCII-cleaned OCR text, or empty string on failure."""
    # ... (function remains the same as before) ...
    try:
        logging.info(f"Attempting OCR for image URL: {image_url}")
        resp = requests.get(image_url, timeout=15) # Increased timeout
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        text = pytesseract.image_to_string(img, lang="eng") # Ensure tesseract + eng lang pack installed
        cleaned_text = "".join(ch for ch in text if ord(ch) < 128).strip()
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
        logging.error(f"❌ OCR failed: Unexpected error processing image {image_url}: {err}")
        return ""


def extract_json_from_text(text: str) -> str:
    """Extract JSON string from text that might contain markdown code blocks or other text."""
    # ... (function remains the same as before) ...
    logging.debug(f"Attempting to extract JSON from text: {text[:200]}...")
    json_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
    match = re.search(json_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        logging.info("JSON extracted from markdown code block.")
        return extracted
    brace_match = re.search(r"(\{[\s\S]*?\})", text)
    if brace_match:
        potential_json = brace_match.group(1).strip()
        if potential_json.startswith("{") and potential_json.endswith("}") and '"' in potential_json:
             logging.info("Potential JSON object found directly in text.")
             return potential_json
    logging.warning("Could not extract valid-looking JSON object from text.")
    return '{"error": "No JSON found in vision response", "MSS": false, "imbalance": false, "liquidity": false}'

# ---------------------------------------------------------------------------
# ROUTES – TEXT ONLY
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=Dict[str, str])
async def ask_question(request: Request) -> Dict[str, str]:
    """Handles text-only questions answered strictly from course material."""
    # ... (route remains the same as before, including DEBUG logging) ...
    try:
        body = await request.json()
        question = body.get("question", "").strip()
        if not question:
            logging.warning("Received empty question in /ask request.")
            return {"answer": "Te rog să specifici o întrebare."}

        logging.info(f"Received /ask request. Question: '{question[:100]}...'")

        # 1. Get Embedding
        try:
            emb_response = openai.embeddings.create(model="text-embedding-ada-002", input=[question])
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
            results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
            matches = results.get("matches", [])
            context = "\n\n---\n\n".join(m["metadata"].get("text", "") for m in matches if m["metadata"].get("text")).strip()
            logging.info(f"Pinecone query returned {len(matches)} matches. Context length: {len(context)}")
            logging.debug(f"DEBUG TXT - Retrieved Course Context Content:\n---\n{context[:1000]}...\n---") # Log first 1000 chars
            if not context:
                logging.warning("Pinecone query returned no relevant context.")
                # Check if it's the specific MSS Agresiv question
                if question.lower() == "ce este un mss agresiv":
                     logging.info("Specific question 'ce este un mss agresiv' detected, providing hardcoded definition.")
                     return {"answer": MSS_AGRESIV_STRUCTURAL_DEFINITION.replace("Definiție Structurală MSS Agresiv: ", "") + " Dacă ești la început, este recomandat în program să nu-l folosești încă."} # Add advice
                return {"answer": "Nu am găsit informații relevante în materialele de curs pentru a răspunde la această întrebare."}
            # Inject structural definition if question is exactly "ce este un mss agresiv" and context might be mixed
            if question.lower() == "ce este un mss agresiv" and MSS_AGRESIV_STRUCTURAL_DEFINITION not in context:
                 logging.info("Injecting core MSS Agresiv structural definition into context for specific text query.")
                 context = f"{MSS_AGRESIV_STRUCTURAL_DEFINITION}\n\n---\n\n{context}"

        except PineconeException as e:
            logging.error(f"Pinecone query error: {e}")
            raise HTTPException(status_code=503, detail="Serviciul de căutare (Pinecone) nu este disponibil momentan.")
        except Exception as e:
            logging.error(f"Unexpected error during Pinecone query: {e}")
            raise HTTPException(status_code=500, detail="A apărut o eroare la căutarea informațiilor.")

        # 3. Generate Answer
        try:
            # Use the potentially modified context
            system_message = SYSTEM_PROMPT_CORE + "\n\nAnswer ONLY based on the provided Context."
             # Add the hardcoded definition directly for the specific question to ensure it's used
            if question.lower() == "ce este un mss agresiv":
                 user_message = f"Question: {question}\n\nContext:\n{MSS_AGRESIV_STRUCTURAL_DEFINITION.replace('Definiție Structurală MSS Agresiv: ', '')}\n\n---\n\n{context}" # Prioritize hardcoded def
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
        raise
    except Exception as e:
        logging.exception(f"Unhandled exception in /ask endpoint: {e}")
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

    vision_dict: Dict[str, Any] = {"error": "Vision analysis not performed", "MSS": False, "imbalance": False, "liquidity": False}
    ocr_text: str = ""
    course_context: str = ""
    visual_keywords: str = ""

    trade_evaluation_keywords = ["trade", "tranzacție", "tranzactie", "setup", "intrare", "ce parere", "ce părere", "cum arata", "valid", "corect", "evalua"]
    is_trade_evaluation = any(keyword in payload.question.lower() for keyword in trade_evaluation_keywords)
    logging.info(f"Is trade evaluation request: {is_trade_evaluation}")
    question_lower = payload.question.lower() # For checks later
    is_mss_agresiv_question = "mss" in question_lower and ("agresiv" in question_lower or "agresivă" in question_lower)

    # --- 1️⃣ Vision Analysis & OCR ---
    try:
        # ... (Image URL check remains the same) ...
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

        # ... (Vision Call remains the same) ...
        try:
            logging.info("Starting GPT-4 Vision analysis...")
            vision_system_prompt = "..." # Same as before
            vision_user_prompt = "..." # Same as before
            vision_resp = openai.chat.completions.create(...) # Same call as before
            raw_response_content = vision_resp.choices[0].message.content.strip()
            logging.info("GPT-4 Vision analysis completed.")
            logging.debug(f"Raw Vision Response: {raw_response_content}")
            json_string = extract_json_from_text(raw_response_content)
            try:
                vision_dict = json.loads(json_string)
                expected_keys = {"MSS", "imbalance", "liquidity"}
                if not all(key in vision_dict and isinstance(vision_dict[key], bool) for key in expected_keys):
                    logging.warning(f"Vision JSON has unexpected structure: {vision_dict}. Attempting to use anyway.")
                    vision_dict = {k: vision_dict.get(k, False) for k in expected_keys}
                logging.info(f"Successfully parsed Vision JSON: {vision_dict}")
                visual_keywords = " ".join([k for k, v in vision_dict.items() if isinstance(v, bool) and v])
                logging.debug(f"Visual keywords extracted: '{visual_keywords}'")
            except json.JSONDecodeError as json_err:
                 logging.error(f"❌ Failed to decode JSON from Vision response: {json_err}. Raw string: '{json_string}'")
                 vision_dict = {"error": "Invalid JSON structure from vision model", "MSS": False, "imbalance": False, "liquidity": False}
        except (APIError, RateLimitError) as e:
            logging.error(f"OpenAI Vision API error: {e}")
            vision_dict = {"error": "Vision API unavailable", "MSS": False, "imbalance": False, "liquidity": False}
        except Exception as e:
            logging.error(f"Unexpected error during Vision processing: {e}")
            vision_dict = {"error": "Unexpected vision processing error", "MSS": False, "imbalance": False, "liquidity": False}

        ocr_text = extract_text_from_image(payload.image_url)

    except HTTPException:
         raise
    except Exception as e:
        logging.exception(f"Unhandled exception during Vision/OCR stage: {e}")

    # --- 2️⃣ Vector Search ---
    try:
        query_parts = [f"Question: {payload.question}"]
        if visual_keywords: query_parts.append(f"Key Visual Elements Identified: {visual_keywords}")
        if len(ocr_text) > 10: query_parts.append(f"OCR Text Snippet: {ocr_text[:200]}")
        combo_query = " ".join(query_parts)
        logging.info(f"Constructed embedding query (first 200 chars): {combo_query[:200]}...")
        emb_response = openai.embeddings.create(model="text-embedding-ada-002", input=[combo_query])
        query_embedding = emb_response.data[0].embedding
        logging.info("Generated embedding for combined query.")
        matches = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        retrieved_matches = matches.get("matches", [])
        course_context = "\n\n---\n\n".join(m["metadata"].get("text", "") for m in retrieved_matches if m["metadata"].get("text")).strip()
        logging.info(f"Pinecone query returned {len(retrieved_matches)} matches. Context length: {len(course_context)}")
        logging.debug(f"DEBUG - Retrieved Course Context Content:\n---\n{course_context}\n---") # Keep logging context

        # ***** START: INJECT MSS AGRESIV DEFINITION WORKAROUND *****
        if is_mss_agresiv_question:
            # Check if the precise structural definition is already likely present
            if MSS_AGRESIV_STRUCTURAL_DEFINITION.lower() not in course_context.lower():
                 logging.info("Injecting core MSS Agresiv structural definition into context as it seems missing.")
                 # Prepend the definition, clearly marking it. Add extra newlines for separation.
                 course_context = f"{MSS_AGRESIV_STRUCTURAL_DEFINITION}\n\n---\n\n{course_context}"
            else:
                 logging.debug("Core MSS Agresiv structural definition already found in retrieved context. No injection needed.")
        # ***** END: INJECT MSS AGRESIV DEFINITION WORKAROUND *****


        if not course_context:
             logging.warning("Pinecone query returned no relevant context for the hybrid query.")
             # If context is empty, provide error message, but still add definition if relevant
             course_context = "[Eroare: Niciun context specific din curs nu a fost găsit pentru această combinație.]"
             if is_mss_agresiv_question:
                 course_context += f"\n\n---\n\n{MSS_AGRESIV_STRUCTURAL_DEFINITION}"


    except (APIError, RateLimitError) as e:
        logging.error(f"OpenAI Embedding API error during hybrid search: {e}")
        course_context = "[Eroare: Nu s-a putut genera embedding pentru căutare context]"
        if is_mss_agresiv_question: course_context += f"\n\n---\n\n{MSS_AGRESIV_STRUCTURAL_DEFINITION}" # Add definition even on error
    except PineconeException as e:
        logging.error(f"Pinecone query error during hybrid search: {e}")
        course_context = "[Eroare: Nu s-a putut căuta în materialele de curs]"
        if is_mss_agresiv_question: course_context += f"\n\n---\n\n{MSS_AGRESIV_STRUCTURAL_DEFINITION}" # Add definition even on error
    except Exception as e:
        logging.exception(f"Unexpected error during vector search stage: {e}")
        course_context = "[Eroare: Problemă neașteptată la căutarea contextului]"
        if is_mss_agresiv_question: course_context += f"\n\n---\n\n{MSS_AGRESIV_STRUCTURAL_DEFINITION}" # Add definition even on error


    # --- 3️⃣ Final Answer Generation (GPT-4) ---
    try:
        visual_evidence_parts = []
        if vision_dict.get("error"):
            visual_evidence_parts.append(f"Analiza vizuală nu a putut fi completată ({vision_dict['error']}).")
        else:
            visual_evidence_parts.append(f"MSS: {'prezent' if vision_dict.get('MSS') else 'NU este prezent'}")
            visual_evidence_parts.append(f"Imbalance/FVG: {'prezent' if vision_dict.get('imbalance') else 'NU este prezent'}")
            visual_evidence_parts.append(f"Lichiditate: {'vizibilă' if vision_dict.get('liquidity') else 'NU este vizibilă'}")
        visual_evidence_str = ". ".join(visual_evidence_parts)
        logging.debug(f"Visual evidence string for prompt: {visual_evidence_str}")

        # Use the refined system prompt focused on context handling
        final_system_prompt = SYSTEM_PROMPT_CORE + (
            "\n\n--- Additional Instructions for Image Analysis ---\n"
             # ... (Points 1-9 remain the same as the previous refined version) ...
             "1. You are provided with **Visual Analysis Results**...\n"
             "2. You are also given **OCR text**... and relevant **Course Material Context**...\n"
             "3. Answer the User's Question concisely...\n"
             "4. **Prioritize the Visual Analysis Results**... Note that the basic visual analysis typically confirms only *presence*...\n"
             "5. Refer to **Course Material** for definitions, rules, and concepts.\n"
             "6. **Handling MSS Type Questions ('Normal' vs 'Agresiv'):** If the user asks to differentiate...\n"
             "    - Explicitly state that the basic visual analysis confirmed MSS *presence* but cannot determine the *type*.\n"
             "    - Consult the **Course Material Context** provided...\n"
             "    - **VERY IMPORTANT for Identification:** ...**focus primarily on the CORE STRUCTURAL DEFINITION** (e.g., the 'single-candle making low/high' rule for 'MSS Agresiv'). \n"
             "    - Explain *this structural rule* clearly.\n"
             "    - If the Course Material *also* mentions conditions for *using* an MSS Agresiv...mention these **briefly and separately**, explicitly stating they are conditions for *usage/application*...\n"
             "    - Apply the structural rule: If you can reasonably infer the type based on the structural rule and image context, state it. \n"
             "    - **If uncertain**...**state the structural rule clearly** and tell the user **what structural feature they should look for**...\n"
             "    - **CRITICAL:** Do **NOT** invent justifications or wrongly apply usage conditions...\n"
             "7. **Crucially: NEVER mention 'BOS'... Use only 'MSS'...\n"
             "8. If asked for an opinion on a trade/setup...provide a direct evaluation...\n"
             "9. Maintain Rareș's direct, helpful, and concise tone...\n"
        )

        # Construct user message using the potentially modified context
        user_message_parts = [
            f"User Question: {payload.question}\n",
            f"--- Visual Analysis Results (from image scan): ---\n{visual_evidence_str}\n",
            f"--- Relevant Course Material Context (Definition possibly injected): ---", # Note context source
            f"{course_context}\n" # Use the potentially modified context
        ]
        if len(ocr_text) > 5:
             user_message_parts.extend([
                 f"--- Text from Image (OCR - may contain errors): ---",
                 f"{ocr_text}\n"
             ])
        user_message_parts.append(
            f"--- Task ---\nAnswer the User Question based *primarily* on the Visual Analysis Results and Course Material Context provided. Follow all instructions in the System Prompt, especially regarding MSS types and trade evaluations. Be concise and direct."
        )
        user_msg = "\n".join(user_message_parts)

        logging.debug(f"Sending to GPT-4. System Prompt (start): {final_system_prompt[:200]}... User Message (start): {user_msg[:300]}...")

        model = "gpt-4-turbo"
        temp = 0.5 if is_trade_evaluation else 0.3

        gpt_resp = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=temp,
            max_tokens=300
        )

        answer = gpt_resp.choices[0].message.content.strip()
        logging.info("Successfully generated final answer using GPT-4.")
        logging.debug(f"Raw GPT-4 Answer: {answer}")

        # ... (Post-Filtering remains the same) ...
        answer = re.sub(r"^(Analizând|Pe baza|Conform|Based on)[^.]*\.\s*", "", answer, flags=re.IGNORECASE).strip()
        answer = re.sub(r"\bBOS\b|\bBreak of Structure\b", "MSS", answer, flags=re.IGNORECASE)
        answer = re.sub(r"\n{2,}", "\n", answer).strip()

        # ... (Fallback Logic remains the same) ...
        if is_trade_evaluation and (not answer or len(answer) < 30 or "nu pot oferi" in answer.lower() or "nu am informații" in answer.lower()):
             # ... fallback logic ...
             logging.info(f"Applied fallback answer: {answer}")

        if not answer:
             logging.error("Generated answer was empty even after potential fallback.")
             answer = "Nu am putut genera un răspuns specific. Te rog reformulează sau verifică imaginea."

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
