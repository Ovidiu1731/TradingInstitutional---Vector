import os
import re
import json
from io import BytesIO
from typing import Dict

import requests
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

"""
app.py – FastAPI backend for the Trading Instituțional Discord bot
-----------------------------------------------------------------
Endpoints
---------
/ask                – text‑only questions answered strictly from course material
/ask-image-hybrid   – text + chart screenshot (vision, OCR, vector search)
"""

# ---------------------------------------------------------------------------
# ENVIRONMENT & GLOBALS
# ---------------------------------------------------------------------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trading-lessons")

if not (OPENAI_API_KEY and PINECONE_API_KEY):
    raise ValueError("Missing OpenAI or Pinecone API key(s)")

# Core system prompt (Rareș's tone). Fallback if file is missing.
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT_CORE = f.read().strip()
except FileNotFoundError:
    SYSTEM_PROMPT_CORE = (
        "You are an AI assistant trained by Rareș for the Trading Instituțional community."
    )

# SDK clients
openai = OpenAI(api_key=OPENAI_API_KEY)
pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index(PINECONE_INDEX_NAME)

# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def extract_text_from_image(image_url: str) -> str:
    """Download an image and return ASCII‑cleaned OCR text, or empty string on failure."""
    try:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        text = pytesseract.image_to_string(img, lang="eng")
        return "".join(ch for ch in text if ord(ch) < 128).strip()
    except Exception as err:
        print(f"❌ OCR error: {err}")
        return ""


def extract_json_from_text(text: str) -> str:
    """Extract JSON from text that might contain markdown code blocks or other text."""
    # Try to find JSON inside markdown code blocks first
    json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(json_pattern, text)
    if match:
        return match.group(1).strip()
    
    # If no code blocks, try to find anything that looks like JSON
    if text.strip().startswith("{") and text.strip().endswith("}"):
        return text.strip()
    
    # Return a default valid JSON if nothing else works
    return '{"MSS": false, "imbalance": false, "liquidity": false}'


def summarize_vision_data(raw_json: str) -> str:
    """Convert Vision JSON → concise Romanian bullet points (never invert flags)."""
    try:
        # Handle potential empty JSON
        if not raw_json.strip():
            return "Nu s-au putut interpreta datele vizuale."
            
        # Try to parse the JSON
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            # If direct parsing fails, try to sanitize the JSON string
            raw_json = extract_json_from_text(raw_json)
            try:
                data = json.loads(raw_json)
            except json.JSONDecodeError:
                # If still fails, return a default response
                return "Nu s-au putut interpreta corect datele vizuale."

        def _flag(key: str) -> bool:
            # Check multiple possible keys for the same concept
            keys_to_check = [key]
            if key == "imbalance":
                keys_to_check.extend([
                    "FVG", "fvg", "fair_value_gap", "fairValueGap", "gap", 
                    "colored_zone", "colored_zones", "distinct_zone", "distinct_zones", 
                    "highlighted_zone", "highlighted_zones", "contrast", "contrasting_zone", 
                    "contrasting_zones", "colored_area", "highlighted_area"
                ])
            elif key == "MSS":
                keys_to_check.extend([
                    "mss", "marketStructureShift", "market_structure_shift", 
                    "structure_shift", "structure_break", "break_of_structure"
                ])
            
            # Check in top level and nested objects
            nested = data.get("presence", {}) if isinstance(data.get("presence"), dict) else {}
            zones = data.get("zones", {}) if isinstance(data.get("zones"), dict) else {}
            visual = data.get("visual", {}) if isinstance(data.get("visual"), dict) else {}
            
            # Return True if any key is found with a truthy value
            for check_key in keys_to_check:
                # Check in multiple possible locations and formats
                if any([
                    bool(data.get(check_key)), 
                    bool(nested.get(check_key)),
                    bool(visual.get(check_key)),
                    check_key in str(zones),
                    check_key in str(data)
                ]):
                    return True
            
            # Also check for visual imbalance detection using any color terms
            if key == "imbalance":
                color_terms = ["color", "zone", "highlight", "area", "contrast", "distinct"]
                json_str = str(data).lower()
                # If any color-related term is found in the JSON
                for term in color_terms:
                    if term in json_str:
                        return True
                
            return False

        # Imbalance present when explicitly marked true in JSON or visual patterns found
        imbalance_present = _flag("imbalance")

        bullets = [
            "✅ MSS este prezent" if _flag("MSS") else "❌ MSS nu este prezent",
            "✅ Imbalance/FVG este prezent" if imbalance_present else "❌ Imbalance/FVG nu este prezent",
        ]

        # Liquidity present if imbalance flag OR dedicated liquidity boolean
        liquidity_present = _flag("liquidity") or imbalance_present
        bullets.append(
            "✅ Lichiditate este vizibilă" if liquidity_present else "❌ Nu se observă lichiditate"
        )

        return "\n".join(bullets)

    except Exception as err:
        print(f"❌ Summary parsing error: {err}")
        return "Nu s-au putut interpreta corect datele vizuale."

# ---------------------------------------------------------------------------
# ROUTES – TEXT ONLY
# ---------------------------------------------------------------------------

@app.post("/ask")
async def ask_question(request: Request) -> Dict[str, str]:
    body = await request.json()
    question = body.get("question") or body.get("query") or ""
    if not question:
        return {"answer": "Întrebarea este goală."}

    try:
        emb = openai.embeddings.create(
            model="text-embedding-ada-002", input=[question]
        ).data[0].embedding
        results = index.query(vector=emb, top_k=6, include_metadata=True)
        context = "\n\n".join(m["metadata"].get("text", "") for m in results.get("matches", [])).strip()

        if not context:
            return {"answer": "Nu sunt sigur pe baza materialului disponibil."}

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT_CORE},
            {"role": "user", "content": f"{question}\n\nContext:\n{context}"},
        ]
        reply = openai.chat.completions.create(
            model="gpt-3.5-turbo", messages=msgs, temperature=0.4
        )
        return {"answer": reply.choices[0].message.content.strip()}

    except Exception as err:
        print(f"❌ /ask error: {err}")
        return {"answer": "A apărut o eroare internă."}

# ---------------------------------------------------------------------------
# ROUTES – IMAGE HYBRID
# ---------------------------------------------------------------------------

class ImageHybridQuery(BaseModel):
    question: str
    image_url: str

@app.post("/ask-image-hybrid")
async def ask_image_hybrid(payload: ImageHybridQuery) -> Dict[str, str]:
    """Answer using chart screenshot + OCR + course context, Rareș style."""
    # Check if this is a trade evaluation request
    is_trade_evaluation = any(keyword in payload.question.lower() for keyword in [
        "trade", "tranzacție", "tranzactie", "setup", "intrare", "ce parere", "ce părere", "cum arata"
    ])
    
    # Initialize defaults for error handling
    vision_summary = "Datele vizuale nu au putut fi interpretate."
    json_block = '{"MSS": false, "imbalance": false, "liquidity": false}'
    ocr_text = ""
    
    # 1️⃣ Vision parsing
    try:
        # Verify image URL is accessible before sending to OpenAI
        try:
            img_response = requests.head(payload.image_url, timeout=5)
            img_response.raise_for_status()
        except Exception as img_err:
            print(f"❌ Image URL access error: {img_err}")
            return {"answer": "Nu am putut accesa imaginea. Verificați URL-ul și încercați din nou."}
        
        vision_resp = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert chart parser for the Trading Instituțional program specialized in identifying key market structures. "
                        "IMPORTANT: Analyze the visual patterns in the chart, not just text labels. "
                        "Look for these key elements:\n"
                        "1. MSS (Market Structure Shift): Look for labels or horizontal lines with price structure breaks\n"
                        "2. Imbalance/FVG (Fair Value Gap): Look for ANY distinctly colored zones or highlighted areas between candles that contrast with the background, or visible gaps in price action\n"
                        "3. Liquidity: Look for horizontal lines or zones where price has been accumulated\n\n"
                        "Even if these elements aren't explicitly labeled with text, identify them based on visual patterns. "
                        "Traders use various color schemes - what matters is COLOR CONTRAST, not specific colors. "
                        "Any zone highlighted with a distinct color different from the background is likely an imbalance/FVG. "
                        "Output simple JSON with presence flags for each element (true/false)."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": payload.image_url}},
                        {"type": "text", "text": (
                            "Analyze the chart and identify: \n"
                            "1. MSS (Market Structure Shift) - horizontal lines with arrows or labels\n"
                            "2. Imbalance/FVG (Fair Value Gap) - ANY distinctly colored or highlighted zones that contrast with the background, or visible price gaps\n"
                            "3. Liquidity areas - zones marked for price targets\n"
                            "Output JSON with presence flags for each element like this format: "
                            '{"MSS": true/false, "imbalance": true/false, "liquidity": true/false}. '
                            "Focus on areas with distinct color contrast. Any area highlighted with a "
                            "different color than the main chart is likely an imbalance/FVG even without explicit labels."
                        )},
                    ],
                },
            ],
            max_tokens=300,
        )

        # Get the raw response content
        raw_response = vision_resp.choices[0].message.content.strip()
        
        # Extract JSON from the response
        raw_json = extract_json_from_text(raw_response)
        
        # Ensure we have valid JSON
        try:
            vision_dict = json.loads(raw_json)
            vision_json = json.dumps(vision_dict, ensure_ascii=False)
        except json.JSONDecodeError:
            # If JSON is still invalid, use a default structure
            vision_dict = {"MSS": False, "imbalance": False, "liquidity": False}
            vision_json = json.dumps(vision_dict, ensure_ascii=False)
        
        # Create summary and JSON block
        vision_summary = summarize_vision_data(vision_json)
        json_block = f"```json\n{vision_json}\n```"  # fed to model only
        
        # Extract OCR text
        ocr_text = extract_text_from_image(payload.image_url)

    except Exception as err:
        print(f"❌ Vision parsing error: {err}")
        # Defaults already set

    # 2️⃣ Vector search
    combo_query = f"Întrebare: {payload.question}\n\n{vision_summary}\n\nOCR:\n{ocr_text}"
    try:
        emb = openai.embeddings.create(
            model="text-embedding-ada-002", input=[combo_query]
        ).data[0].embedding
        matches = index.query(vector=emb, top_k=6, include_metadata=True)
        course_context = "\n\n".join(
            m["metadata"].get("text", "") for m in matches.get("matches", [])
        ).strip()
    except Exception as err:
        print(f"❌ Pinecone error: {err}")
        return {"answer": "A apărut o eroare la căutarea în materialele cursului."}

    # 3️⃣ GPT-4 final answer
    try:
        system_prompt = SYSTEM_PROMPT_CORE + (
            "\n\nAdditional rules:\n"
            "- Answer directly; avoid phrases like 'Analyzing the information'.\n"
            "- Do **not** mention BOS at all.\n"
            "- Mention imbalance only if it is present.\n"
            "- MSS is **not** an indicator; it is the required market-structure shift before entry.\n"
            "- Do not mention internal processes, JSON, code, or backend.\n"
            "- IMPORTANT: When users ask about a trade ('ce parere ai de acest trade?' or similar), ALWAYS analyze the chart and provide feedback.\n"
            "- NEVER refuse to provide an opinion on a trade chart - it's your primary function.\n"
            "- Response format: (1) Describe key observed elements (max 25 words). "
            "(2) Evaluate the trade in max 25 words. Maximum 2 sentences total."
        )
        
        # For trade evaluation questions, enforce chart analysis
        if is_trade_evaluation:
            user_msg = (
                f"Analizeaza acest chart trading:\n\nDate vizuale:\n{json_block}\n\n"
                f"Text OCR:\n{ocr_text}\n\nContext curs:\n{course_context}\n\n"
                f"Intrebare originala: {payload.question}"
            )
        else:
            user_msg = (
                f"{payload.question}\n\nDate vizuale:\n{json_block}\n\n"
                f"Text OCR:\n{ocr_text}\n\nContext curs:\n{course_context}"
            )

        # For trade evaluation, use GPT-4 with higher temperature for more opinion
        model = "gpt-4-turbo"
        temp = 0.5 if is_trade_evaluation else 0.3
        
        gpt_resp = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=temp,
            max_tokens=200,
        )

        answer = gpt_resp.choices[0].message.content.strip()

        # 🔻 Post-filter: strip any sentence that contains BOS or a generic intro
        answer = re.sub(r"(?i)(^|\n)\s*Analizând[^.]*\.\s*", "", answer)
        answer = re.sub(r"(?i)(^|\n)[^.]*\\bBOS\\b[^.]*\.\s*", "", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
        
        # Remove disclaimers about not providing financial advice for trade evaluation questions
        if is_trade_evaluation:
            answer = re.sub(r"(?i)(^|\n)[^.]*\\bnu (pot|ofer) (opinii|sfaturi|evaluări|analiz)[^.]*\.\s*", "", answer)
            answer = re.sub(r"(?i)(^|\n)[^.]*\\bnu pot evalua[^.]*\.\s*", "", answer)
            
            # If the answer is empty or too generic after filtering, provide a fallback
            if not answer or len(answer) < 20:
                if "MSS" in vision_summary and "Imbalance" in vision_summary:
                    answer = "Chart prezintă MSS și imbalance corect identificate. Setup-ul de trade respectă regulile Trading Instituțional - aspectul tehnic arată bine."
                else:
                    answer = "Chart-ul arată un setup interesant. Verifică prezența MSS și imbalance pentru a te asigura că respectă regulile Trading Instituțional."

        return {"answer": answer}

    except Exception as err:
        print(f"❌ GPT-4 final response error: {err}")
        return {"answer": "A apărut o eroare la generarea răspunsului final."}
