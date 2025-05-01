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


def summarize_vision_data(raw_json: str) -> str:
    """Convert Vision JSON → concise Romanian bullet points (never invert flags)."""
    try:
        data = json.loads(raw_json)

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
    # 1️⃣ Vision parsing
    try:
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
                            "Output JSON with presence flags for each element. Remember that traders use different color schemes - "
                            "focus on areas with distinct color contrast rather than specific colors. Any area highlighted with a "
                            "different color than the main chart is likely an imbalance/FVG even without explicit labels."
                        )},
                    ],
                },
            ],
            max_tokens=300,
        )

        raw_json = vision_resp.choices[0].message.content.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json.removeprefix("```json").removesuffix("```").strip()
        elif raw_json.startswith("```"):
            raw_json = raw_json.removeprefix("```").removesuffix("```").strip()

        vision_dict = json.loads(raw_json)
        vision_json = json.dumps(vision_dict, ensure_ascii=False)
        vision_summary = summarize_vision_data(vision_json)
        json_block = f"```json\n{vision_json}\n```"  # fed to model only
        ocr_text = extract_text_from_image(payload.image_url)

    except Exception as err:
        print(f"❌ Vision parsing error: {err}")
        vision_summary = "Datele vizuale nu au putut fi interpretate."
        json_block = ""
        ocr_text = ""

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
            "- Response format: (1) Describe key observed elements (max 25 words). "
            "(2) Evaluate the trade in max 25 words. Maximum 2 sentences total."
        )

        user_msg = (
            f"{payload.question}\n\nDate vizuale:\n{json_block}\n\n"
            f"Text OCR:\n{ocr_text}\n\nContext curs:\n{course_context}"
        )

        gpt_resp = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=200,
        )

        answer = gpt_resp.choices[0].message.content.strip()

        # 🔻 Post-filter: strip any sentence that contains BOS or a generic intro
        answer = re.sub(r"(?i)(^|\n)\s*Analizând[^.]*\.\s*", "", answer)
        answer = re.sub(r"(?i)(^|\n)[^.]*\\bBOS\\b[^.]*\.\s*", "", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer).strip()

        return {"answer": answer}

    except Exception as err:
        print(f"❌ GPT-4 final response error: {err}")
        return {"answer": "A apărut o eroare la generarea răspunsului final."}
