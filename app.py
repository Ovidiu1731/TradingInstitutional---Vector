import os
import json
from io import BytesIO
from typing import Dict

import pytesseract
import requests
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel

"""
app.py – FastAPI backend for the Trading Instituțional Discord bot
-----------------------------------------------------------------
• /ask                 → text‑only questions answered from course material
• /ask-image-hybrid    → text + chart screenshot (vision, OCR, vector search)

Major revisions 2025‑04‑30:
✓ Robust `summarize_vision_data()` (flat or nested JSON)
✓ System prompt rules enforce: MSS ≠ indicator, BOS ≠ entry‑gate, no backend talk
✓ JSON block fed to the model but never referenced in the reply
"""

# ---------------------------------------------------------------------------
# ENV & GLOBALS
# ---------------------------------------------------------------------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trading-lessons")

if not (OPENAI_API_KEY and PINECONE_API_KEY):
    raise ValueError("Missing OpenAI or Pinecone API key(s)")

# Core system prompt (Rareș’s tone, loaded from file if it exists)
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT_CORE = f.read().strip()
except FileNotFoundError:
    SYSTEM_PROMPT_CORE = (
        "You are an AI assistant trained by Rareș for the Trading Instituțional community."
    )

openai = OpenAI(api_key=OPENAI_API_KEY)
pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index(PINECONE_INDEX_NAME)

# ---------------------------------------------------------------------------
# FASTAPI BOILERPLATE
# ---------------------------------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Discord gateway performs OPTIONS‑preflight
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def extract_text_from_image(image_url: str) -> str:
    """OCR helper that returns ASCII‑cleaned text or an empty string."""
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
    """Return concise Romanian bullet points that *never* invert the booleans."""
    try:
        data = json.loads(raw_json)

        def _flag(key: str) -> bool:
            nested = data.get("presence", {}) if isinstance(data.get("presence"), dict) else {}
            return bool(nested.get(key)) or bool(data.get(key))

        bullets = [
            "✅ MSS este prezent" if _flag("MSS") else "❌ MSS nu este prezent",
            "✅ Imbalance este prezent" if _flag("imbalance") else "❌ Imbalance nu este prezent",
            "✅ BOS este prezent" if _flag("BOS") else "❌ BOS nu este prezent",
        ]

        liquidity_present = (
            _flag("liquidity")
            or data.get("zones", {}).get("demand_zone", {}).get("visible")
            or data.get("zones", {}).get("supply_zone", {}).get("visible")
        )
        bullets.append(
            "✅ Lichiditate este vizibilă" if liquidity_present else "❌ Nu se observă lichiditate"
        )

        return "\n".join(bullets)

    except Exception as err:
        print(f"❌ Summary parsing error: {err}")
        return "Nu s‑au putut interpreta corect datele vizuale."

# ---------------------------------------------------------------------------
# ROUTES – TEXT ONLY
# ---------------------------------------------------------------------------

@app.post("/ask")
async def ask_question(request: Request) -> Dict[str, str]:
    body = await request.json()
    question = body.get("question") or body.get("query") or ""
    if not question:
        return {"answer": "Întrebarea este goală."}

    try:
        emb = openai.embeddings.create(model="text-embedding-ada-002", input=[question]).data[0].embedding
        results = index.query(vector=emb, top_k=6, include_metadata=True)
        context = "\n\n".join(m["metadata"].get("text", "") for m in results.get("matches", [])).strip()

        if not context:
            return {"answer": "Nu sunt sigur pe baza materialului disponibil."}

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT_CORE},
            {"role": "user", "content": f"{question}\n\nContext:\n{context}"},
        ]
        resp = openai.chat.completions.create(model="gpt-3.5-turbo", messages=msgs, temperature=0.4)
        return {"answer": resp.choices[0].message.content.strip()}

    except Exception as err:
        print(f"❌ /ask error: {err}")
        return {"answer": "A apărut o eroare internă."}

# ---------------------------------------------------------------------------
# ROUTES – IMAGE HYBRID
# ---------------------------------------------------------------------------

class ImageHybridQuery(BaseModel):
    question: str
    image_url: str


@app.post("/ask-image-hybrid")
async def ask_image_hybrid(payload: ImageHybridQuery) -> Dict[str, str]:
    """Merge chart vision, OCR, vector search and produce a terse Rareș‑style answer."""
    # 1️⃣ GPT‑4 Vision: parse the chart
    try:
        vision = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a chart parser trained to assist the Trading Instituțional program. "
                        "Describe ONLY what is clearly labeled or visually present. "
                        "Do NOT infer external indicators like LuxAlgo, RSI, etc. "
                        "Output simple JSON. No explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": payload.image_url}},
                        {"type": "text", "text": (
                            "Extract: timeframe (TF), any indicators, presence of MSS, BOS, "
                            "imbalance, and visible zone types. Output JSON only."
                        )},
                    ],
                },
            ],
            max_tokens=300,
        )

        raw_json = vision.choices[0].message.content.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json.removeprefix("```json").removesuffix("```").strip()
        elif raw_json.startswith("```"):
            raw_json = raw_json.removeprefix("```").removesuffix("```").strip()

        vision_dict = json.loads(raw_json)
        vision_json = json.dumps(vision_dict, ensure_ascii=False)
        vision_summary = summarize_vision_data(vision_json)
        json_block = f"```json\n{vision_json}\n```"  # sent to model only
        ocr_text = extract_text_from_image(payload.image_url)

    except Exception as err:
        print(f"❌ Vision parsing error: {err}")
        vision_summary = "Datele vizuale nu au putut fi interpretate."
        json_block = ""
        ocr_text = ""

    # 2️⃣ Pinecone vector search using compressed query
    full_query = f"Întrebare: {payload.question}\n\n{vision_summary}\n\nOCR:\n{ocr_text}"
    try:
        emb = openai.embeddings.create(model="text-embedding-ada-002", input=[full_query]).data[0].embedding
        results = index.query(vector=emb, top_k=6, include_metadata=True)
        course_context = "\n\n".join(m["metadata"].get("text", "") for m in results.get("matches", [])).strip()
    except Exception as err:
        print(f"❌ Pinecone error: {err}")
        return {"answer": "A apărut o eroare la căutarea în materialele cursului."}

    # 3️⃣ Final GPT‑4‑turbo response
    try:
        system_prompt = (
            SYSTEM_PROMPT_CORE
            + "\n\nReguli suplimentare:\n"
            + "- MSS NU este indicator; este schimbare de structură necesară înainte de intrare.\n"
            + "- BOS NU este criteriu de intrare; e doar confirmare post‑trade. Lipsa BOS nu invalidează setup‑ul.\n"
            + "- Nu menționa procesul intern, JSON, cod, API sau backend.\n"
            + "- Confirmă/infimă elementele cu ✅/❌ apoi maximum 30 de cuvinte explicație.\n"
            + "- Evită sfaturi generice (risk management, mindset etc.) dacă nu sunt cerute explicit."
        )

        user_msg = (
            f"{payload.question}\n\nDate vizuale:\n{json_block}\n\n"
            f"Text OCR:\n{ocr_text}\n\nContext curs:\n{course_context}"
        )

        resp = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        return {"answer": resp.choices[0].message.content.strip()}

    except Exception as err:
        print(f"❌ GPT‑4 final response error: {err}")
        return {"answer": "A apărut o eroare la generarea răspunsului final."}
