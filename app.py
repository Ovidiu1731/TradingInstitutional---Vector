import pytesseract
from PIL import Image
import requests
from io import BytesIO
import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
from typing import Dict, List, Any, Optional

# Load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trading-lessons")

# Check for required environment variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is required")

# Load system prompt
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read().strip()
except FileNotFoundError:
    print("Warning: system_prompt.txt not found, using default prompt")
    SYSTEM_PROMPT = """You are an AI assistant trained by Rareș for the Trading Instituțional community. You reply only in Romanian. Use the same tone and terminology as Rareș. Be concise, confident, and practical. Avoid general trading theory or over-explaining. Base your answer strictly on the course materials and what is clearly visible in the image or context. When something is visible but not perfect, make a judgment call as Rareș would. If a concept is unclear, say so directly and ask for clarification or more context."""

# Init clients
openai = OpenAI(api_key=OPENAI_API_KEY)
pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index(PINECONE_INDEX_NAME)

# Init FastAPI
app = FastAPI()

# Enable CORS if needed (e.g. for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility function for OCR - moved outside the handler function
def extract_text_from_image(image_url: str) -> str:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        text = pytesseract.image_to_string(image, lang="eng")
        return text.strip()
    except Exception as e:
        print(f"❌ OCR error: {e}")
        return ""

@app.post("/ask")
async def ask_question(request: Request) -> Dict[str, str]:
    body = await request.json()
    question = body.get("question") or body.get("query") or ""

    print("📩 Incoming question:", question)

    if not question:
        return {"answer": "Întrebarea este goală."}

    try:
        # Embed the question
        embedding = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=[question]
        ).data[0].embedding

        # Query Pinecone
        search_result = index.query(vector=embedding, top_k=6, include_metadata=True)

        context_chunks = [match['metadata'].get('text', '') for match in search_result.get('matches', [])]
        context = "\n\n".join(context_chunks).strip()

        print("📚 Retrieved context chunks:", len(context_chunks))
        for i, chunk in enumerate(context_chunks):
            print(f"— Chunk {i + 1}: {chunk[:100]}...")

        if not context:
            return {"answer": "Nu sunt sigur pe baza materialului disponibil. Îți recomand să verifici cu mentorul sau să întrebi un membru cu mai multă experiență."}

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{question}\n\nContext:\n{context}"}
        ]

        chat_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.4
        )

        answer = chat_response.choices[0].message.content.strip()
        print("✅ Final answer:", answer)

        return {"answer": answer}

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {"answer": "A apărut o eroare internă. Încearcă din nou sau contactează administratorul."}


# ========= IMAGE + TEXT HYBRID HANDLER ============
class ImageHybridQuery(BaseModel):
    question: str
    image_url: str

@app.post("/ask-image-hybrid")
async def ask_image_hybrid(payload: ImageHybridQuery) -> Dict[str, str]:
    print("🧠 Hybrid Vision Input:", payload.question, payload.image_url)

    # STEP 1: Visual Feature Extraction from image
    try:
        vision_response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a chart parser trained to assist the Trading Instituțional program. Describe ONLY what is clearly labeled or visually present. Do NOT infer external indicators like LuxAlgo, RSI, etc. If something is not clearly marked, ignore it. Output simple JSON with neutral labels. Do not include explanations."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": payload.image_url}},
                        {"type": "text", "text": "Extract: timeframe (TF), any indicators, presence of MSS, BOS, imbalance, and visible zone types (support/resistance, liquidity zones, etc). Output JSON only."}
                    ]
                }
            ],
            max_tokens=300
        )
        vision_json = vision_response.choices[0].message.content.strip()
        ocr_text = extract_text_from_image(payload.image_url)
        print("🧾 OCR Extracted Text:", ocr_text)
        print("📊 Extracted Vision Data:", vision_json)
    except Exception as e:
        print(f"❌ Vision error:", e)
        return {"answer": f"A apărut o eroare la extragerea informației din imagine: {e}"}

    # STEP 2: Build combined query
    combined_query = (
    f"Întrebare: {payload.question}\n\n"
    f"Context vizual extras:\n{vision_json}\n\n"
    f"Text detectat în imagine (OCR):\n{ocr_text}"
    )

    # STEP 3: Embed combined query
    try:
        embedding = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=[combined_query]
        ).data[0].embedding
    except Exception as e:
        print(f"❌ Embedding error:", e)
        return {"answer": "A apărut o eroare la generarea embedding-ului."}

    # STEP 4: Search Pinecone
    try:
        search_result = index.query(vector=embedding, top_k=6, include_metadata=True)
        context_chunks = [match['metadata'].get('text', '') for match in search_result.get('matches', [])]
        course_context = "\n\n".join(context_chunks).strip()
    except Exception as e:
        print(f"❌ Pinecone error:", e)
        return {"answer": "A apărut o eroare la căutarea în materialele cursului."}

   # STEP 5: Final GPT-4 response
try:
    final_prompt = [
        {"role": "system", "content": """You are an AI assistant trained by Rareș for the Trading Instituțional community. Always answer in Romanian. Your job is to confirm or reject what the user asked based only on what is visible in the image and the course material.

Keep your answer concise but clear. If the concept is present, confirm it in 1–2 short sentences, using the terms taught in the program (e.g. MSS, imbalance, SLG, TCG, liquidity). If it's partially valid or invalid, explain why briefly. Do not make generic trading commentary. Never refer to JSON, backend logic, indicators not shown in the picture, or add context that isn't clearly visible.

Always use the tone and logic Rareș uses: direct, practical, and based only on chart evidence."""},

        {"role": "user", "content": f"{combined_query}\n\nFragmente din curs:\n{course_context}"}
    ]

    final_response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=final_prompt,
        temperature=0.4,
        max_tokens=300
    )

    answer = final_response.choices[0].message.content.strip()
    print("✅ Final answer:", answer)
    return {"answer": answer}

except Exception as e:
    print(f"❌ GPT-4 final response error:", e)
    return {"answer": "A apărut o eroare la generarea răspunsului final."}
