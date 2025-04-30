import pytesseract
from PIL import Image
import requests
from io import BytesIO
import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
from typing import Dict

# Load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trading-lessons")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing OpenAI or Pinecone API key")

# Load system prompt
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read().strip()
except FileNotFoundError:
    SYSTEM_PROMPT = "You are an AI assistant trained by Rareș for the Trading Instituțional community..."

# Init clients
openai = OpenAI(api_key=OPENAI_API_KEY)
pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index(PINECONE_INDEX_NAME)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_image(image_url: str) -> str:
    try:
        response = requests.get(image_url, timeout=10)
        image = Image.open(BytesIO(response.content))
        # Restrict image size to avoid memory issues
        image.thumbnail((2000, 2000))
        text = pytesseract.image_to_string(image, lang="eng")
        # Sanitize strange unicode characters
        sanitized_text = ''.join(char for char in text if ord(char) < 128)
        return sanitized_text.strip()
    except Exception as e:
        print(f"❌ OCR error: {e}")
        return ""

def summarize_vision_data(vision_json: str) -> str:
    try:
        data = json.loads(vision_json)
        summary = []
        if data.get("MSS", False):
            summary.append("MSS este vizibil")
        else:
            summary.append("MSS nu este prezent")
        if data.get("imbalance", False):
            summary.append("imbalance este prezent")
        else:
            summary.append("nu se observă imbalance")
        if data.get("liquidity") == "taken":
            summary.append("lichiditatea a fost luată")
        elif data.get("liquidity") == "present":
            summary.append("lichiditate marcată, dar nu luată")
        else:
            summary.append("nu se observă lichiditate clară")
        # Add more fields if needed...
        return ". ".join(summary) + "."
    except Exception as e:
        print(f"❌ Summary parsing error: {e}")
        return "Nu s-au putut interpreta corect datele vizuale."

@app.post("/ask", response_class=JSONResponse)
async def ask_question(request: Request) -> Dict[str, str]:
    body = await request.json()
    question = body.get("question") or body.get("query") or ""
    if not question:
        return {"answer": "Întrebarea este goală."}
    try:
        embedding = openai.embeddings.create(model="text-embedding-ada-002", input=[question]).data[0].embedding
        search_result = index.query(vector=embedding, top_k=6, include_metadata=True)
        context_chunks = [match['metadata'].get('text', '') for match in search_result.get('matches', [])]
        context = "\n\n".join(context_chunks).strip()
        if not context:
            return {"answer": "Nu sunt sigur pe baza materialului disponibil."}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{question}\n\nContext:\n{context}"}
        ]
        chat_response = openai.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0.4
        )
        answer = chat_response.choices[0].message.content.strip()
        return {"answer": answer}
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {"answer": "A apărut o eroare internă."}

class ImageHybridQuery(BaseModel):
    question: str
    image_url: str

@app.post("/ask-image-hybrid", response_class=JSONResponse)
async def ask_image_hybrid(payload: ImageHybridQuery) -> Dict[str, str]:
    try:
        vision_response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a chart parser trained to assist the Trading Instituțional program. Describe ONLY what is clearly labeled or visually present. Do NOT infer external indicators like LuxAlgo, RSI, etc. Output simple JSON. No explanations."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": payload.image_url}},
                        {"type": "text", "text": "Extract: timeframe (TF), any indicators, presence of MSS, BOS, imbalance, and visible zone types. Output JSON only."}
                    ]
                }
            ],
            max_tokens=300
        )
        
        # Check if content exists
        raw_content = vision_response.choices[0].message.content
        if not raw_content:
            raise ValueError("Vision model returned empty content.")
            
        try:
            vision_data_raw = raw_content.strip()
            if not vision_data_raw:
               raise ValueError("OpenAI returned empty vision response")
            vision_json = json.loads(vision_data_raw)  # to ensure it's valid JSON
            print("📊 Extracted Vision Data:", json.dumps(vision_json))
            vision_summary = summarize_vision_data(json.dumps(vision_json))  # pass back as string
            print("📄 Vision Summary:", vision_summary)
        except Exception as e:
            print(f"❌ Vision JSON error: {e}")
            vision_summary = "Datele vizuale nu au putut fi interpretate."
            
        ocr_text = extract_text_from_image(payload.image_url)
    except Exception as e:
        print(f"❌ Vision error: {e}")
        return {"answer": "A apărut o eroare la extragerea informației din imagine."}

    combined_query = f"Întrebare: {payload.question}\n\nSumar vizual:\n{vision_summary}\n\nText detectat în imagine (OCR):\n{ocr_text}"

    try:
        embedding = openai.embeddings.create(model="text-embedding-ada-002", input=[combined_query]).data[0].embedding
        search_result = index.query(vector=embedding, top_k=6, include_metadata=True)
        context_chunks = [match['metadata'].get('text', '') for match in search_result.get('matches', [])]
        course_context = "\n\n".join(context_chunks).strip()
    except Exception as e:
        print(f"❌ Pinecone error: {e}")
        return {"answer": "A apărut o eroare la căutarea în materialele cursului."}

    try:
        final_prompt = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant trained by Rareș for the Trading Instituțional community. "
                    "Always answer in Romanian. Your job is to confirm or reject what the user asked based only on what is visible in the image and the course material.\n\n"
                    "Keep your answer concise but clear. If the concept is present, confirm it in 1–2 short sentences using the terms taught (e.g. MSS, imbalance, SLG, TCG, liquidity). "
                    "If it's partially valid or invalid, explain briefly. No generic commentary. Never refer to JSON, backend logic, or anything not in the image.\n\n"
                    "Use Rareș's tone and logic: direct, practical, and based only on visual chart evidence."
                )
            },
            {
                "role": "user",
                "content": f"{payload.question}\n\nSumar vizual:\n{vision_summary}\n\nText detectat:\n{ocr_text}\n\nFragmente din curs:\n{course_context}"
            }
        ]
        final_response = openai.chat.completions.create(
            model="gpt-4-turbo", messages=final_prompt, temperature=0.4, max_tokens=500
        )
        answer = final_response.choices[0].message.content.strip()
        return {"answer": answer}
    except Exception as e:
        print(f"❌ GPT-4 final response error: {e}")
        return {"answer": "A apărut o eroare la generarea răspunsului final."}
