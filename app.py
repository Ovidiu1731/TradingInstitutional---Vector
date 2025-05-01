import pytesseract
from PIL import Image
import requests
from io import BytesIO
import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
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

try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read().strip()
except FileNotFoundError:
    SYSTEM_PROMPT = "You are an AI assistant trained by Rareș for the Trading Instituțional community..."

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
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        text = pytesseract.image_to_string(image, lang="eng")
        return ''.join(char for char in text if ord(char) < 128).strip()
    except Exception as e:
        print(f"❌ OCR error: {e}")
        return ""

def summarize_vision_data(vision_json: str) -> str:
    try:
        data = json.loads(vision_json)
        summary = []

        # Key structure flags
        summary.append("✅ MSS este prezent" if data.get("presence", {}).get("MSS") else "❌ MSS nu este prezent")
        summary.append("✅ Imbalance este prezent" if data.get("presence", {}).get("imbalance") else "❌ Imbalance nu este prezent")
        summary.append("✅ BOS este prezent" if data.get("presence", {}).get("BOS") else "❌ BOS nu este prezent")

        # Zone visibility
        demand_visible = data.get("zones", {}).get("demand_zone", {}).get("visible", False)
        supply_visible = data.get("zones", {}).get("supply_zone", {}).get("visible", False)
        if demand_visible or supply_visible:
            summary.append("✅ Lichiditate este vizibilă")
        else:
            summary.append("❌ Nu se observă lichiditate")

        return "\n".join(summary)

    except Exception as e:
        print(f"❌ Summary parsing error: {e}")
        return "Nu s-au putut interpreta corect datele vizuale."

@app.post("/ask")
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

@app.post("/ask-image-hybrid")
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
        raw_content = vision_response.choices[0].message.content.strip()
        print("🖼️ Raw vision response content:", raw_content)

        if raw_content.startswith("```json"):
            raw_content = raw_content.removeprefix("```json").removesuffix("```").strip()
        elif raw_content.startswith("```"):
            raw_content = raw_content.removeprefix("```").removesuffix("```").strip()

        vision_json = json.loads(raw_content)
        print("📊 Parsed Vision JSON:", json.dumps(vision_json, indent=2))
        vision_summary = summarize_vision_data(json.dumps(vision_json))
        ocr_text = extract_text_from_image(payload.image_url)
    except Exception as e:
        print(f"❌ Vision JSON error: {e}")
        vision_summary = "Datele vizuale nu au putut fi interpretate."
        ocr_text = ""

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
                    "Always answer in Romanian. Use the same tone, terms, and judgment logic Rareș teaches in the program.\n\n"
                    "Your job is to confirm or reject the setup based strictly on the visual summary, OCR text, and course context.\n\n"
                    "✅ If MSS is prezent, start by confirming it directly, even if imbalance or BOS are absent.\n"
                    "✅ If imbalance is missing, mention it briefly, but don't reject the setup unless it's critical.\n"
                    "✅ If the trade looks valid but was a loss, mention that the logic was still correct.\n\n"
                    "✅If the user asks for validation of a loss, emphasize the quality of the decision, not just the outcome.\n\n"
                    "Keep answers short and specific. Maximum 2 short paragraphs. Avoid generic commentary, and do not explain JSON or backend logic. Use Rareș’s tone: direct, competent, practical."
                    "Avoid generic commentary and do not refer to technical terms like 'JSON' or 'backend'."
                )
            },
            {
                "role": "user",
                "content": f"{payload.question}\n\nSumar vizual:\n{vision_summary}\n\nText detectat:\n{ocr_text}\n\nFragmente din curs:\n{course_context}"
            }
        ]
        final_response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=final_prompt,
            temperature=0.4,
            max_tokens=500
        )
        answer = final_response.choices[0].message.content.strip()
        return {"answer": answer}
    except Exception as e:
        print(f"❌ GPT-4 final response error: {e}")
        return {"answer": "A apărut o eroare la generarea răspunsului final."}
