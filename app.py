import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

# Load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trading-lessons")

# Load system prompt
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

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

@app.post("/ask")
async def ask_question(request: Request):
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
async def ask_image_hybrid(payload: ImageHybridQuery):
    print("🧠 Hybrid Vision Input:", payload.question, payload.image_url)

    # STEP 1: Visual Feature Extraction from image
    try:
        vision_response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a visual parser. Look at the chart and extract ONLY relevant trading features in JSON format. Do not explain or analyze."},
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
        print("📊 Extracted Vision Data:", vision_json)
    except Exception as e:
        print(f"❌ Vision error:", e)
        return {"answer": f"A apărut o eroare la extragerea informației din imagine: {e}"}

    # STEP 2: Build combined query
    combined_query = f"Întrebare: {payload.question}\n\nContext vizual extras:\n{vision_json}"

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
            {"role": "system", "content": "You are a professional AI assistant trained on Rareș's Trading Instituțional program. Answer in Romanian. Be direct, short, and only use information found in the course excerpts below. Do not invent or generalize."},
            {"role": "user", "content": f"{combined_query}\n\nFragmente din curs:\n{course_context}"}
        ]

        final_response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=final_prompt,
            temperature=0.4,
            max_tokens=500
        )

        answer = final_response.choices[0].message.content.strip()
        print("✅ Final answer:", answer)
        return {"answer": answer}

    except Exception as e:
        print(f"❌ GPT-4 final response error:", e)
        return {"answer": "A apărut o eroare la generarea răspunsului final."}
