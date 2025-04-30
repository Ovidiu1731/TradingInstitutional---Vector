import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
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

# Init FastAPI app
app = FastAPI()

# Allow CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== TEXT-ONLY /ask ENDPOINT ==========
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

        # Search Pinecone
        search_result = index.query(vector=embedding, top_k=6, include_metadata=True)
        context_chunks = [match['metadata'].get('text', '') for match in search_result.get('matches', [])]
        context = "\n\n".join(context_chunks).strip()

        print("📚 Retrieved context chunks:", len(context_chunks))
        for i, chunk in enumerate(context_chunks):
            print(f"— Chunk {i + 1}: {chunk[:100]}...")

        if not context:
            return {
                "answer": "Nu sunt sigur pe baza materialului disponibil. Îți recomand să verifici cu mentorul sau să întrebi un membru cu mai multă experiență."
            }

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


# ========== IMAGE+TEXT /ask-image ENDPOINT ==========
class ImageQuery(BaseModel):
    question: str
    image_url: str

@app.post("/ask-image")
async def ask_with_image(payload: ImageQuery):
    print("🖼️ Received image-based question:", payload.question)

    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an AI trading assistant that answers in the style of the Trading Instituțional community. You reply only in Romanian. Keep responses short, direct, and clear, based strictly on what is visible in the image and the user’s question. Do not include general trading theory, outside knowledge, or invented examples. If the image doesn’t contain enough information to answer confidently, say so clearly."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": payload.image_url}},
                        {"type": "text", "text": payload.question}
                    ]
                }
            ],
            max_tokens=500
        )

        answer = response.choices[0].message.content.strip()
        print("✅ Vision model answer:", answer)
        return {"answer": answer}

    except Exception as e:
        print(f"❌ Vision ERROR: {e}")
        return {"answer": f"A apărut o eroare la procesarea imaginii: {e}"}
