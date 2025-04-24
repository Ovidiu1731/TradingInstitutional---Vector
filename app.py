import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
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
    question = body.get("question", "")

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

        # Extract matched text chunks
        context_chunks = [match['metadata'].get('text', '') for match in search_result.get('matches', [])]
        context = "\n\n".join(context_chunks).strip()

        if not context:
            return {"answer": "Nu sunt sigur pe baza materialului disponibil. Îți recomand să verifici cu mentorul sau să întrebi un membru cu mai multă experiență."}

        # Chat completion
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
        return {"answer": answer}

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {"answer": "A apărut o eroare internă. Încearcă din nou sau contactează administratorul."}
