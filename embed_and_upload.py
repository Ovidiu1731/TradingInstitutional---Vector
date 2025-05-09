import os
import json
import tiktoken
from dotenv import load_dotenv

# New OpenAI SDK import
from openai import OpenAI

# Pinecone SDK import
from pinecone import Pinecone, ServerlessSpec

# 1. Load environment variables
load_dotenv()
OPENAI_KEY    = os.getenv("OPENAI_API_KEY")
PINECONE_KEY  = os.getenv("PINECONE_API_KEY")
PINECONE_ENV  = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME    = os.getenv("PINECONE_INDEX_NAME", "trading-lessons")

# 2. Init clients
client = OpenAI(api_key=OPENAI_KEY)
pc     = Pinecone(api_key=PINECONE_KEY)

# 3. Ensure Pinecone index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
index = pc.Index(INDEX_NAME)

# 4. Load your lessons.json (list of lesson objects)
with open("lessons.json", "r", encoding="utf-8") as f:
    lessons = json.load(f)

# 5. Split text into overlapping chunks to keep headings with their content
def split_text(text, max_tokens=500, overlap=50):
    """
    Break `text` into chunks of up to `max_tokens` tokens, 
    carrying `overlap` words from the end of one chunk to the start of the next.
    """
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    words = text.split()
    chunks = []
    curr = []
    for w in words:
        curr.append(w)
        # once we exceed max_tokens, flush the chunk
        if len(enc.encode(" ".join(curr))) >= max_tokens:
            chunks.append(" ".join(curr))
            # keep the last `overlap` words to overlap into the next chunk
            curr = curr[-overlap:]
    # any remainder
    if curr:
        chunks.append(" ".join(curr))
    return chunks


# 6. Process each lesson
for lesson in lessons:
    lesson_id    = lesson["id"]
    files        = lesson.get("files", {})
    transcript_p = files.get("transcript", "").strip()
    summary_p    = files.get("summary", "").strip()

    # resolve relative → absolute
    transcript_path = os.path.normpath(os.path.join(os.getcwd(), transcript_p))
    summary_path    = os.path.normpath(os.path.join(os.getcwd(), summary_p))

    if not os.path.isfile(transcript_path) or not os.path.isfile(summary_path):
        print(f"⚠️ Skipping {lesson_id} — bad file path.")
        continue

    # read your files
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read().strip()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary    = f.read().strip()

    # combine and chunk
    full_text = transcript + "\n\n" + summary
    chunks = split_text(full_text)

    for i, chunk in enumerate(chunks):
        try:
            resp = client.embeddings.create(
                input=[chunk],
                model="text-embedding-ada-002"
            )
            embedding = resp.data[0].embedding
        except Exception as e:
            print(f"❌ Embedding error {lesson_id} chunk {i}: {e}")
            continue

        # upsert to Pinecone, now including the text itself in metadata
        index.upsert([
            {
                "id":     f"{lesson_id}_chunk_{i}",
                "values": embedding,
                "metadata": {
                    "title":       lesson.get("title", ""),
                    "chapter":     lesson.get("chapter_label", ""),
                    "lesson":      lesson.get("lesson_label", ""),
                    "chunk_index": i,
                    "text":        chunk
                }
            }
        ])

    print(f"✅ Uploaded {lesson_id} ({len(chunks)} chunks)")

print("🎉 All done.")
