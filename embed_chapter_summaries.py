import os
import json
import re
import time
import tiktoken
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables (assuming you have a .env file)
from dotenv import load_dotenv
load_dotenv()

# Initialize clients
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trading-lessons")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1-aws")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# ---- Use your improved chunking strategy ----
def split_text(text, max_tokens=500):
    # Find sections in the text
    sections = re.split(r'---|\n#{1,3} ', text)
    
    # Process each section
    chunks = []
    current_chunk = []
    current_tokens = 0
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        section_tokens = len(enc.encode(section))
        
        # If this section would exceed max_tokens, start a new chunk
        if current_tokens + section_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0
            
        # If section itself is too large, split it by paragraphs
        if section_tokens > max_tokens:
            paragraphs = section.split("\n\n")
            for para in paragraphs:
                para_tokens = len(enc.encode(para))
                if para_tokens > max_tokens:
                    # If paragraph is too large, split by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    for sent in sentences:
                        chunks.append(sent.strip())
                else:
                    if current_tokens + para_tokens > max_tokens and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_tokens = 0
                    current_chunk.append(para)
                    current_tokens += para_tokens
        else:
            current_chunk.append(section)
            current_tokens += section_tokens
            
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

# Process each chapter summary file
def process_chapter_summary(file_path, chapter_num):
    print(f"Processing {file_path}...")
    try:
        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        # Create a unique prefix for IDs from this file
        prefix = f"chapter_{chapter_num}_summary"
        
        # Split into chunks
        chunks = split_text(content)
        print(f"  Split into {len(chunks)} chunks")
        
        # Create vectors
        vectors = []
        for i, chunk in enumerate(chunks):
            # Create embeddings
            embedding_resp = openai_client.embeddings.create(
                input=chunk,
                model="text-embedding-ada-002"
            )
            embedding = embedding_resp.data[0].embedding
            
            # Create vector object
            vector_id = f"{prefix}_{i}"
            vector = {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "source": os.path.basename(file_path),
                    "chapter": chapter_num,
                    "chunk_type": "chapter_summary",
                    "chunk_index": i
                }
            }
            vectors.append(vector)
            
        # Upload vectors to Pinecone in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)
            print(f"  Uploaded batch {i//batch_size + 1}/{(len(vectors)//batch_size) + 1}")
        
        return len(vectors)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

# Main function
def main():
    # Scan for chapter summary files in each chapter directory
    base_dir = os.getcwd()
    total_vectors = 0
    
    # Process each chapter folder
    for chapter_num in range(1, 12):  # Assuming chapters 1-11
        chapter_dir = os.path.join(base_dir, f"Capitolul {chapter_num}")
        if not os.path.exists(chapter_dir):
            continue
            
        # Look for the summary file
        summary_file = None
        for filename in os.listdir(chapter_dir):
            if filename.startswith("Summary Capitolul"):
                summary_file = os.path.join(chapter_dir, filename)
                break
                
        if summary_file:
            vectors_added = process_chapter_summary(summary_file, chapter_num)
            total_vectors += vectors_added
            print(f"Added {vectors_added} vectors from Chapter {chapter_num} summary")
        else:
            print(f"No summary file found for Chapter {chapter_num}")
    
    print(f"\nTotal vectors added: {total_vectors}")
    print("Embedding completed!")

if __name__ == "__main__":
    main()