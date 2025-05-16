import os
import json
import re
import time
import tiktoken
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trading-lessons")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1-aws")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Chunking strategy (same as in your main embedding script)
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

# Function to process a single lesson
def process_lesson(summary_path, transcript_path, chapter_num, lesson_num, lesson_title):
    combined_text = ""
    
    # Load summary content
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_content = f.read().strip()
                if summary_content:
                    combined_text += f"Rezumat:\n{summary_content}\n\n"
        except Exception as e:
            print(f"Error reading summary file {summary_path}: {e}")
    
    # Load transcript content
    if os.path.exists(transcript_path):
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_content = f.read().strip()
                if transcript_content:
                    combined_text += f"Trascriere:\n{transcript_content}"
        except Exception as e:
            print(f"Error reading transcript file {transcript_path}: {e}")
    
    if not combined_text:
        print("No content found in either summary or transcript file!")
        return 0
    
    # Create lesson identifier
    lesson_id = f"capitol_{chapter_num}_lectia_{str(lesson_num).zfill(2)}"
    
    # Split content into chunks
    chunks = split_text(combined_text)
    print(f"Split content into {len(chunks)} chunks")
    
    # Create vectors for each chunk
    vectors = []
    for i, chunk in enumerate(chunks):
        try:
            # Create embedding
            embedding_resp = openai_client.embeddings.create(
                input=chunk,
                model="text-embedding-ada-002"
            )
            embedding = embedding_resp.data[0].embedding
            
            # Create vector object
            vector_id = f"{lesson_id}_{i}"
            vector = {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "lesson_id": lesson_id,
                    "chapter": int(chapter_num),  # Convert to integer
                    "lesson_number": int(lesson_num),  # Convert to integer
                    "title": lesson_title,
                    "chunk_index": i
                }
            }
            vectors.append(vector)
        except Exception as e:
            print(f"Error creating embedding for chunk {i}: {e}")
    
    # Upload vectors in batches
    if vectors:
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)
            print(f"Uploaded batch {i//batch_size + 1}/{(len(vectors)//batch_size) + 1}")
    
    return len(vectors)

def main():
    # Get lesson details from user
    print("=== Upload Single Lesson ===")
    chapter_num = input("Enter chapter number: ")
    lesson_num = input("Enter lesson number: ")
    lesson_title = input("Enter lesson title: ")
    
    # Build file paths
    chapter_dir = f"Capitolul {chapter_num}"
    summary_filename = input(f"Enter summary filename in {chapter_dir}/: ")
    transcript_filename = input(f"Enter transcript filename in {chapter_dir}/: ")
    
    summary_path = os.path.join(chapter_dir, summary_filename)
    transcript_path = os.path.join(chapter_dir, transcript_filename)
    
    # Verify files exist
    files_exist = True
    if not os.path.exists(summary_path):
        print(f"Warning: Summary file not found: {summary_path}")
        files_exist = False
    if not os.path.exists(transcript_path):
        print(f"Warning: Transcript file not found: {transcript_path}")
        files_exist = False
    
    if not files_exist:
        proceed = input("Some files were not found. Continue anyway? (y/n): ")
        if proceed.lower() != 'y':
            print("Aborted.")
            return
    
    # Process the lesson
    start_time = time.time()
    vectors_added = process_lesson(summary_path, transcript_path, chapter_num, lesson_num, lesson_title)
    elapsed_time = time.time() - start_time
    
    print(f"\n=== Upload Complete ===")
    print(f"Added {vectors_added} vectors for lesson {lesson_num} in chapter {chapter_num}")
    print(f"Time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()