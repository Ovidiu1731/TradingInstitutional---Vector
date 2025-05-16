import os
import json
import re
import time
import datetime
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

# Load or create tracking file
TRACKING_FILE = "embedding_tracking.json"
if os.path.exists(TRACKING_FILE):
    with open(TRACKING_FILE, "r", encoding="utf-8") as f:
        tracking_data = json.load(f)
else:
    tracking_data = {
        "last_update": "",
        "processed_files": {},
        "indexed_lessons": []
    }

# Same chunking function as before
# [Insert your split_text function here]

# Function to identify new or updated files
def find_new_content():
    new_files = []
    base_dir = os.getcwd()
    last_update = tracking_data.get("last_update", "")
    last_update_time = datetime.datetime.strptime(last_update, "%Y-%m-%d %H:%M:%S") if last_update else datetime.datetime.min
    
    # Check each chapter directory
    for chapter_num in range(1, 20):  # Assuming up to 20 chapters
        chapter_dir = os.path.join(base_dir, f"Capitolul {chapter_num}")
        if not os.path.exists(chapter_dir):
            continue
        
        # Check all files in this chapter
        for filename in os.listdir(chapter_dir):
            file_path = os.path.join(chapter_dir, filename)
            
            # Skip non-text files
            if not filename.endswith(".txt"):
                continue
                
            # Check if file is new or modified since last update
            file_mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            file_hash = f"{os.path.getsize(file_path)}_{file_mod_time.timestamp()}"
            old_hash = tracking_data["processed_files"].get(file_path, "")
            
            # If file is new or changed
            if file_hash != old_hash:
                # Try to parse lesson details
                lesson_match = re.search(r"Lectia\s+(\d+)\s*[-–]\s*([^-]+)", filename, re.IGNORECASE)
                if lesson_match:
                    lesson_num = lesson_match.group(1)
                    lesson_title = lesson_match.group(2).strip()
                    
                    # Determine file type
                    file_type = "unknown"
                    if "Summary" in filename:
                        file_type = "summary"
                    elif "Transcript" in filename:
                        file_type = "transcript"
                    
                    new_files.append({
                        "path": file_path,
                        "chapter": chapter_num,
                        "lesson": lesson_num,
                        "title": lesson_title,
                        "type": file_type,
                        "hash": file_hash
                    })
    
    return new_files

# Group files by lesson
def group_by_lesson(files):
    lessons = {}
    for file in files:
        key = f"chapter_{file['chapter']}_lesson_{file['lesson']}"
        if key not in lessons:
            lessons[key] = {
                "chapter": file['chapter'],
                "lesson": file['lesson'],
                "title": file['title'],
                "files": {}
            }
        lessons[key]["files"][file["type"]] = file["path"]
        lessons[key]["hashes"] = lessons[key].get("hashes", {})
        lessons[key]["hashes"][file["type"]] = file["hash"]
    
    return list(lessons.values())

# Main update function
def update_index():
    print(f"=== Starting weekly update - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # Find new or changed files
    new_files = find_new_content()
    if not new_files:
        print("No new or updated content found.")
        return
    
    print(f"Found {len(new_files)} new or updated files")
    
    # Group by lesson
    lessons_to_process = group_by_lesson(new_files)
    print(f"Identified {len(lessons_to_process)} lessons to process")
    
    # Process each lesson
    total_vectors = 0
    for lesson in lessons_to_process:
        chapter = lesson["chapter"]
        lesson_num = lesson["lesson"]
        title = lesson["title"]
        
        # Create lesson identifier
        lesson_id = f"capitol_{chapter}_lectia_{str(lesson_num).zfill(2)}"
        
        # Check if this lesson already exists
        lesson_vectors = []
        try:
            # Try to find existing vectors for this lesson
            fetch_response = index.query(
                vector=[0.0] * 1536,  # Dummy vector
                filter={"lesson_id": lesson_id},
                top_k=1,
                include_metadata=True
            )
            
            if fetch_response["matches"]:
                print(f"Lesson {lesson_id} already exists in index. Updating...")
                
                # Delete existing vectors
                delete_result = index.delete(
                    filter={"lesson_id": lesson_id}
                )
                print(f"Deleted existing vectors for {lesson_id}")
        except Exception as e:
            print(f"Error checking for existing lesson: {e}")
        
        # Combine content
        combined_text = ""
        if "summary" in lesson["files"] and os.path.exists(lesson["files"]["summary"]):
            with open(lesson["files"]["summary"], "r", encoding="utf-8") as f:
                summary_content = f.read().strip()
                if summary_content:
                    combined_text += f"Rezumat:\n{summary_content}\n\n"
        
        if "transcript" in lesson["files"] and os.path.exists(lesson["files"]["transcript"]):
            with open(lesson["files"]["transcript"], "r", encoding="utf-8") as f:
                transcript_content = f.read().strip()
                if transcript_content:
                    combined_text += f"Trascriere:\n{transcript_content}"
        
        if not combined_text:
            print(f"No content found for lesson {lesson_id}, skipping")
            continue
        
        # Split content
        chunks = split_text(combined_text)
        print(f"Split lesson {lesson_id} into {len(chunks)} chunks")
        
        # Create vectors
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
                        "chapter": chapter,
                        "lesson_number": lesson_num,
                        "title": title,
                        "chunk_index": i
                    }
                }
                vectors.append(vector)
            except Exception as e:
                print(f"Error creating embedding for chunk {i}: {e}")
        
        # Upload vectors
        if vectors:
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                index.upsert(vectors=batch)
                print(f"Uploaded batch {i//batch_size + 1}/{(len(vectors)//batch_size) + 1}")
        
        # Update tracking
        for file_type, file_path in lesson["files"].items():
            tracking_data["processed_files"][file_path] = lesson["hashes"][file_type]
        
        if lesson_id not in tracking_data["indexed_lessons"]:
            tracking_data["indexed_lessons"].append(lesson_id)
        
        total_vectors += len(vectors)
        print(f"Processed lesson {lesson_id}: {len(vectors)} vectors")
    
    # Update last update timestamp
    tracking_data["last_update"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save tracking data
    with open(TRACKING_FILE, "w", encoding="utf-8") as f:
        json.dump(tracking_data, f, indent=2)
    
    print(f"\n=== Update Complete ===")
    print(f"Added/updated {total_vectors} vectors across {len(lessons_to_process)} lessons")
    print(f"Tracking file updated: {TRACKING_FILE}")

if __name__ == "__main__":
    update_index()