import os
import json
import tiktoken
import re
from dotenv import load_dotenv
import time

# New OpenAI SDK import
from openai import OpenAI

# Pinecone SDK import
from pinecone import Pinecone, ServerlessSpec

# Function to sanitize IDs (remove special characters, replace spaces with underscores)
def sanitize_id(text):
    # Remove non-ASCII characters
    ascii_text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Replace spaces, dots and other special chars with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', ascii_text)
    # Ensure it's not empty and starts with a letter or number
    if not sanitized:
        return "vector_id"
    return sanitized

# Function to extract chapter and lesson numbers for sorting
def extract_lesson_info(lesson_id):
    # Try to extract chapter and lesson numbers
    chapter_match = re.search(r'Capitolul\s+(\d+)', lesson_id)
    lesson_match = re.search(r'Lectia\s+(\d+)', lesson_id)
    
    chapter_num = int(chapter_match.group(1)) if chapter_match else 999
    lesson_num = int(lesson_match.group(1)) if lesson_match else 999
    
    return (chapter_num, lesson_num, lesson_id)  # Return original id as tie-breaker

def extract_document_structure(lesson_id, text):
    """
    Extract hierarchical structure from a lesson document.
    
    Args:
        lesson_id: The identifier of the lesson
        text: The full content of the lesson
        
    Returns:
        Dict with document metadata and list of sections with paragraphs
    """
    # Extract document-level metadata
    chapter_match = re.search(r'(Capitolul\s+\d+)', lesson_id)
    chapter = chapter_match.group(1) if chapter_match else ""
    
    lesson_match = re.search(r'Lectia\s+(\d+)', lesson_id)
    lesson_num = lesson_match.group(1) if lesson_match else ""
    
    title_match = re.search(r'Lectia\s+\d+\s*[-–]\s*(.*?)(?:\s*-\s*Capitolul|$)', lesson_id)
    title = title_match.group(1).strip() if title_match else ""
    
    # Create document metadata
    doc_metadata = {
        "document_id": f"{chapter.replace(' ', '_').lower()}_lectia_{lesson_num.zfill(2)}",
        "document_title": lesson_id,
        "chapter": chapter,
        "lesson_number": lesson_num,
        "title": title
    }
    
    # First split text into key parts - detect if it contains "Rezumat:" and "Trascriere:"
    rezumat_match = re.search(r'Rezumat:(.*?)(?=Trascriere:|$)', text, re.DOTALL)
    transcriere_match = re.search(r'Trascriere:(.*?)$', text, re.DOTALL)
    
    sections = []
    
    # If there's a summary, add it as a section
    if rezumat_match:
        summary_text = rezumat_match.group(1).strip()
        summary_paragraphs = [p.strip() for p in summary_text.split("\n\n") if p.strip()]
        
        summary_section = {
            "title": "Rezumat",
            "section_type": "summary",
            "paragraphs": []
        }
        
        for idx, paragraph in enumerate(summary_paragraphs):
            summary_section["paragraphs"].append({
                "text": paragraph,
                "paragraph_id": f"summary_{idx}"
            })
            
        sections.append(summary_section)
    
    # If there's a transcript, process it with potential section headings
    if transcriere_match:
        transcript_text = transcriere_match.group(1).strip()
        transcript_paragraphs = [p.strip() for p in transcript_text.split("\n\n") if p.strip()]
        
        current_section = {
            "title": "Introducere",
            "section_type": "transcript",
            "paragraphs": []
        }
        
        # Pattern for potential section headings
        section_heading_pattern = r'^(?:\d+\.\s+)?([A-Z0-9][A-Za-z0-9\s:]+)$'
        paragraph_counter = 0
        
        for paragraph in transcript_paragraphs:
            # Check if this paragraph looks like a section heading
            if re.match(section_heading_pattern, paragraph.strip()):
                # If we've been collecting paragraphs, save the current section
                if current_section["paragraphs"]:
                    sections.append(current_section)
                
                # Start a new section
                current_section = {
                    "title": paragraph.strip(),
                    "section_type": "transcript",
                    "paragraphs": []
                }
            else:
                # Add this paragraph to the current section
                current_section["paragraphs"].append({
                    "text": paragraph,
                    "paragraph_id": f"transcript_{paragraph_counter}"
                })
                paragraph_counter += 1
        
        # Add the last section if it has content
        if current_section["paragraphs"]:
            sections.append(current_section)
    
    # If no specific sections were found, create a default one with all paragraphs
    if not sections:
        default_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        default_section = {
            "title": "Content",
            "section_type": "default",
            "paragraphs": []
        }
        
        for idx, paragraph in enumerate(default_paragraphs):
            default_section["paragraphs"].append({
                "text": paragraph,
                "paragraph_id": f"default_{idx}"
            })
            
        sections.append(default_section)
    
    return {
        "metadata": doc_metadata,
        "sections": sections
    }

def create_hierarchical_embeddings(document_structure, client):
    """
    Create embeddings with hierarchical metadata for a document structure.
    
    Args:
        document_structure: Output from extract_document_structure
        client: OpenAI client for creating embeddings
        
    Returns:
        List of vector objects ready to be uploaded to Pinecone
    """
    doc_metadata = document_structure["metadata"]
    sections = document_structure["sections"]
    
    vectors = []
    
    # For each section in the document
    for section_idx, section in enumerate(sections):
        section_id = f"{doc_metadata['document_id']}_section_{section_idx}"
        section_title = section["title"]
        section_type = section.get("section_type", "default")
        
        # For each paragraph in the section
        for para_idx, paragraph in enumerate(section["paragraphs"]):
            para_text = paragraph["text"]
            para_id = paragraph["paragraph_id"]
            
            # Skip empty paragraphs
            if not para_text.strip():
                continue
            
            # Create a unique ID for this vector
            vector_id = f"{section_id}_para_{para_id}"
            
            # Get embedding from OpenAI
            try:
                resp = client.embeddings.create(
                    input=[para_text],
                    model="text-embedding-ada-002"
                )
                embedding = resp.data[0].embedding
                
                # Create the complete metadata
                metadata = {
                    # Document-level metadata
                    "document_id": doc_metadata["document_id"],
                    "document_title": doc_metadata["document_title"],
                    "chapter": doc_metadata["chapter"],
                    "lesson_number": doc_metadata["lesson_number"],
                    "lesson_title": doc_metadata["title"],
                    
                    # Section-level metadata
                    "section_id": section_id,
                    "section_title": section_title,
                    "section_index": section_idx,
                    "section_type": section_type,
                    
                    # Paragraph-level metadata
                    "paragraph_id": para_id,
                    "paragraph_index": para_idx,
                    
                    # Hierarchy path
                    "path": f"{doc_metadata['chapter']}/{doc_metadata['document_id']}/{section_id}/{para_id}",
                    
                    # Original lesson ID for backward compatibility
                    "lesson_id": doc_metadata["document_title"],
                    
                    # The actual text content
                    "text": para_text,
                    
                    # Special field to identify this as a hierarchical vector
                    "vector_type": "hierarchical_paragraph"
                }
                
                # Add to our list of vectors
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
            except Exception as e:
                print(f"Error creating embedding for {vector_id}: {e}")
                continue
                
    return vectors

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

# 4. Load your lessons.json (dict of lesson_id → combined text)
base_dir = os.getcwd()
lessons_path = os.path.join(base_dir, "lessons.json")
with open(lessons_path, "r", encoding="utf-8") as f:
    lessons_dict = json.load(f)

# Initialize counters for final summary
total_lessons = len(lessons_dict)
processed_count = 0
skipped_count = 0
failed_count = 0
total_chunks = 0

# Load checkpoint file if it exists
checkpoint_path = os.path.join(base_dir, "embedding_checkpoint.json")
processed_lessons = set()
if os.path.exists(checkpoint_path):
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            processed_lessons = set(json.load(f))
        print(f"Loaded checkpoint: {len(processed_lessons)} lessons already processed")
    except Exception as e:
        print(f"Error loading checkpoint file: {e}")
        print("Starting fresh")

# 5. Optional: a splitter if you want to chunk long texts (kept for backward compatibility)
def split_text(text, max_tokens=500):
    """
    Split text into chunks while preserving section integrity and respecting token limits.
    Prioritizes keeping important sections together when possible.
    """
    # Identify key topics that should be kept together when possible
    key_topics = ["Sesiuni", "Principale", "Tranzacționare", "Market Structure", "MSS", "FVG", "Rezumat"]
    
    # Improved section splitting pattern that preserves headers
    section_pattern = r'((?:#{1,4}\s+[^\n]+\n)|(?:---\n))'
    section_matches = re.split(section_pattern, text, flags=re.MULTILINE)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    current_header = ""
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    i = 0
    while i < len(section_matches):
        section = section_matches[i]
        
        # Skip empty sections
        if not section.strip():
            i += 1
            continue
        
        # Check if this is a header
        if re.match(r'^#{1,4}\s+', section) or section.strip() == '---':
            current_header = section
            i += 1
            if i < len(section_matches):
                section = section_matches[i]
            else:
                break
        
        section_with_header = current_header + section if current_header else section
        section_tokens = len(enc.encode(section_with_header))
        
        # If this is a key topic section, try to keep it together
        is_key_section = any(topic in current_header for topic in key_topics)
        
        # Special handling for summary sections
        is_summary = "Rezumat" in current_header
        
        # If this section would exceed max_tokens but is an important section,
        # finish the current chunk and put this important section in its own chunk
        if current_tokens > 0 and (current_tokens + section_tokens > max_tokens):
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # For important sections that are still too big, we might need to split
            if section_tokens > max_tokens:
                if is_key_section or is_summary:
                    # For key sections, try to keep more together - increase token limit
                    max_section_tokens = max_tokens * 1.5  # Allow 50% more tokens for important sections
                    
                    if section_tokens <= max_section_tokens:
                        # It fits within our expanded limit
                        chunks.append(section_with_header)
                    else:
                        # Still too big, split by paragraphs but keep header
                        paragraphs = section.split("\n\n")
                        temp_chunk = [current_header] if current_header else []
                        temp_tokens = len(enc.encode(current_header)) if current_header else 0
                        
                        for para in paragraphs:
                            para_tokens = len(enc.encode(para))
                            if temp_tokens + para_tokens > max_tokens and temp_chunk:
                                chunks.append("\n\n".join(temp_chunk))
                                temp_chunk = [current_header] if current_header else []
                                temp_tokens = len(enc.encode(current_header)) if current_header else 0
                            
                            temp_chunk.append(para)
                            temp_tokens += para_tokens
                        
                        if temp_chunk:
                            chunks.append("\n\n".join(temp_chunk))
                else:
                    # For regular sections, use the original paragraph splitting approach
                    paragraphs = section.split("\n\n")
                    temp_chunk = [current_header] if current_header else []
                    temp_tokens = len(enc.encode(current_header)) if current_header else 0
                    
                    for para in paragraphs:
                        para_tokens = len(enc.encode(para))
                        if temp_tokens + para_tokens > max_tokens and temp_chunk:
                            chunks.append("\n\n".join(temp_chunk))
                            temp_chunk = [current_header] if current_header else []
                            temp_tokens = len(enc.encode(current_header)) if current_header else 0
                        
                        temp_chunk.append(para)
                        temp_tokens += para_tokens
                    
                    if temp_chunk:
                        chunks.append("\n\n".join(temp_chunk))
            else:
                # It fits as a single chunk
                chunks.append(section_with_header)
        else:
            # Add to current chunk
            if current_header and not current_chunk:
                current_chunk.append(current_header)
            current_chunk.append(section)
            current_tokens += section_tokens
        
        i += 1
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks

# Sort lesson IDs by chapter number, then lesson number
sorted_lesson_ids = sorted(lessons_dict.keys(), key=extract_lesson_info)
print(f"Will process {len(sorted_lesson_ids)} lessons ({len(processed_lessons)} already done)")

# Track start time
start_time = time.time()

# 6. Process each lesson in order using the new hierarchical approach
for lesson_id in sorted_lesson_ids:
    # Skip lessons that have already been processed
    if lesson_id in processed_lessons:
        print(f"Skipping already processed lesson: {lesson_id}")
        skipped_count += 1
        continue
        
    combined_text = lessons_dict[lesson_id]
    
    # Use the new hierarchical approach
    print(f"Processing lesson: {lesson_id}")
    
    # Extract hierarchical structure
    document_structure = extract_document_structure(lesson_id, combined_text)
    
    # Create vectors with hierarchical metadata
    vectors = create_hierarchical_embeddings(document_structure, client)
    lesson_chunks = len(vectors)
    total_chunks += lesson_chunks
    
    print(f"Created {lesson_chunks} vectors with hierarchical metadata")
    
    # Track if all vectors for this lesson were uploaded successfully
    all_vectors_successful = True
    
    # Batch upload vectors to Pinecone (in batches of 100)
    for i in range(0, len(vectors), 100):
        batch = vectors[i:i+100]
        try:
            index.upsert(batch)
            print(f"  ✅ Uploaded batch {i//100 + 1}/{(len(vectors)//100) + 1} ({len(batch)} vectors)")
        except Exception as e:
            print(f"  ❌ Error uploading batch: {e}")
            all_vectors_successful = False
    
    if all_vectors_successful:
        # Mark lesson as processed and update checkpoint
        processed_lessons.add(lesson_id)
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(list(processed_lessons), f)
        print(f"✅ Uploaded {lesson_id} ({lesson_chunks} vectors) - Checkpoint updated")
        processed_count += 1
    else:
        print(f"⚠️ Uploaded {lesson_id} with some errors - Not marked as complete")
        failed_count += 1

# Calculate elapsed time
elapsed_time = time.time() - start_time
minutes, seconds = divmod(elapsed_time, 60)
hours, minutes = divmod(minutes, 60)

# Print final summary
print("\n" + "="*50)
print("UPLOAD SUMMARY")
print("="*50)
print(f"Total lessons processed: {processed_count + skipped_count} of {total_lessons}")
print(f"  - Successfully processed: {processed_count}")
print(f"  - Previously processed: {skipped_count}")
print(f"  - Failed: {failed_count}")
print(f"Total chunks uploaded: {total_chunks}")
print(f"Total time elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s")
print(f"Average time per lesson: {elapsed_time/(processed_count or 1):.2f}s")
print("="*50)
print("🎉 All done!")

index_stats = index.describe_index_stats()
print(f"Total vectors in index: {index_stats.total_vector_count}")
