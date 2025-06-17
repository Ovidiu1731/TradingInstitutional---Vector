import os
import json
import tiktoken
import re
from dotenv import load_dotenv
import time
import logging
from pinecone import Pinecone
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Function to sanitize IDs (remove special characters, replace spaces with underscores)
def sanitize_id(text):
    """Clean up text to create valid vector IDs - ASCII only."""
    # Romanian character replacements
    replacements = {
        'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ț': 't',
        'Ă': 'A', 'Â': 'A', 'Î': 'I', 'Ș': 'S', 'Ț': 'T',
        'ü': 'u', 'Ü': 'U', 'ö': 'o', 'Ö': 'O',
        '–': '-', '—': '-', ''': "'", ''': "'", '"': '"', '"': '"'
    }
    
    # Replace Romanian characters
    for romanian, replacement in replacements.items():
        text = text.replace(romanian, replacement)
    
    # Keep only ASCII letters, numbers, hyphens, and underscores
    text = re.sub(r'[^a-zA-Z0-9\-_]', '_', text)
    
    # Remove multiple consecutive underscores
    text = re.sub(r'_+', '_', text)
    
    # Remove leading/trailing underscores
    text = text.strip('_')
    
    # Ensure it's not empty
    if not text:
        text = "document"
    
    return text

# Function to extract chapter and lesson numbers for sorting
def extract_lesson_info(filepath):
    """Extract chapter and lesson numbers from the actual file pattern."""
    filename = os.path.basename(filepath)
    chapter_num = 0
    lesson_num = 0
    
    # Extract from directory path first (most reliable)
    if 'Capitolul' in filepath:
        dir_match = re.search(r'Capitolul\s*(\d+)', filepath)
        if dir_match:
            chapter_num = int(dir_match.group(1))
    
    # Extract lesson number from filename
    # Pattern: "Lectia X - Title - Capitolul Y - Summary/Transcript.txt"
    lesson_match = re.search(r'Lectia\s*(\d+)', filename)
    if lesson_match:
        lesson_num = int(lesson_match.group(1))
    
    # If chapter not found from directory, try from filename
    if chapter_num == 0:
        chapter_match = re.search(r'Capitolul\s*(\d+)', filename)
        if chapter_match:
            chapter_num = int(chapter_match.group(1))
    
    return chapter_num, lesson_num

def extract_document_structure(filepath, text):
    """Extracts the hierarchical structure and ensures metadata is always present."""
    # Use filename as document_id
    document_id = os.path.basename(filepath)
    # Try to extract a title from the first markdown header, else use filename
    title_match = re.search(r'^#+\s*(.+)', text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else document_id
    
    # Split into sections by markdown headers
    sections = []
    header_matches = list(re.finditer(r'(^#+\s*.+$)', text, re.MULTILINE))
    
    if header_matches:
        for i, match in enumerate(header_matches):
            section_title = match.group(0).strip('#').strip()
            start = match.end()
            
            # Find the end of this section (start of next header or end of text)
            if i + 1 < len(header_matches):
                end = header_matches[i + 1].start()
            else:
                end = len(text)
            
            section_content = text[start:end].strip()
            
            sections.append({
                'title': section_title,
                'content': section_content
            })
    else:
        # If no headers, treat the whole text as one section
        sections = [{
            'title': title,
            'content': text.strip()
        }]
    
    # Ensure all sections have both title and content
    for i, section in enumerate(sections):
        if 'title' not in section:
            section['title'] = f"Section_{i+1}"
        if 'content' not in section:
            section['content'] = ""
    
    return {
        'metadata': {
            'document_id': document_id,
            'title': title
        },
        'content': text.strip(),
        'sections': sections
    }

def create_hierarchical_embeddings(document_structure, client):
    """Create embeddings with hierarchical metadata for a document structure."""
    embeddings = []
    
    # Get chapter and lesson numbers using the filepath
    # We need to pass the original filepath, which we can get from document_id
    filepath = document_structure['metadata']['document_id']
    chapter_num, lesson_num = extract_lesson_info(filepath)
    
    # Create embedding for the entire document
    full_text = document_structure['content']
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=full_text
    ).data[0].embedding
    
    # Create sanitized document ID
    sanitized_doc_id = sanitize_id(document_structure['metadata']['document_id'])
    
    # Add document-level metadata
    metadata = {
        'document_id': document_structure['metadata']['document_id'],
        'title': document_structure['metadata']['title'],
        'chapter': chapter_num,
        'lesson': lesson_num,
        'type': 'document'
    }
    
    embeddings.append({
        'id': f"{sanitized_doc_id}_full",
        'values': embedding,
        'metadata': metadata
    })
    
    # Create embeddings for each section
    for i, section in enumerate(document_structure['sections']):
        section_text = section['content']
        if not section_text.strip():
            continue  # Skip empty sections
            
        section_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=section_text
        ).data[0].embedding
        
        # Get section title, use fallback if missing
        section_title = section.get('title', f'Section_{i+1}')
        sanitized_section_title = sanitize_id(section_title)
        
        # Add section-level metadata
        section_metadata = {
            'document_id': document_structure['metadata']['document_id'],
            'title': section_title,
            'chapter': chapter_num,
            'lesson': lesson_num,
            'type': 'section'
        }
        
        embeddings.append({
            'id': f"{sanitized_doc_id}_{sanitized_section_title}",
            'values': section_embedding,
            'metadata': section_metadata
        })
    
    return embeddings

def process_directory(directory_path):
    """Process all lesson files in the directory and upload to Pinecone."""
    # Initialize clients
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(os.getenv('PINECONE_INDEX_NAME', 'trading-lessons'))
    
    # Find all lesson files - ONLY from Capitolul directories
    lesson_files = []
    
    # Process each Capitolul directory
    for i in range(1, 12):  # Capitolul 1 through 11
        capitolul_dir = f"Capitolul {i}"
        if os.path.exists(capitolul_dir):
            for file in os.listdir(capitolul_dir):
                if (file.endswith('.txt') and 
                    not file.startswith('.') and
                    ('Lectia' in file or 'Summary Capitolul' in file)):
                    lesson_files.append(os.path.join(capitolul_dir, file))
    
    print(f"Found {len(lesson_files)} lesson files to process")
    
    # Process each file
    for file_path in lesson_files:
        try:
            print(f"Processing: {file_path}")
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print(f"Skipping empty file: {file_path}")
                continue
            
            # Create document structure
            doc_structure = extract_document_structure(file_path, content)
            
            # Create embeddings
            embeddings = create_hierarchical_embeddings(doc_structure, client)
            
            # Upload to Pinecone in batches
            batch_size = 50  # Smaller batches for reliability
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i+batch_size]
                index.upsert(vectors=batch)
            
            print(f"✓ Successfully processed: {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}")
            continue  # Continue with next file
    
    print(f"\n🎉 Upload process completed! Processed {len(lesson_files)} files.")
    
    # Verify the upload
    stats = index.describe_index_stats()
    print(f"📊 Total vectors in index: {stats['total_vector_count']}")
    
    return True

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Process current directory
    process_directory(".")
