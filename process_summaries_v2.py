# process_summaries_v2.py

import os
import re
import json
import logging
from pathlib import Path # Using pathlib can make path handling easier

# --- Step A: Setup ---

# Configure logging (especially useful for tracking progress and errors)
# Keep level DEBUG to see detailed messages during development
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Step B: Find Files ---

SOURCE_DIR = Path("/Users/ovidiudodan/Desktop/Lectii/Summaries v1")
# Or if you placed it inside your project, maybe: SOURCE_DIR = Path("./data/summaries")

all_summary_files = []
# Use Path.rglob to find all files ending with "Summary.txt" recursively
logging.info(f"Searching for summary files in: {SOURCE_DIR}")
all_summary_files = list(SOURCE_DIR.rglob("*Summary.txt"))

if not all_summary_files:
    logging.error(f"No summary files found in {SOURCE_DIR}. Please check the path.")
    exit() # Stop if no files are found

logging.info(f"Found {len(all_summary_files)} summary files.")

# --- Step C & D: Process Each File and Create Chunk Objects ---
all_chunks = []
skipped_files_no_headings = 0

for file_path in all_summary_files:
    logging.debug(f"Processing file: {file_path}")
    try:
        # --- Extract Base Metadata from Filename ---
        filename = file_path.name # Get just the filename (e.g., "Lectia 1 - ... - Summary.txt")

        # Refined Regex based on confirmed pattern:
        # Captures: Lesson Number/ID, Title, Chapter Number
        match = re.match(r"Lectia (\S+) - (.*?) - Capitolul (\d+) - Summary\.txt", filename, re.IGNORECASE)

        if match:
            lesson_num_str = match.group(1).strip() # Use group(1) for lesson number/id
            lesson_title = match.group(2).strip()   # Use group(2) for title
            chapter_num = int(match.group(3).strip())# Use group(3) for chapter number
            # Create a lesson_id (combine lesson and chapter for uniqueness)
            lesson_id = f"L{lesson_num_str}_C{chapter_num}"
        else:
            # Fallback if regex fails - maybe log and skip or use filename parts
            logging.warning(f"Could not parse standard metadata from filename: {filename}. Using fallback.")
            lesson_id = f"UnknownID_{filename[:15]}" # Less ideal ID
            lesson_title = "Unknown Title"
            chapter_num = 0

        base_metadata = {
            "lesson_id": lesson_id,
            "lesson_title": lesson_title,
            "chapter": chapter_num,
            "source_type": "summary",
            "source_filename": filename
        }

        # --- Read File Content ---
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # --- Pre-processing: Remove Preamble (text before the first ### or #### heading) ---
        # Find the start index of the first real heading
        first_heading_match = re.search(r"^(### |#### )", content, re.MULTILINE)
        if first_heading_match:
            content_start_index = first_heading_match.start()
            content_to_split = content[content_start_index:]
        else:
            logging.warning(f"No '### ' or '#### ' headings found in {filename}. Skipping file.")
            skipped_files_no_headings += 1
            continue # Skip this file if no headings are found

        # --- Splitting into Chunks by Heading ---
        # Regex splits *before* the delimiter (the heading line itself)
        # It looks for lines STARTING with ### or #### followed by a space
        # `re.MULTILINE` ensures `^` matches start of lines, not just start of string
        raw_chunks = re.split(r'^(?=### |#### )', content_to_split, flags=re.MULTILINE)

        current_heading = "Unknown Heading" # Keep track of the heading for the current chunk

        for raw_chunk in raw_chunks:
            chunk_text = raw_chunk.strip()

            if not chunk_text: # Skip empty strings that can result from splitting
                continue

            # --- Extract Heading for Metadata ---
            lines = chunk_text.splitlines() # Split the chunk into lines
            if lines and (lines[0].startswith("### ") or lines[0].startswith("#### ")):
                # Extract the first line as the heading
                # Clean it up: remove hash marks, leading/trailing spaces, maybe markdown like **
                heading_text = re.sub(r'^[#]+\s+', '', lines[0]) # Remove '### ' or '#### '
                heading_text = heading_text.replace('**', '').strip() # Remove bold markdown and trim spaces
                current_heading = heading_text
            # If a chunk somehow doesn't start with a heading after splitting,
            # it will retain the 'current_heading' from the previous chunk,
            # or the "Unknown Heading" default if it's the very first part.

            # --- Create Final Chunk Object ---
            metadata = base_metadata.copy() # Start with file-level metadata
            metadata['heading'] = current_heading # Add the specific heading for this chunk
            # metadata['char_count'] = len(chunk_text) # Optional: track length

            # Add the complete chunk data to our master list
            all_chunks.append({'text': chunk_text, 'metadata': metadata})

    except Exception as e:
        # Log any error during the processing of a single file
        logging.exception(f"Error processing file {file_path}: {e}")


# --- Step E: Output / Save for Review ---
logging.info(f"\nProcessing complete.")
logging.info(f"Successfully created {len(all_chunks)} chunks.")
if skipped_files_no_headings > 0:
    logging.warning(f"Skipped {skipped_files_no_headings} files because no '### ' or '#### ' headings were found.")

# --- Verification: Save the processed chunks to a JSON file ---
# !!! Recommended: Uncomment this block for your first run !!!
output_filename = "processed_summary_chunks.json"
try:
    with open(output_filename, "w", encoding="utf-8") as f:
        # Use ensure_ascii=False to keep Romanian characters correct in JSON
        # Use indent=2 for readability
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    logging.info(f"Successfully saved processed chunks to {output_filename}")
except Exception as e:
    logging.error(f"Failed to save chunks to JSON: {e}")

# --- Verification: Print some examples ---
# Optional: Uncomment to print first few chunks to console
# logging.info("\nExample Chunks:")
# for i, chunk_data in enumerate(all_chunks[:3]): # Print first 3
#      logging.debug(f"--- Chunk {i+1} ---")
#      logging.debug(f"Metadata: {chunk_data['metadata']}")
#      logging.debug(f"Text Snippet: {chunk_data['text'][:500]}...") # Print start of text
#      logging.debug("-" * 20)

# Next steps would be to take 'all_chunks', generate embeddings for the 'text',
# and upload to Pinecone with the corresponding 'metadata'.
