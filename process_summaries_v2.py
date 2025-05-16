import json
from pathlib import Path
import re

# --- Configuration ---
CHAPTER_PREFIX       = "Capitolul "
OUTPUT_JSON_FILENAME = "lessons.json"
# --- End Configuration ---

def normalize_text(text):
    """Normalize text by replacing en-dashes with hyphens and standardizing spaces"""
    text = text.replace("–", "-")  # Replace en-dash with hyphen
    text = re.sub(r'\s+', ' ', text).strip()  # Standardize whitespace
    return text

def process_chapters(project_root: Path) -> dict:
    lessons = {}
    print(f"Starting processing in base directory: {project_root}")

    # Debugging: Print all chapter directories found
    chapter_dirs = [d for d in project_root.iterdir() if d.is_dir() and d.name.startswith(CHAPTER_PREFIX)]
    print(f"Found {len(chapter_dirs)} chapter directories: {[d.name for d in chapter_dirs]}")

    # 1) Iterate all chapter folders
    for chapter_dir in chapter_dirs:
        # Debugging: Print all summary files in this chapter
        summary_files = list(chapter_dir.glob("*Summary.txt"))
        print(f"Found {len(summary_files)} summary files in {chapter_dir.name}")

        # 2) For each summary file in this folder
        for summary_path in summary_files:
            # Build the lesson stem (drop " - Summary")
            stem = summary_path.stem.replace(" - Summary", "")
            
            # Check if this is a REF file (chapter reference)
            if stem.startswith("REF"):
                # Extract chapter number from directory name
                chapter_match = re.search(r'Capitolul\s+(\d+)', chapter_dir.name)
                chapter_num = chapter_match.group(1) if chapter_match else "?"
                
                # Add REF file directly without looking for transcript
                try:
                    summary_text = summary_path.read_text(encoding="utf-8").strip()
                    key = stem  # Use the REF filename as the key
                    lessons[key] = summary_text
                    print(f"Processed chapter reference: {key}")
                    continue  # Skip to next summary file
                except Exception as e:
                    print(f"ERROR processing REF file {stem}: {e}")
                    continue
            
            # For normal lessons, extract lesson number using regex for more robust matching
            lesson_match = re.search(r'Lectia\s+(\d+)', stem, re.IGNORECASE)
            if not lesson_match:
                print(f"WARNING: Could not extract lesson number from '{stem}' in {chapter_dir.name}")
                continue
            
            lesson_num = lesson_match.group(1)
            
            # 3) Find transcript for this lesson using more flexible matching
            transcript_pattern = f"*Lectia*{lesson_num}*Transcript*.txt"
            transcript_candidates = list(chapter_dir.glob(transcript_pattern))
            
            if not transcript_candidates:
                print(f"WARNING: No transcript found for 'Lectia {lesson_num}' in {chapter_dir.name}")
                continue
            
            transcript_path = transcript_candidates[0]

            try:
                # 4) Read files
                summary_text    = summary_path.read_text(encoding="utf-8").strip()
                transcript_text = transcript_path.read_text(encoding="utf-8").strip()

                # 5) Combine with labels
                parts = []
                if summary_text:
                    parts.append(f"Rezumat:\n{summary_text}")
                if transcript_text:
                    parts.append(f"Trascriere:\n{transcript_text}")
                combined = "\n\n".join(parts)

                # 6) Key by "Capitolul X • Lectia Y – Title"
                # Extract chapter number
                chapter_match = re.search(r'Capitolul\s+(\d+)', chapter_dir.name)
                chapter_num = chapter_match.group(1) if chapter_match else "?"
                
                # Extract title from file stem
                title_match = re.search(r'Lectia\s+\d+\s*[-–]\s*(.*?)(?:\s*-\s*Capitolul|$)', stem)
                title = title_match.group(1).strip() if title_match else stem
                
                key = f"Capitolul {chapter_num} • Lectia {lesson_num} - {title}"
                lessons[key] = combined
                print(f"Processed lesson: {key}")

            except Exception as e:
                print(f"ERROR processing lesson {stem}: {e}")

    print(f"\nTotal lessons processed: {len(lessons)}")
    return lessons

def write_json_output(data: dict, output_path: Path):
    print(f"\nWriting combined data to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("Successfully wrote lessons JSON.")

if __name__ == "__main__":
    root = Path(__file__).parent.resolve()
    lessons_dict = process_chapters(root)
    if lessons_dict:
        write_json_output(lessons_dict, root / OUTPUT_JSON_FILENAME)
    else:
        print("No lessons processed; JSON file not written.")
