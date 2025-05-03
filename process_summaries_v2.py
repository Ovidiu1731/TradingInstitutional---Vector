import os
from pathlib import Path
import json

# --- Configuration ---
# Number of chapter folders to look for (e.g., Capitolul 1 to Capitolul 11)
NUM_CHAPTERS = 11
# Names of the files within each chapter folder
SUMMARY_FILENAME = 'summary.txt'
TRANSCRIPT_FILENAME = 'transcript.txt'
# Name of the output JSON file
OUTPUT_JSON_FILENAME = 'lessons.json'
# --- End Configuration ---

def process_chapters(num_chapters: int, project_root: Path) -> dict:
    """
    Reads summary and transcript files from chapter folders, combines them,
    and returns a dictionary mapping chapter number to combined text.
    """
    lessons_content = {}
    print(f"Starting processing in base directory: {project_root}")

    for i in range(1, num_chapters + 1):
        chapter_folder_name = f'Capitolul {i}'
        source_directory = project_root / chapter_folder_name

        summary_path = source_directory / SUMMARY_FILENAME
        transcript_path = source_directory / TRANSCRIPT_FILENAME

        # Check if the required files exist in the chapter folder
        if summary_path.is_file() and transcript_path.is_file():
            try:
                print(f"Processing Chapter {i} in {source_directory}...")
                # Read summary
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_content = f.read().strip()

                # Read transcript
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript_content = f.read().strip()

                # Combine summary and transcript
                combined_content_parts = []
                if summary_content:
                     combined_content_parts.append(f"Rezumat:\n{summary_content}")
                if transcript_content:
                    combined_content_parts.append(f"Trascriere:\n{transcript_content}")
                combined_content = "\n\n".join(combined_content_parts)

                # Store the combined text using chapter number (as string) as key
                lessons_content[str(i)] = combined_content
                print(f"Successfully processed Chapter {i}.")

            except Exception as e:
                print(f"ERROR processing Chapter {i}: {e}")
        else:
            # Print a warning if files are missing for a chapter
            missing = []
            if not source_directory.is_dir():
                 print(f"WARNING: Skipping Chapter {i}: Directory not found: {source_directory}")
                 continue
            if not summary_path.is_file():
                missing.append(summary_path.name)
            if not transcript_path.is_file():
                missing.append(transcript_path.name)
            print(f"WARNING: Skipping Chapter {i}: Missing file(s) in {source_directory}: {', '.join(missing)}")

    return lessons_content

def write_json_output(data: dict, output_path: Path):
    """Writes the processed data dictionary to a JSON file."""
    try:
        print(f"\nWriting combined data to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Successfully created/updated JSON data file: {output_path}")
    except Exception as e:
        print(f"ERROR writing JSON file: {e}")

if __name__ == "__main__":
    # Determine the project root directory (where the script is located)
    # This makes the script work correctly regardless of where you run it from,
    # as long as the script itself is in the root project folder.
    project_root_dir = Path(__file__).parent.resolve()

    # Process the chapters
    lessons_dictionary = process_chapters(NUM_CHAPTERS, project_root_dir)

    # Define the output JSON file path
    json_output_path = project_root_dir / OUTPUT_JSON_FILENAME

    # Write the dictionary to the JSON file
    if lessons_dictionary:
        write_json_output(lessons_dictionary, json_output_path)
    else:
        print("No chapters were processed successfully. JSON file not written.")