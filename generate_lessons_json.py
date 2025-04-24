import os
import re
import json

BASE_DIR = "."  # Current folder

lessons = []
total_files_checked = 0
total_txt_files = 0
total_regex_matches = 0

print(f"Looking for lessons in: {os.path.abspath(BASE_DIR)}")

# Walk through all directories and subdirectories
for root, dirs, files in os.walk(BASE_DIR):
    for filename in sorted(files):
        file_path = os.path.join(root, filename)
        relative_path = os.path.join(os.path.relpath(root, BASE_DIR), filename)
        
        total_files_checked += 1
        
        # Skip non-txt files
        if not filename.endswith('.txt'):
            continue
            
        total_txt_files += 1
        print(f"📄 Checking file: {relative_path}")
        
        # Super simplified pattern that just looks for numbers in the right places
        # This matches: digit after "Lectia" and digit after "Capitolul"
        simple_match = re.search(r"Lec\w*\s*(\d+).*Capitolul\s*(\d+)", filename)
        
        if not simple_match:
            print(f"  ⚠️ Could not parse: {filename}")
            continue
            
        total_regex_matches += 1
        
        lesson_num, chapter_num = simple_match.groups()
        
        # Try to extract title - anything between the lesson number and "Capitolul"
        title_pattern = r"Lec\w*\s*\d+\s*[-–]?\s*(.*?)(?:\s*[-–]\s*Capitolul|\s*[-–]\s*Summary|\s*[-–]\s*Transcript)"
        title_match = re.search(title_pattern, filename)
        
        if title_match:
            title = title_match.group(1).strip()
            # Clean up title by removing any trailing dashes
            title = re.sub(r'\s*[-–].*$', '', title).strip()
        else:
            title = f"Lesson {lesson_num}"
        
        lesson_id = f"capitol_{chapter_num}_lectia_{lesson_num.zfill(2)}"
        chapter_label = f"Capitolul {chapter_num}"
        lesson_label = f"Lecția {lesson_num}"
        
        print(f"  ✅ Parsed lesson {lesson_num} in chapter {chapter_num}: {title}")

        # Check for file type (summary or transcript)
        file_type = "unknown"
        if "Summary" in filename:
            file_type = "summary"
        elif "Transcript" in filename:
            file_type = "transcript"
        
        # Find existing lesson or create new one
        existing = next((l for l in lessons if l["id"] == lesson_id), None)
        
        if not existing:
            lesson_data = {
                "id": lesson_id,
                "chapter_num": int(chapter_num),
                "lesson_num": int(lesson_num),
                "title": title,
                "chapter_label": chapter_label,
                "lesson_label": lesson_label,
                "files": {}
            }
            lesson_data["files"][file_type] = file_path
            lessons.append(lesson_data)
        else:
            # Update existing lesson with this file type
            if file_type not in existing["files"]:
                existing["files"][file_type] = file_path
                print(f"  📝 Added {file_type} file to existing lesson")
            else:
                print(f"  ⚠️ Duplicate {file_type} file found for lesson ID: {lesson_id}")

# Sort lessons by chapter and lesson number
lessons.sort(key=lambda x: (x["chapter_num"], x["lesson_num"]))

# Write lessons to a JSON file
with open("lessons.json", "w", encoding="utf-8") as f:
    json.dump(lessons, f, ensure_ascii=False, indent=2)

print(f"\n====== Summary ======")
print(f"Total items checked: {total_files_checked}")
print(f"Total .txt files found: {total_txt_files}")
print(f"Total regex matches: {total_regex_matches}")
print(f"Total lessons processed: {len(lessons)}")
