import requests
import json
from datetime import datetime
import os

# Configuration
API_URL = "https://web-production-4b33.up.railway.app/admin/export-feedback"
API_KEY = "e4784571eb1350e2f70bfa9f74ca86af"
BACKUP_DIR = os.path.join(os.path.expanduser("~/Desktop/Current GitHub TI Project Files"), "feedback_logs")
MASTER_FILE = os.path.join(BACKUP_DIR, "all_feedback_logs.json")

# Print configuration details for logging
print(f"Script running at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Using API URL: {API_URL}")
print(f"Backup directory: {BACKUP_DIR}")
print(f"Master log file: {MASTER_FILE}")

# Create backup directory if it doesn't exist
os.makedirs(BACKUP_DIR, exist_ok=True)

# Fetch logs
print(f"Fetching logs from {API_URL}...")
try:
    response = requests.get(f"{API_URL}?api_key={API_KEY}")
    
    print(f"Response status code: {response.status_code}")
    
    if response.status_code == 200:
        new_data = response.json()
        new_logs = new_data.get('logs', [])
        log_count = len(new_logs)
        print(f"Retrieved {log_count} log entries")
        
        # Load existing logs if file exists
        existing_logs = []
        if os.path.exists(MASTER_FILE):
            try:
                with open(MASTER_FILE, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    existing_logs = existing_data.get('logs', [])
                print(f"Loaded {len(existing_logs)} existing log entries")
            except Exception as e:
                print(f"Error loading existing logs: {e}")
                # Create a backup of the corrupt file if it exists
                if os.path.exists(MASTER_FILE):
                    backup_name = f"{MASTER_FILE}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
                    os.rename(MASTER_FILE, backup_name)
                    print(f"Created backup of corrupt log file: {backup_name}")
        
        # Get unique identifier for each log entry (e.g., timestamp + session_id)
        # This prevents duplicate entries
        existing_ids = set()
        for entry in existing_logs:
            # Create a unique ID from timestamp and session_id
            entry_id = f"{entry.get('timestamp', '')}-{entry.get('session_id', '')}"
            existing_ids.add(entry_id)
        
        # Add new entries that don't already exist
        added_count = 0
        for entry in new_logs:
            entry_id = f"{entry.get('timestamp', '')}-{entry.get('session_id', '')}"
            if entry_id not in existing_ids:
                existing_logs.append(entry)
                added_count += 1
        
        # Create the combined data
        combined_data = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_entries": len(existing_logs),
            "logs": existing_logs
        }
        
        # Save to the master file
        with open(MASTER_FILE, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        print(f"Added {added_count} new entries to master log file")
        print(f"Total entries in master log file: {len(existing_logs)}")
        
        # Also save a timestamped backup once a week (on Sundays)
        if datetime.now().weekday() == 6:  # 6 = Sunday
            backup_file = os.path.join(BACKUP_DIR, f"backup_logs_{datetime.now().strftime('%Y%m%d')}.json")
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)
            print(f"Created weekly backup: {backup_file}")
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Exception occurred: {str(e)}")