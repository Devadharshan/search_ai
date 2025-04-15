from fastapi import FastAPI, Query
import os
from pathlib import Path
from typing import List

app = FastAPI()

# Set the root log directory
LOG_ROOT_DIR = Path("./logs")

# Utility function to search keyword in all files under the root directory
def search_keyword_in_files(root_dir: Path, keyword: str):
    keyword = keyword.lower()
    total_count = 0
    matched_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
                    file_content = file.read().lower()
                    count = file_content.count(keyword)
                    if count > 0:
                        matched_files.append({"file": filepath, "count": count})
                        total_count += count
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    return {
        "total_count": total_count,
        "matched_files": matched_files
    }

@app.get("/search")
def search_logs(keyword: str = Query(..., description="Keyword to search in log files")):
    results = search_keyword_in_files(LOG_ROOT_DIR, keyword)
    return results