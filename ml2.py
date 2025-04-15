from fastapi import FastAPI, Query
import os
from typing import List

app = FastAPI()

# You can add as many directories here as needed
log_dirs = [
    "./logs/service1",
    "./logs/service2"
]

# Utility function to search keyword in all files under multiple directories
def search_keyword_in_files(directories: List[str], keyword: str):
    keyword = keyword.lower()
    total_count = 0
    matched_files = []

    for root_dir in directories:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
                        content = file.read().lower()
                        count = content.count(keyword)
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
    results = search_keyword_in_files(log_dirs, keyword)
    return results