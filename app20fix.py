LOG_DIRECTORIES = ["logs", "service_logs", "payments_logs"]  # Add your folders here

@app.on_event("startup")
def load_logs():
    global cached_logs
    cached_logs = {}
    for log_dir in LOG_DIRECTORIES:
        cached_logs.update(read_logs_from_directory(log_dir))

def read_logs_from_directory(directory: str) -> Dict[str, List[str]]:
    logs = {}
    path = Path(directory)
    if not path.exists():
        return logs

    for file in path.rglob("*.log"):
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                logs[file.name] = group_multiline_logs(f.readlines())
        except Exception as e:
            print(f"Failed to read {file}: {e}")

    for file in path.rglob("*.gz"):
        try:
            with gzip.open(file, "rt", encoding="utf-8", errors="ignore") as f:
                logs[file.name] = group_multiline_logs(f.readlines())
        except Exception as e:
            print(f"Failed to read {file}: {e}")

    return logs