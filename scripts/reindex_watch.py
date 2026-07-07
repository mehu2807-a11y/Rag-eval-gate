"""
Watches data/ for added/changed/removed documents. On any change, debounces
briefly, then re-runs ingestion and the eval suite so you immediately know
whether new documents helped or hurt retrieval quality.

Usage:
    python -m scripts.reindex_watch

This is a lightweight stand-in for a Prefect/Airflow-orchestrated pipeline --
swap this out for a scheduled Prefect flow if you want to demo real
orchestration tooling (see the docstring at the bottom for how).
"""
import subprocess
import sys
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app import config

DEBOUNCE_SECONDS = 3


class ReindexHandler(FileSystemEventHandler):
    def __init__(self):
        self._last_event_time = 0

    def on_any_event(self, event):
        if event.is_directory:
            return
        if not event.src_path.lower().endswith((".txt", ".md")):
            return
        self._last_event_time = time.time()

    def maybe_run(self):
        if self._last_event_time and time.time() - self._last_event_time > DEBOUNCE_SECONDS:
            self._last_event_time = 0
            print("Change detected in data/ -- re-indexing and re-evaluating...")
            subprocess.run([sys.executable, "-m", "app.ingest"], check=False)
            subprocess.run([sys.executable, "-m", "eval.run_eval"], check=False)


def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    handler = ReindexHandler()
    observer = Observer()
    observer.schedule(handler, str(config.DATA_DIR), recursive=True)
    observer.start()
    print(f"Watching {config.DATA_DIR} for changes (Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(1)
            handler.maybe_run()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()

# --- Prefect version (optional upgrade path) ---
# from prefect import flow, task
#
# @task
# def reindex():
#     from app.ingest import build_index
#     return build_index(reset=True)
#
# @task
# def evaluate():
#     import subprocess, sys
#     subprocess.run([sys.executable, "-m", "eval.run_eval"], check=True)
#
# @flow
# def reindex_and_eval():
#     reindex()
#     evaluate()
#
# if __name__ == "__main__":
#     reindex_and_eval.serve(name="reindex-on-schedule", interval=3600)
