# Your documents go here

Drop `.txt` or `.md` files anywhere in this folder (subfolders are fine — the
ingester walks recursively). Nothing in this project ships with pre-loaded
documents; you provide the corpus.

Good options, easiest to hardest:
1. A static set of technical docs (course notes, a library's documentation).
2. A GitHub repo's README + docs + issues, pulled down periodically.

Once files are here, run:

```bash
python -m app.ingest
```

to (re)build the vector index in `chroma_db/` (created automatically, gitignored).
