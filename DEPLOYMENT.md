# Deploying to Streamlit Community Cloud

## Repository layout

- **Main module:** `streamlit_app.py` (set this as the app entry file in the Cloud dashboard).
- **Python deps:** `requirements.txt` at the repo root.
- **Vector index:** The app reads a **on-disk Chroma** directory. By default that is `chroma_db/` next to the code. You must either:
  - **Commit `chroma_db/`** to the same Git repo you connect to Streamlit Cloud, or
  - Build the index in CI and push it, or use **Git LFS** if the folder is large, or
  - Set secret **`CHROMA_PERSIST_DIR`** to a path that exists in the deployed tree (for example after a custom build step).

Streamlit Community Cloud does **not** persist arbitrary local files across redeploys unless you use supported storage; treating the index as **part of the repo** (or rebuilt on each deploy) is the straightforward approach.

**Ingestion (`ingest.py`)** expects PDFs under `GM 2022 - External Folder/…` and is normally run **locally** or in CI, not on Streamlit’s runtime (no Tesseract in the default image unless you add `packages.txt`).

## Secrets (App → Settings → Secrets)

Paste TOML matching the shape in `secrets.toml.example`. Minimum:

| Key | Required | Purpose |
|-----|----------|---------|
| `GROQ_API_KEY` | Yes | Groq API key for the chat model |
| `CHROMA_PERSIST_DIR` | No | Path to persisted Chroma (default `chroma_db`) |
| `HF_OFFLINE` | No | Set `true` only if models are pre-cached; **omit on Cloud** so embeddings can download |
| `GROQ_VISION_DELAY` | No | Used only by `ingest.py` with `--groq-vision` |

The app resolves configuration with `env_config.get_secret()`, which reads **`st.secrets` first**, then `os.environ` (including a local `.env`).

## Hugging Face / cold start

By default, **HF offline mode is off** unless you set `HF_OFFLINE=true` (in secrets or `.env`). On Streamlit Cloud, the first run may **download** `sentence-transformers/all-MiniLM-L6-v2`, which needs outbound network access and adds cold-start time.

For **local** air-gapped use, set `HF_OFFLINE=true` and ensure the model is already in your HF cache.

## Local vs Cloud quick reference

| Item | Local | Streamlit Cloud |
|------|--------|------------------|
| Groq key | `.env` | Secrets → `GROQ_API_KEY` |
| Chroma | `./chroma_db` or `CHROMA_PERSIST_DIR` | Same; folder must exist in deployment |
| HF download | Optional `HF_OFFLINE=true` | Omit `HF_OFFLINE` |

## Caching

`streamlit_app.get_engine()` is decorated with `@st.cache_resource` so **embeddings, Chroma, and the LLM client** are created once per server process, not on every rerun or button click.
