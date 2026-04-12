# Demo Recording Checklist

This runbook helps you record a clean 30-45 second product demo for GitHub.

## 1) Pre-flight Setup

- Activate the project environment.
- Ensure model artifacts exist in models/.
- Confirm ports are free: 8000 (API), 8501 (Streamlit).
- Close unrelated apps/tabs to reduce visual noise.

## 2) Start Services

Option A: manual

- uvicorn app.main:app --reload
- streamlit run app/ui.py

Option B: helper script

- powershell -ExecutionPolicy Bypass -File scripts/start-demo.ps1

## 3) Recording Tool Settings

Use OBS, ShareX, or ScreenToGif.

Recommended settings:

- Resolution: 1280x720
- FPS: 20 to 30
- Cursor highlight: On
- Capture area: browser window only

## 4) Shot List (35 sec)

- 0s to 5s: Open API docs at http://127.0.0.1:8000/docs
- 5s to 10s: Show POST /search request body
- 10s to 22s: Switch to Streamlit UI, enter query: gaming laptop
- 22s to 30s: Click Search and scroll top results
- 30s to 35s: Repeat with query: wireless earbuds

## 5) Narration Script (Optional)

- This is a two-stage ranking system.
- BM25 retrieves top candidates first.
- A LightGBM LambdaRank model reranks them using lexical and semantic features.
- You can see relevant items pushed to the top in real time.

## 6) Export and Convert to GIF

- Export MP4 first.
- Auto-convert with helper script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/convert-demo-to-gif.ps1 -InputMp4 docs/assets/demo.mp4 -OutputGif docs/assets/demo.gif
```

- Or convert with ScreenToGif / ezgif.com (keep under 8-12 MB).
- Save as docs/assets/demo.gif.

## 7) Replace README Placeholder

- Open README.md
- Replace docs/assets/demo-placeholder.svg with docs/assets/demo.gif

## 8) Quick Quality Gate

- Text readable at normal zoom.
- No terminal errors visible.
- Demo under 45 seconds.
- Shows at least two different queries.
