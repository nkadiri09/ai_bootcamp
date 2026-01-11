# Financials Extractor Streamlit App

This small Streamlit app extracts revenue, EPS, and forecast figures from pasted or uploaded financial report text using an LLM (OpenAI optional) or a local fallback parser.

Setup

1. Create a virtualenv and install deps:

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` if you want to use OpenAI.

Run

  streamlit run app.py

Notes

- The app attempts to use OpenAI if the package is installed and API key provided; otherwise a fallback heuristic parser is used.
- The app expects the LLM to return strict JSON (an array of objects). The prompt requests JSON-only output and the client attempts to extract JSON from the model output.

