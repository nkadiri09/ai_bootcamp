# Streamlit app for extracting financials from text
# Minimal, provider-agnostic app that calls llm_client.extract_financials

try:
    import streamlit as st
except Exception:
    raise ImportError("Streamlit is required to run this app. Install with `pip install streamlit`.")

import json
from llm_client import extract_financials

st.set_page_config(page_title="Financials Extractor", layout="wide")

st.title("Financial report -> Financials extractor")
st.write("Paste a company financial report (or upload a .txt) and click Extract to parse revenue, EPS and forecast.")

with st.form("input_form"):
    uploaded = st.file_uploader("Upload a text file with the report", type=["txt"], help="Optional: upload a .txt file")
    text_input = st.text_area("Or paste the report text here", height=300)
    col1, col2 = st.columns([1, 1])
    with col1:
        model = st.selectbox("LLM Provider (if installed)", ["auto (builtin fallback)", "openai"], index=0)
    with col2:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    submitted = st.form_submit_button("Extract")

if uploaded and not text_input:
    try:
        text_input = uploaded.read().decode("utf-8")
    except Exception:
        st.error("Failed to read uploaded file. Make sure it's a UTF-8 text file.")

if submitted:
    if not text_input or text_input.strip() == "":
        st.warning("Please paste or upload report text before extracting.")
    else:
        with st.spinner("Calling LLM and parsing results..."):
            result = extract_financials(text_input, provider=("gpt-5.2" if model == "openai" else None), temperature=float(temperature))

        if isinstance(result, dict) and result.get("error"):
            st.error("Failed to extract financials: {}".format(result.get("error")))
            st.subheader("Raw LLM output")
            st.code(result.get("raw", ""))
        else:
            # result should be a list of dicts or a dict
            st.success("Extraction complete")
            try:
                import pandas as pd
                df = pd.DataFrame(result)
                st.subheader("Extracted table")
                st.dataframe(df)
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, file_name="financials.csv", mime="text/csv")
            except Exception:
                st.subheader("Extracted JSON")
                st.json(result)
                st.download_button("Download JSON", json.dumps(result, indent=2), file_name="financials.json", mime="application/json")

            st.subheader("Raw parsed output")
            st.code(json.dumps(result, indent=2))


st.markdown("---")
st.caption("This app uses a local fallback parser if no LLM SDK is installed. To use a real LLM, install the provider SDK (e.g. openai) and set environment variables.")

