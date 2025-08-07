# 📦 app.py - Streamlit UI for Insurance Query System

import streamlit as st
import requests
import os
import tempfile
import json
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util

# === Config ===
USE_MOCK_MODE = False
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")


# === Load embedding model ===
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# === Helper: Extract clauses ===
footer_keywords = ["Bajaj Allianz", "Airport Road", "www.bajajallianz.com", "Reg. No.", "GLOBAL HEALTH CARE"]

def is_boilerplate(para):
    return any(kw.lower() in para.lower() for kw in footer_keywords)

def extract_clauses(uploaded_files):
    clauses = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        reader = PdfReader(tmp_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                for para in text.split("\n\n"):
                    para = para.strip()
                    if len(para) > 50 and not is_boilerplate(para):
                        embedding = model.encode(para, convert_to_tensor=True)
                        clauses.append({
                            "text": para,
                            "embedding": embedding,
                            "source": file.name,
                            "page": i + 1
                        })
        os.unlink(tmp_path)
    return clauses

# === Clause retrieval ===
def retrieve(query, clauses, top_k=3):
    query_emb = model.encode(query, convert_to_tensor=True)
    scored = [(c, util.pytorch_cos_sim(query_emb, c['embedding']).item()) for c in clauses]
    top = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
    return [{"text": c[0]['text'], "source": c[0]['source'], "page": c[0]['page']} for c in top]

# === Analyze decision ===
def analyze(query, facts):
    if USE_MOCK_MODE:
        return {
            "decision": "rejected",
            "amount": 0,
            "justification": "Knee surgery not covered under 3-month-old policy.",
            "referenced_clauses": [{ "source": f["source"], "page": f["page"] } for f in facts]
        }
    else:
        context = "\n---\n".join([
            f"From {f['source']} (Page {f['page']}):\n{f['text'][:500]}"
            for f in facts
        ])

        prompt = (
            f"You are an insurance policy assistant. Analyze the claim and return a JSON object with fields: "
            f"decision, amount, justification, and referenced_clauses (with source and page).\n\n"
            f"Query: \"{query}\"\n\n"
            f"Policy Clauses:\n{context}\n\n"
            f"Respond in JSON only."
        )
        print("LLM Prompt:\n", prompt)


        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.3,
                "do_sample": False
            }
        }

        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
                headers=headers,
                json=payload
            )

            output = response.json()

            if isinstance(output, list) and "generated_text" in output[0]:
                generated = output[0]["generated_text"]
                json_start = generated.find("{")
                return json.loads(generated[json_start:])

            return {"error": "Unexpected LLM output", "raw_output": output}

        except Exception as e:
            return {"error": str(e)}


# === Streamlit UI ===
st.set_page_config(page_title="Insurance Decision Assistant")
st.title("🧠 Insurance Query Decision Assistant")

uploaded_files = st.file_uploader("Upload policy documents (PDF)", type=["pdf"], accept_multiple_files=True)
query = st.text_area("Enter your insurance query (e.g. '46M, knee surgery, 3-month-old policy')")

if st.button("Analyze"):
    if not uploaded_files or not query.strip():
        st.warning("Please upload at least one PDF and enter a query.")
    else:
        st.info("🔍 Extracting and analyzing... please wait.")
        clauses = extract_clauses(uploaded_files)
        top = retrieve(query, clauses)

        st.subheader("🔎 Matched Clauses")
        for i, c in enumerate(top):
            with st.expander(f"Clause {i+1} (Page {c['page']} - {c['source']})"):
                st.write(c['text'])

        st.subheader("📋 Decision")
        result = analyze(query, top)
        st.json(result)
