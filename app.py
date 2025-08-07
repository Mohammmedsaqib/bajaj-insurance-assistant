import os
import streamlit as st
import requests
import json
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader

# === Configuration ===
USE_MOCK_MODE = False
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

st.set_page_config(page_title="Insurance Query Assistant", layout="wide")
st.title("🧠 Insurance Query Decision Assistant")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text:
                texts.append((i + 1, text.strip()))
        except:
            continue
    return texts

def embed_texts(texts):
    embeddings = model.encode([text for (_, text) in texts], convert_to_tensor=True)
    return embeddings

def search_clauses(query, pdf_data, top_k=3):
    all_texts = []
    metadata = []

    for filename, content in pdf_data.items():
        for page_num, text in content:
            all_texts.append(text)
            metadata.append((filename, page_num))

    clause_embeddings = model.encode(all_texts, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)

    hits = util.semantic_search(query_embedding, clause_embeddings, top_k=top_k)[0]

    top_clauses = []
    for hit in hits:
        idx = hit['corpus_id']
        score = hit['score']
        source, page = metadata[idx]
        top_clauses.append({
            "text": all_texts[idx],
            "score": float(score),
            "source": source,
            "page": page
        })

    return top_clauses

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

        print("🧠 LLM Prompt:\n", prompt)

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
                "https://api-inference.huggingface.co/models/google/flan-t5-base",
                headers=headers,
                json=payload
            )


            print("🔁 Raw HTTP status:", response.status_code)
            print("🔁 Raw response text:", response.text)
            output = response.json()
            if isinstance(output, list) and "generated_text" in output[0]:
                generated = output[0]["generated_text"]
                json_start = generated.find("{")
                try:
                    return json.loads(generated[json_start:])
                except json.JSONDecodeError as e:
                    return {
                        "error": f"Failed to parse LLM output: {str(e)}",
                        "raw_output": generated
                    }

            return {
                "error": "Unexpected LLM output structure",
                "raw_output": output
            }

        except Exception as e:
            return {
                "error": str(e)
            }

# === Streamlit UI ===

st.sidebar.header("Upload Policy Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

query = st.text_input("Enter Insurance Query (e.g., '46M, knee surgery, Pune, 3-month policy')")

if uploaded_files and query:
    with st.spinner("📄 Reading documents..."):
        pdf_data = {}
        for file in uploaded_files:
            name = file.name
            text = extract_text_from_pdf(file)
            pdf_data[name] = text

    with st.spinner("🔎 Retrieving relevant clauses..."):
        clauses = search_clauses(query, pdf_data, top_k=3)

    st.subheader("🔹 Top Matched Clauses:")
    for c in clauses:
        st.markdown(f"**Clause (from {c['source']} - page {c['page']}):**")
        st.code(c["text"][:800] + ("..." if len(c["text"]) > 800 else ""), language="markdown")

    with st.spinner("🧠 Analyzing with LLM..."):
        result = analyze(query, clauses)

    st.subheader("✅ Decision Result")
    st.json(result)

elif not uploaded_files:
    st.info("⬅️ Please upload one or more insurance policy PDFs.")
elif not query:
    st.info("✏️ Enter a natural language insurance query.")
