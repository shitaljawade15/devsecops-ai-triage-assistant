# =====================================================================
# 🖥️ STREAMLIT FRONTEND INTERFACE
# =====================================================================

import os
import requests
import streamlit as st

st.set_page_config(
    page_title="DevSecOps AI Triage Assistant",
    page_icon="🛡️",
    layout="wide"
)

BASE_API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# --- UI Header ---
st.title("🛡️ DevSecOps Automated Triage Copilot")
st.markdown("Query pipeline vulnerabilities, triage third-party risks, and generate targeted remediation commands.")

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I am your DevSecOps Copilot. You can ask general security questions or upload scan reports in the sidebar for contextual triage.",
            "sources": []
        }
    ]

# --- Sidebar: File Upload (Supports All Formats) ---
with st.sidebar:
    st.header("📂 Security Reports & Context")
    st.markdown("Upload scans (`.json`, `.sarif`), checklists (`.pdf`), policies (`.txt`), or threat catalogs (`.csv`).")

    uploaded_file = st.file_uploader(
        "Upload Security Document",
        type=["pdf", "json", "csv", "txt", "log", "yaml", "yml", "sarif"]
    )

    if st.button("📥 Process & Vectorize Document", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Parsing, embedding, and vectorizing content..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    res = requests.post(f"{BASE_API_URL}/upload-file", files=files, timeout=120)

                    if res.ok:
                        data = res.json()
                        st.success(f"Indexed {data['chunks_indexed']} chunks from `{data['filename']}` ({data['file_type']})!")
                    else:
                        st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"Cannot connect to FastAPI backend: {str(e)}")
        else:
            st.warning("Please choose a file to upload first.")

    st.markdown("---")
    st.markdown("### 📋 Release Gate Policy Quick-Guide")
    st.markdown("- **Blocked:** Critical CVEs, plaintext secrets, unapproved license risks")
    st.markdown("- **Passed:** Waived CVEs, unreachable dependencies")

# --- Chat History Display ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("🔍 View Context Sources Used"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {idx}:** {src}")

# --- User Input & Chat Processing ---
user_query = st.chat_input("Ask about vulnerabilities, secrets, or release blockers...")

if user_query:
    # 1. Display user query
    st.session_state["messages"].append({"role": "user", "content": user_query, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Get answer from backend
    with st.chat_message("assistant"):
        with st.spinner("Triaging findings with fine-tuned model..."):
            try:
                res = requests.post(
                    f"{BASE_API_URL}/chat",
                    json={"question": user_query, "top_k": 3},
                    timeout=60
                )
                if res.ok:
                    data = res.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources_used", [])

                    st.markdown(answer)
                    if sources:
                        with st.expander("🔍 View Context Sources Used"):
                            for idx, src in enumerate(sources, 1):
                                st.markdown(f"**Chunk {idx}:** {src}")

                    # Save to state
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"Error from assistant server: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")