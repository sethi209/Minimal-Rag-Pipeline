# app_ui.py

import streamlit as st
import requests

st.title("Minimal RAG UI")

query = st.text_input("Enter your question:")

if st.button("Submit") and query.strip():
    response = requests.post(
        "http://localhost:8000/query",
        json={"query": query}
    )
    if response.status_code == 200:
        data = response.json()
        st.write("### Answer:")
        st.write(data["answer"])
        st.write("### Retrieved Context Chunks:")
        for doc in data["context_documents"]:
            st.write(f"- **Score**: {doc['score']}")
            st.write(f"**Metadata**: {doc['metadata']}")
            st.write(f"**Summary**: {doc['summary']}")
            st.write("---")
    else:
        st.error(f"Error: {response.text}")
