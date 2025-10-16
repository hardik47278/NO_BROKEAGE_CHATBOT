
import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv


from vectorsearch import initialize_vectorstores, semantic_search
from llm_utils import get_llm_response

load_dotenv()  

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


def main():
    st.set_page_config(page_title="Property Search AI", layout="wide")
    st.title("🏠 Smart Property Search Assistant")
    st.markdown("Examples: **3BHK flat in Pune under ₹2 Cr**")

    # Initialize session state
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hi — this chat is for general conversation. Use the main search box for property queries."}
        ]

    
    query = st.text_input("Enter property search query:", key="property_search_input")

    if st.button("Search") and query:
        # Call FastAPI backend
        try:
            resp = requests.post(f"{API_URL}/search", json={"query": query, "top_n": 6})
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.error(f"Could not connect to backend or error occurred: {e}")
            return

        filters = data.get("filters", {})
        results = data.get("results", [])
        summary = data.get("summary", "No summary available.")
        results_df = pd.DataFrame(results)

        
        st.markdown("**Parsed Filters:**")
        st.json({k: v if v is not None else "—" for k, v in filters.items()})

        if results_df.empty:
            st.warning("No exact matches found for your query.")
        else:
            
            st.subheader("🏘️ Top Matching Properties")
            for _, row in results_df.iterrows():
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**{row['ProjectName']}**")
                    st.markdown(f"{row['CityLocality']}")
                    st.markdown(f"**{row['BHK']}** • {row['Price']} • {row['PossessionStatus']}")
                    if row["Amenities"]:
                        st.markdown(f"Amenities: {row['Amenities']}")
                with cols[1]:
                    st.write("")
                st.markdown("---")

            # Semantic search
            vectorstore = initialize_vectorstores(results_df)
            st.divider()
            st.subheader("🔍 Semantically Similar Matches")
            sem_results = semantic_search(vectorstore, query, top_k=5)
            for i, r in enumerate(sem_results, 1):
                st.markdown(f"**{i}.** {r}")

        # Summary from backend
        st.subheader("🧠 Summary")
        st.write(summary)

        # Store search history
        st.session_state.search_history.append({"query": query, "count": len(results_df)})

    
    with st.sidebar:
        st.subheader("Recent Searches")
        if st.session_state.search_history:
            for s in reversed(st.session_state.search_history[-6:]):
                st.write(f"- `{s['query']}` — {s['count']} results")

    # -------------------- General Chat --------------------
    st.markdown("---")
    st.subheader("💬 General chat (not property search)")
    st.markdown("Use this for general conversation. Property queries should go into the search box above.")
    for m in st.session_state.chat_messages:
        role = "You" if m["role"] == "user" else "Bot"
        st.markdown(f"**{role}:** {m['content']}")

    with st.form("chat_form", clear_on_submit=True):
        chat_input = st.text_input("Say something:", placeholder="Ask general questions or say hi")
        chat_submit = st.form_submit_button("Send")

        if chat_submit and chat_input:
            st.session_state.chat_messages.append({"role": "user", "content": chat_input})

            # Modular LLM response
            reply = get_llm_response(chat_input)
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})

    st.caption("Chat replies are produced by the LLM (Groq). No echo fallback is used.")


if __name__ == "__main__":
    main()
