# app.py
import os
import re
import json
import pandas as pd
from textwrap import shorten
import streamlit as st
from dotenv import load_dotenv


load_dotenv()  # read .env

# optional LLM summarizer (Groq)
try:
    from langchain.prompts import ChatPromptTemplate
    from langchain.chains import LLMChain
    from langchain_groq import ChatGroq
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

DATA_PATH = os.path.join("data", "merged_properties.csv")
NORMALIZED_PATH = os.path.join("data", "merged_properties_normalized.csv")
GROQ_MODEL_NAME = "llama-3.1-8b-instant"


# -------------------------- normalization / utils --------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    s = str(text).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^\w\s,.-]', '', s)  # keep commas/dots for readability
    return s


def normalize_df(df):
    df = df.copy()
    for col in ["FullAddress", "ProjectName", "BHK", "PossessionStatus", "Amenities", "slug"]:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(clean_text)
        else:
            df[col] = ""
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce") if "Price" in df.columns else pd.Series([pd.NA]*len(df))
    df["BHK"] = df["BHK"].str.replace(r'\s+', '', regex=True)
    df["PossessionStatus"] = df["PossessionStatus"].str.replace("_", " ").str.strip()
    return df


def format_price(p):
    try:
        if pd.isna(p): return "-"
        p = float(p)
        if p >= 1e7:
            return f"₹{p/1e7:.2f} Cr"
        else:
            return f"₹{p/1e5:.2f} L"
    except:
        return "-"


# -------------------------- parsing + search --------------------------
def parse_filters(query):
    q = query.strip()
    city = re.search(r"\b(Pune|Mumbai|Chennai|Delhi|Hyderabad)\b", q, re.I)
    bhk = re.search(r"(\d+(?:\.\d+)?)\s*BHK", q, re.I)
    budget = re.search(r"(?:under|below|upto|less than)\s*₹?\s*([0-9]*\.?[0-9]+)\s*(?:cr|crore|l|lakhs|lakh)?", q, re.I)
    readiness = re.search(r"(ready[- ]?to[- ]?move|under construction)", q, re.I)
    locality = re.search(r"(?:in|near|at)\s+([A-Za-z0-9, \-]+?)(?:\s+(?:under|below|upto|for|with|near)\b|$)", q, re.I)
    project_name = re.search(r"project\s+([A-Za-z0-9 &\-]+)", q, re.I)

    loc = locality.group(1).strip().lower() if locality else None
    if loc:
        loc = re.sub(r'^\s*(in|near|at)\s+', '', loc)
        loc = re.sub(r'\b(under|below|upto|for|with|near)\b.*$', '', loc).strip()

    return {
        "city": city.group(1).lower() if city else None,
        "bhk": (bhk.group(1).strip() + "bhk").lower() if bhk else None,
        "budget": float(budget.group(1)) if budget else None,
        "readiness": readiness.group(1).lower().replace("-", " ") if readiness else None,
        "locality": loc,
        "project_name": project_name.group(1).strip().lower() if project_name else None,
    }


def extract_top_amenities(amen_str, top_n=3):
    if not amen_str or amen_str in ("nan", "[]", "none"):
        return []
    s = amen_str.strip()
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed][:top_n]
    except Exception:
        parts = [p.strip() for p in re.split(r'[,;|]', s) if p.strip()]
        return parts[:top_n]
    return []


def search_properties(df, filters, top_n=6):
    results = df.copy()
    f_city = filters.get("city")
    f_bhk = (filters.get("bhk") or "").lower().replace(" ", "")
    f_budget = filters.get("budget")
    f_read = (filters.get("readiness") or "").lower()
    f_locality = (filters.get("locality") or "").lower()
    f_proj = (filters.get("project_name") or "").lower()

    if f_city:
        results = results[results["FullAddress"].str.contains(f_city, case=False, na=False)]
    if f_bhk:
        results = results[results["BHK"].str.contains(f_bhk, case=False, na=False)]
    if f_budget is not None:
        max_price = float(f_budget) * 1e7
        results = results[results["Price"].notna() & (results["Price"] <= max_price)]
    if f_read:
        results = results[results["PossessionStatus"].str.contains(f_read, case=False, na=False)]
    if f_locality and f_locality != f_city:
        results = results[results["FullAddress"].str.contains(f_locality, case=False, na=False)]
    if f_proj:
        results = results[results["ProjectName"].str.contains(f_proj, case=False, na=False)]

    display_rows = []
    for _, row in results.head(top_n).iterrows():
        display_rows.append({
            "ProjectName": row.get("ProjectName", "").title(),
            "CityLocality": (row.get("FullAddress") or "").title(),
            "BHK": row.get("BHK", ""),
            "Price": format_price(row.get("Price")),
            "PossessionStatus": (row.get("PossessionStatus") or "").title(),
            "Amenities": ", ".join(extract_top_amenities(row.get("Amenities", ""), top_n=3))
        })
    display_df = pd.DataFrame(display_rows, columns=["ProjectName","CityLocality","BHK","Price","PossessionStatus","Amenities"])
    return results, display_df


# -------------------------- summarization --------------------------
def summarize_with_groq(display_df):
    if display_df.empty:
        return "No properties found for your query."
    if not LLM_AVAILABLE:
        return None
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return None

    lines = []
    for _, r in display_df.iterrows():
        addr_short = shorten(r["CityLocality"], width=90, placeholder="...")
        lines.append(f"{r['ProjectName']} | {r['BHK']} | {r['Price']} | {r['PossessionStatus']} | {addr_short} | Amenities: {r['Amenities'] or '—'}")
    properties_text = "\n".join(lines)

    prompt_template = """
You are a helpful real estate assistant. Given the following list of properties (each line is a project with fields),
write a short (2-4 sentences) factual summary describing the best matches and where they are located.
Ground your summary strictly in the text below. Do NOT add any external information or hallucinations.

Properties:
{properties_text}
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatGroq(model_name=GROQ_MODEL_NAME, groq_api_key=groq_key)
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(properties_text=properties_text)
    return summary.strip()


def summarize_fallback(display_df):
    if display_df.empty:
        return "No properties found for your query."

    count = len(display_df)
    top_projects = display_df["ProjectName"].unique().tolist()[:3]
    poss = sorted([p for p in display_df["PossessionStatus"].unique() if p])
    loc_tokens = []
    for s in display_df["CityLocality"].tolist():
        parts = [p.strip() for p in s.split(',') if p.strip()]
        if parts:
            loc_tokens.append(parts[-2] if len(parts) >= 2 else parts[-1])
    top_locs = pd.Series(loc_tokens).value_counts().head(3).index.tolist() if loc_tokens else []
    amenities_series = []
    for a in display_df["Amenities"].tolist():
        if a and a != "nan":
            amenities_series.extend([x.strip() for x in a.split(',') if x.strip()])
    top_amen = pd.Series(amenities_series).value_counts().head(3).index.tolist() if amenities_series else []

    sentences = []
    sentences.append(f"I found {count} matching properties — top projects include: {', '.join(top_projects)}.")
    if top_locs:
        sentences.append(f"Most are located around: {', '.join(top_locs[:3])}.")
    if poss:
        sentences.append(f"Possession status includes: {', '.join(poss)}.")
    if top_amen:
        sentences.append(f"Common amenities listed: {', '.join(top_amen[:3])}.")
    return " ".join(sentences[:4])


# -------------------------- Main property search & chat UI --------------------------
def main():
    st.set_page_config(page_title="Property Search AI", layout="wide")
    if not os.path.exists(DATA_PATH):
        st.error(f"Error: {DATA_PATH} not found. Put your merged CSV at {DATA_PATH}")
        return

    raw = pd.read_csv(DATA_PATH)
    df = normalize_df(raw)
    df.to_csv(NORMALIZED_PATH, index=False)

    # initialize session state containers
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role": "assistant", "content": "Hi — this chat is for general conversation. Use the main search box for property queries."}]

    # -------------------------- Property search UI --------------------------
    st.title("🏠 Smart Property Search Assistant")
    st.markdown("Examples: **3BHK flat in Pune under ₹2 Cr**")
    query = st.text_input("Enter property search query:", key="property_search_input")
    if st.button("Search", key="property_search_button") and query:
        filters = parse_filters(query)
        st.markdown("**Parsed Filters:**")
        st.json({k: v if v is not None else "—" for k,v in filters.items()})

        results_raw, display_df = search_properties(df, filters, top_n=6)

        if display_df.empty:
            st.warning("No exact matches found for your query.")

            # city/BHK debug tables
            if filters.get("city"):
                city = filters["city"]
                sample = df[df["FullAddress"].str.contains(city, case=False, na=False)][["FullAddress","BHK","Price"]].head(10)
                st.markdown(f"Rows matching city '{city}': {len(sample)} (showing up to 10)")
                if not sample.empty: st.table(sample)
            if filters.get("bhk"):
                sample_bhk = df[df["BHK"].str.contains(filters['bhk'].lower().replace(" ",""), case=False, na=False)][["FullAddress","BHK","Price"]].head(10)
                st.markdown(f"Rows matching BHK '{filters['bhk']}': {len(sample_bhk)} (showing up to 10)")
                if not sample_bhk.empty: st.table(sample_bhk)

            # looser search
            loose = filters.copy()
            loose["locality"] = None
            loose["budget"] = None
            _, loose_display = search_properties(df, loose, top_n=6)
            if not loose_display.empty:
                st.markdown("Looser search (ignoring locality & budget) found:")
                st.dataframe(loose_display, use_container_width=True)

            summary = summarize_fallback(loose_display) if not loose_display.empty else "No properties found for your query."
            st.subheader("🧠 Summary")
            st.write(summary)

            st.session_state.search_history.append({"query": query, "filters": filters, "count": len(loose_display)})
        else:
            # show results as cards
            st.subheader("🏘️ Top Matching Properties")
            for idx, row in display_df.iterrows():
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**{row['ProjectName']}**")
                    st.markdown(f"{row['CityLocality']}")
                    st.markdown(f"**{row['BHK']}** • {row['Price']} • {row['PossessionStatus']}")
                    if row["Amenities"]:
                        st.markdown(f"Amenities: {row['Amenities']}")
                with cols[1]:
                    st.write("")  # reserved spot
                st.markdown("---")

            # summary
            summary = summarize_with_groq(display_df)
            if summary is None:
                summary = summarize_fallback(display_df)
            st.subheader("🧠 Summary")
            st.write(summary)

            st.session_state.search_history.append({"query": query, "filters": filters, "count": len(display_df)})

    # -------------------------- Sidebar: recent searches --------------------------
    with st.sidebar:
        st.subheader("Recent Searches")
        if st.session_state.search_history:
            for s in reversed(st.session_state.search_history[-6:]):
                st.write(f"- `{s['query']}` — {s['count']} results")

    # -------------------------- Chat UI --------------------------
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
            reply = None
            if LLM_AVAILABLE and os.getenv("GROQ_API_KEY"):
                try:
                    prompt_template = "You are a helpful assistant. Reply briefly and helpfully to the user message: {msg}"
                    prompt = ChatPromptTemplate.from_template(prompt_template)
                    llm = ChatGroq(model_name=GROQ_MODEL_NAME, groq_api_key=os.getenv("GROQ_API_KEY"))
                    chain = LLMChain(llm=llm, prompt=prompt)
                    reply = chain.run(msg=chat_input)
                except Exception:
                    reply = f"(LLM error) Echo: {chat_input}"
            else:
                reply = f"Echo: {chat_input}"
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})

    st.caption("Chat replies are produced by an LLM only if langchain_groq is installed and GROQ_API_KEY is set; otherwise replies are echoed.")


if __name__ == "__main__":
    main()

