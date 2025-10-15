# app.py
import os
import re
import json
import pandas as pd
from textwrap import shorten
from dotenv import load_dotenv

load_dotenv()  # load .env automatically

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

# --------------------------
# Normalization functions
# --------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    s = str(text).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^\w\s,.-]', '', s)  # keep commas/dots
    return s

def normalize_df(df):
    df = df.copy()
    for col in ["FullAddress", "ProjectName", "BHK", "PossessionStatus", "Amenities"]:
        df[col] = df[col].astype(str).apply(clean_text) if col in df.columns else ""
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

# --------------------------
# Query parsing
# --------------------------
def parse_filters(query):
    q = query.lower().strip()

    city = re.search(r"\b(pune|mumbai|chennai|delhi|hyderabad)\b", q)
    bhk = re.search(r"(\d+)\s*bhk", q)
    budget = re.search(r"(?:under|below|upto|less than)\s*₹?\s*([0-9]*\.?[0-9]+)\s*(l|lakhs?|cr|crore)?", q)
    readiness = re.search(r"(ready[- ]to[- ]move|under construction)", q)
    locality = re.search(r"(?:in|near|at)\s+([a-z0-9, \-]+?)(?:\s+(?:under|below|upto|for|with|near)\b|$)", q)
    project_name = re.search(r"project\s+([a-z0-9 &\-]+)", q)

    # Process budget
    budget_val = None
    if budget:
        num = float(budget.group(1))
        unit = budget.group(2)
        if unit and "cr" in unit:
            num *= 1e7
        elif unit and "l" in unit:
            num *= 1e5
        budget_val = num

    loc = locality.group(1).strip() if locality else None
    if loc:
        loc = re.sub(r'^\s*(in|near|at)\s+', '', loc)
        loc = re.sub(r'\b(under|below|upto|for|with|near)\b.*$', '', loc).strip()

    return {
        "city": city.group(1) if city else None,
        "bhk": f"{bhk.group(1)}bhk" if bhk else None,
        "budget": budget_val,
        "readiness": readiness.group(1).replace("-", " ") if readiness else None,
        "locality": loc,
        "project_name": project_name.group(1).strip() if project_name else None
    }

def extract_top_amenities(amen_str, top_n=3):
    if not amen_str or amen_str in ("nan", "[]", "none"):
        return []
    try:
        parsed = json.loads(amen_str)
        if isinstance(parsed, list):
            return [str(x) for x in parsed][:top_n]
    except:
        parts = [p.strip() for p in re.split(r'[,;|]', amen_str) if p.strip()]
        return parts[:top_n]
    return []

# --------------------------
# Property search
# --------------------------
def search_properties(df, filters, top_n=5):
    results = df.copy()

    f_city = filters.get("city")
    f_bhk = (filters.get("bhk") or "").lower().replace(" ", "")
    f_budget = filters.get("budget")
    f_read = (filters.get("readiness") or "").lower()
    f_locality = (filters.get("locality") or "").lower()
    f_proj = (filters.get("project_name") or "").lower()

    if f_city:
        results = results[results["FullAddress"].str.contains(f_city, na=False)]
    if f_bhk:
        results = results[results["BHK"].str.contains(f_bhk, na=False)]
    if f_budget is not None:
        results = results[results["Price"].notna() & (results["Price"] <= f_budget)]
    if f_read:
        results = results[results["PossessionStatus"].str.contains(f_read, na=False)]
    if f_locality and f_locality != f_city:
        results = results[results["FullAddress"].str.contains(f_locality, na=False)]
    if f_proj:
        results = results[results["ProjectName"].str.contains(f_proj, na=False)]

    display_rows = []
    for _, row in results.head(top_n).iterrows():
        display_rows.append({
            "ProjectName": row.get("ProjectName", "").title(),
            "BHK": row.get("BHK", ""),
            "Price": format_price(row.get("Price")),
            "PossessionStatus": (row.get("PossessionStatus") or "").title(),
            "FullAddress": row.get("FullAddress", ""),
            "Amenities": ", ".join(extract_top_amenities(row.get("Amenities", ""), top_n=3))
        })
    display_df = pd.DataFrame(display_rows, columns=["ProjectName","BHK","Price","PossessionStatus","FullAddress","Amenities"])
    return results, display_df

# --------------------------
# LLM summarization
# --------------------------
def summarize_with_groq(display_df):
    if display_df.empty:
        return "No properties found for your query."
    if not LLM_AVAILABLE:
        return "LLM summarization not available."

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return "GROQ_API_KEY not set."

    lines = []
    for _, r in display_df.iterrows():
        addr_short = shorten(r["FullAddress"], width=90, placeholder="...")
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

# --------------------------
# Main interactive loop
# --------------------------
def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    raw = pd.read_csv(DATA_PATH)
    normalized = normalize_df(raw)
    normalized.to_csv(NORMALIZED_PATH, index=False)
    print(f"Loaded {len(raw):,} rows; normalized CSV saved to: {NORMALIZED_PATH}")

    print("\nExample queries: '3BHK flat in Pune under ₹1.2 Cr', '2BHK ready to move in Pune under ₹80 L'\nType 'exit' to quit.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not query or query.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        filters = parse_filters(query)
        print("\nParsed Filters:", filters)
        results_raw, display_df = search_properties(normalized, filters, top_n=6)

        if display_df.empty:
            print("\nTop Results:\n Empty DataFrame\n")
            loose_filters = filters.copy()
            loose_filters["locality"] = None
            loose_filters["budget"] = None
            _, loose_display = search_properties(normalized, loose_filters, top_n=6)
            if not loose_display.empty:
                print("Looser search (ignoring locality & budget):")
                print(loose_display.to_string(index=False))
            else:
                print("No properties found in looser search.\n")
            continue

        print("\nTop Results:")
        print(display_df.to_string(index=False))
        summary = summarize_with_groq(display_df)
        print("\nSummary:\n", summary, "\n")

if __name__ == "__main__":
    main()
