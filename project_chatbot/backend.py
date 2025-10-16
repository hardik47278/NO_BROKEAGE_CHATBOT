# backend.py
import os
import re
import json
from typing import Optional, Dict, Any
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# -------------------- Paths --------------------
DATA_PATH = os.path.join("data", "merged_properties.csv")
NORMALIZED_PATH = os.path.join("data", "merged_properties_normalized.csv")


app = FastAPI(title="Property Search API", version="1.0")


def clean_text(text: str) -> str:   #PREPROCESSING CSV 
    if pd.isna(text):
        return ""
    s = str(text).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^\w\s,.-]', '', s)  
    return s

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:   #NORMALIZING THE DATAFRAME
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

def format_price(p):    #formatting the price to have common units
    try:
        if pd.isna(p): return "-"
        p = float(p)
        if p >= 1e7:
            return f"₹{p/1e7:.2f} Cr"
        else:
            return f"₹{p/1e5:.2f} L"
    except:
        return "-"

def extract_top_amenities(amen_str, top_n=3):    #extract ammentities
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

def summarize_fallback(display_df: pd.DataFrame) -> str:      #fallback mechanism
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

# -------------------- Load & normalize data --------------------
df = pd.read_csv(DATA_PATH)
df = normalize_df(df)
df.to_csv(NORMALIZED_PATH, index=False)


def parse_filters(query: str) -> Dict[str, Optional[Any]]:  #QUERY PARSING FILTERS USING REGEX
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


def search_properties(filters: Dict[str, Optional[Any]], top_n=6):  #PROPERTY SEARCH
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
    summary = summarize_fallback(display_df)  # <-- summary included in backend
    return results, display_df, summary


class QueryRequest(BaseModel):
    query: str
    top_n: Optional[int] = 6

@app.post("/search")   #ENDPOINTS
def search_properties_endpoint(req: QueryRequest):
    filters = parse_filters(req.query)
    _, display_df, summary = search_properties(filters, top_n=req.top_n)
    return {
        "filters": filters,
        "results_count": len(display_df),
        "results": display_df.to_dict(orient="records"),
        "summary": summary  # <-- new field
    }

@app.get("/")   #ENDPOINTS
def root():
    return {"message": "Property Search API is running!"}
