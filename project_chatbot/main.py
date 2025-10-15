from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

app = FastAPI()

# Load CSVs
address = pd.read_csv("data/ProjectAddress.csv")
project = pd.read_csv("data/project.csv")
address = pd.read_csv("data/ProjectAddress.csv")
merged = project.merge(address, on="id", how="inner")

# Model setup
llm = ChatGroq(temperature=0, model_name="mixtral-8x7b")

class Query(BaseModel):
    query: str

@app.post("/search")
def search_properties(data: Query):
    query = data.query

    # Use your parser function or regex to extract filters
    filters = {"bhk": "3BHK", "budget": "2 Cr", "city": "Pune"}  # Example placeholder

    # Filter dataset (simple example)
    filtered = merged[
        merged["price"].astype(str).str.contains("1.39", case=False, na=False)
    ].head(3)

    # LLM summary
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize in one helpful sentence the property details:\n{data}"
    )
    chain = summary_prompt | llm
    summary = chain.invoke({"data": filtered.to_dict(orient="records")}).content

    return {
        "filters": filters,
        "properties": filtered.to_dict(orient="records"),
        "summary": summary,
    }
