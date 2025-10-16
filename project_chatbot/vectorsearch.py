
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

def initialize_vectorstores(df: pd.DataFrame) -> InMemoryVectorStore:
    """
    Initializes a vectorstore from a DataFrame for semantic search.
    """
    df = df.copy()
    df["content"] = df.apply(
        lambda x: f"{x.get('ProjectName', '')}, {x.get('CityLocality', '')}, "
                  f"{x.get('BHK', '')}, {x.get('Price', '')}, "
                  f"Status: {x.get('PossessionStatus', '')}, "
                  f"Amenities: {x.get('Amenities', '')}", axis=1
    )
    loader = DataFrameLoader(df, page_content_column="content")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = InMemoryVectorStore.from_documents(docs, embedding=embeddings)
    return vectorstore

def semantic_search(vectorstore: InMemoryVectorStore, query: str, top_k: int = 5) -> list:
    """
    Perform semantic search on the vectorstore and return top_k results.
    """
    docs = vectorstore.similarity_search(query, k=top_k)
    return [doc.page_content for doc in docs]
