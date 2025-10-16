STEPS TO RUN 

1.python -m venv v
2..\v\Scripts\activate (activate the script)
3.cd project_chatbot
4.Make .env GROQ_API_KEY = ""
4.uvicorn backend:app --reload
5.cd project_chatbot(Another terminal)
6.streamlit run ui.py

APPLICATION LINK-https://huggingface.co/spaces/hardik1247/NO-BROKEAGE-CHATBOT

PIPELINE-USED REGEX FILTERS TO PARSE USER QUERY
SUMMARIZATION PIPELINE USING LLM CHAT GROQ  META LLAMA 3.1 M MODEL
USED SEMANTIC SEARCH USING HUGGINGFACE EMBEDDING MODEL SENTENCE-TRANSFORMERS AND VECTORSTORE INMEMORY-VECTORSTORE
CHAT MECHANISM ALONG WITH PROPERTY SEARCH FEATURE INTEGRATED

DEMO LINK-[![Watch the video](https://img.youtube.com/vi/cH7IBHWinoI/0.jpg)](https://youtu.be/cH7IBHWinoI)








