STEPS TO RUN 

1.python -m venv v
2..\v\Scripts\activate (activate the script)
3.cd project_chatbot
4.Make .env GROQ_API_KEY = ""
4.uvicorn backend:app --reload
5.cd project_chatbot(Another terminal)
6.streamlit run ui.py
