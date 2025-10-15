# llm_utils.py
import os
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

GROQ_MODEL_NAME = "llama-3.1-8b-instant"

def get_llm_response(user_msg: str) -> str:
    """
    Given a user message, return a response from the LLM.
    """
    try:
        prompt_template = "You are a helpful assistant. Reply briefly and helpfully to the user message: {msg}"
        prompt = ChatPromptTemplate.from_template(prompt_template)
        llm = ChatGroq(model_name=GROQ_MODEL_NAME, groq_api_key=os.getenv("GROQ_API_KEY"))
        chain = LLMChain(llm=llm, prompt=prompt)
        reply = chain.run(msg=user_msg)
    except Exception as e:
        reply = f"(LLM error: {e})"
    return reply
