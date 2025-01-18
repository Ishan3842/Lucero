# back-end/chains/chat_chain.py
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


def chat_chain(message: str) -> str:
    """
    Simple chain for a general healthcare Q&A with an LLM.
    """
    template = (
        "You are a helpful healthcare assistant. "
        "Answer the user's questions about health responsibly. "
        "Do not be rude or provide incorrect information. "
        "Be helpful and informative. "
        "Don't mention that you do not know this, instead try asking follow up questions to understand the issue and solve it. "
        "Don't be rude or behave like an AI, behave like a human trying to help. "
        "Try to help the user as much as possible. "
        "Be detailed in your reponses. "
    )

    prompt = PromptTemplate(template=template, input_variables=["user_query"])

    llm = ChatOpenAI(
        temperature=0.7, openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain
    response = chain.run(user_query=message)
    return response.strip()
