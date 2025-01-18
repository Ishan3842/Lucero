import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


def pdf_qa_chain(question: str, pdf_text: str) -> str:
    # 1. Split PDF text into smaller chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([pdf_text])

    # 2. Create embeddings + vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(docs, embeddings)

    # 3. Build the ConversationalRetrievalQA chain
    llm = ChatOpenAI(
        temperature=0.7, openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=db.as_retriever(), return_source_documents=False
    )

    # 4. Ask question
    result = qa_chain({"question": question, "chat_history": []})
    return result["answer"]
