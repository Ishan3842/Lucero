import os
import jwt
import bcrypt
import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import Optional

# For chat usage
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from pdfminer.high_level import extract_text
from io import BytesIO

from dotenv import load_dotenv

load_dotenv()

# ENV
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGO = os.getenv("JWT_ALGO")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

# DB Setup
engine = create_engine(
    f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}",
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)


Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://your-frontend-url.vercel.app",  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SignupRequest(BaseModel):
    email: str
    password: str
    first_name: str
    last_name: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    message: str


class PDFQARequest(BaseModel):
    question: str
    pdf_text: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/signup")
def signup(req: SignupRequest, db=Depends(get_db)):
    user_exists = db.query(User).filter(User.email == req.email).first()
    if user_exists:
        raise HTTPException(status_code=400, detail="User already exists")
    hashed = bcrypt.hashpw(req.password.encode("utf-8"), bcrypt.gensalt())
    new_user = User(
        email=req.email,
        password_hash=hashed.decode("utf-8"),
        first_name=req.first_name,
        last_name=req.last_name,
    )
    db.add(new_user)
    db.commit()
    return {"message": "Signup successful"}


@app.post("/login")
def login(req: LoginRequest, db=Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not bcrypt.checkpw(
        req.password.encode("utf-8"), user.password_hash.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Use timezone-aware datetime
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "sub": user.id,
        "exp": now_utc + datetime.timedelta(hours=12),
    }

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
    return {
        "access_token": token,
        "user": {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
        },
    }


def decode_jwt(token: str):
    print("Decoding token:", token)
    print("Using secret:", JWT_SECRET)
    try:
        decoded = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        print("Decoded payload:", decoded)
        return decoded
    except Exception as e:
        print("Decode error:", e)
        return None


# @app.get("/chat/sessions")
# def get_sessions(token: str):
#     decoded = decode_jwt(token)
#     if not decoded:
#         raise HTTPException(status_code=401, detail="Invalid token")
#     # Hardcoded sessions
#     return [{"id": 1, "title": "Session One"}, {"id": 2, "title": "Session Two"}]


@app.post("/chat")
def chat_endpoint(req: ChatRequest, token: Optional[str] = None):
    if token:
        decoded = decode_jwt(token)
        if not decoded:
            raise HTTPException(status_code=401, detail="Invalid token")
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name=OPENAI_MODEL,
        temperature=0.3,
        max_tokens=1000,
    )
    template = """
    You are a healthcare-oriented language assistant. 
    You can provide general health information, 
    but you must include a disclaimer that you are not a licensed doctor 
    and cannot diagnose or prescribe treatment.
    Try keeping everything to the point and help the user as much as possible.
    Don't mention that you do not know this, instead try asking follow up questions to understand the issue and solve it. 
    Don't be rude or behave like an AI, behave like a human trying to help. 
    If there is a lot of information that the user needs, split it up into points. 
    if there is a pdf, explain it in points which is easy to understand.
    Once you have split it up into points, give the answer in a good format. 
    Don't put the points in a paragraph, instead put them in a list.
    You respond to health-related questions with empathy and helpfulness.
    
    If the user asks a question for example - "what vaccines do I need?" and if you don't have the previous chat history,
    as to why the user is asking then you can ask follow-up questions to understand what is the issue and then provide the answer.
    
    Example 1:
    User: "I have a headache. What do I do?"
    
    If a user asks a question like this, always try to understand the issue first.
    Ask follow-up questions to understand the issue better. Act like a professional doctor but when it gets too deep, 
    advise them to seek professional help.

    Follow these rules:
        1. If the user's query sounds like it requires urgent care, advise them to seek professional help. Still give your best advice.
        2. Use layman's terms but remain accurate.
        3. Avoid giving definitive diagnoses—always recommend a professional if in doubt.
        
    User query: {user_query}    
    """
    prompt = PromptTemplate(template=template, input_variables=["user_query"])
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(user_query=req.message)
    return {"message": response.strip()}


@app.post("/upload_pdf")
def upload_pdf_endpoint(file: UploadFile = File(...), token: Optional[str] = None):
    if token:
        decoded = decode_jwt(token)
        if not decoded:
            raise HTTPException(status_code=401, detail="Invalid token")
    content = file.file.read()
    text = extract_text(BytesIO(content))
    return {"pdf_text": text}


@app.post("/pdf_qa")
def pdf_qa_endpoint(req: PDFQARequest, token: Optional[str] = None):
    if token:
        decoded = decode_jwt(token)
        if not decoded:
            raise HTTPException(status_code=401, detail="Invalid token")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([req.pdf_text])
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(docs, embeddings)
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0.7
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=db.as_retriever(), return_source_documents=False
    )
    result = qa_chain({"question": req.question, "chat_history": []})
    return {"answer": result["answer"]}


@app.get("/")
def root():
    return {
        "message": "Backend up",
        "status": "success",
        "Docs": "http://localhost:8000/docs",
    }
