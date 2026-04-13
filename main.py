from langchain_community.document_loaders import UnstructuredMarkdownLoader, CSVLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Dict, List
from pydantic import BaseModel

base_path = "data"

all_docs = []

for dept in os.listdir(base_path):
    dept_path = os.path.join(base_path, dept)

    if os.path.isdir(dept_path):
        for file in os.listdir(dept_path):
            file_path = os.path.join(dept_path, file)

            if file.endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
                docs = loader.load()

            elif file.endswith(".csv"):
                loader = CSVLoader(file_path)
                docs = loader.load()

            else:
                continue

            # ✅ Attach metadata
            for doc in docs:
                doc.metadata["department"] = dept
                doc.metadata["source_file"] = file

            all_docs.extend(docs)


headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

final_chunks = []

for doc in all_docs:

    # If CSV → keep as is
    if doc.metadata["source_file"].endswith(".csv"):
        final_chunks.append(doc)

    # If Markdown → split properly
    else:
        splits = markdown_splitter.split_text(doc.page_content)

        for split in splits:
            split.metadata.update(doc.metadata)

        split_chunks = text_splitter.split_documents(splits)
        final_chunks.extend(split_chunks)

# create embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

persist_dir = "chroma_db"

if not os.path.exists(persist_dir):
    print("Creating new vector database...")
    
    vector_store = Chroma.from_documents(
        documents=final_chunks,   
        embedding=embeddings,
        persist_directory=persist_dir
    )

else:
    print("Loading existing vector database...")
    
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

load_dotenv()

llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY")
    )

security = HTTPBasic()
# ─── USER DATABASE ────────────────────────────────────────────
users_db: Dict[str, Dict[str, str]] = {
    "Tony":    {"password": "password123", "role": "engineering"},
    "Bruce":   {"password": "securepass",  "role": "marketing"},
    "Sam":     {"password": "financepass", "role": "finance"},
    "Peter":   {"password": "pete123",     "role": "engineering"},
    "Sid":     {"password": "sidpass123",  "role": "marketing"},
    "Natasha": {"password": "hrpass123",   "role": "hr"},
    "Nick":    {"password": "ceopass",     "role": "c-level"},
    "Happy":   {"password": "emppass",     "role": "employee"},
}



# ─── ROLE → DEPARTMENT MAPPING ────────────────────────────────
# Each role defines which ChromaDB `department` metadata values it can query.
# None means no filter = access everything (c-level).
ROLE_DEPT_MAP: Dict[str, list | None] = {
    "finance":     ["finance"],
    "marketing":   ["marketing"],
    "hr":          ["hr"],
    "engineering": ["engineering"],
    "c-level":     None,               # no filter = all departments
    "employee":    ["general"],        # general = policies, FAQs, events
}

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    user = users_db.get(credentials.username)
    if not user or user["password"] != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": credentials.username, "role": user["role"]}

app = FastAPI()

# ─── REQUEST BODY ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

# ─── ENDPOINTS ────────────────────────────────────────────────
@app.get("/login")
def login(user=Depends(authenticate)):
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}


@app.post("/chat")
def chat(body: ChatRequest, user=Depends(authenticate)):
    role      = user["role"]
    query     = body.message
    allowed   = ROLE_DEPT_MAP.get(role)   # None = no restriction

    # Build ChromaDB filter
    if allowed is None:
        # c-level: no filter, search all documents
        results = vector_store.similarity_search(query, k=5)
    elif len(allowed) == 1:
        # Single department: simple equality filter
        results = vector_store.similarity_search(
            query, k=5,
            filter={"department": allowed[0]}
        )
    else:
        # Multiple departments: $in operator
        results = vector_store.similarity_search(
            query, k=5,
            filter={"department": {"$in": allowed}}
        )

    if not results:
        return {
            "answer": "I don't have any relevant information for your query.",
            "sources": []
        }

    # Build context + collect sources
    context = "\n\n".join([doc.page_content for doc in results])
    sources = list({
        f"{doc.metadata.get('department', '?')} / {doc.metadata.get('source_file', '?')}"
        for doc in results
    })

    # Prompt
    messages = [
        ("system",
         "You are a company assistant. Answer ONLY using the provided context. "
         "If the answer is not in the context, say 'I don't know'. "
         "Be concise and factual."),
        ("human", f"Context:\n{context}\n\nQuestion: {query}"),
    ]

    response = llm.invoke(messages)

    return {
        "answer":  response.content,
        "sources": sources,         
        "role":    role,
    }

