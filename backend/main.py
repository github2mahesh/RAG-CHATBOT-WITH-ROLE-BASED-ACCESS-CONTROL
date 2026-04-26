from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Dict, List
from pydantic import BaseModel
from guardrails import check_input, check_output

base_path = "data"

all_docs = []

for dept in os.listdir(base_path):
    dept_path = os.path.join(base_path, dept)

    if os.path.isdir(dept_path):
        for file in os.listdir(dept_path):
            file_path = os.path.join(dept_path, file)

            if file.endswith(".md"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                docs = [Document(
                    page_content=content,
                    metadata={"department": dept, "source_file": file}
                )]

            elif file.endswith(".csv"):
                loader = CSVLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["department"] = dept
                    doc.metadata["source_file"] = file

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
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)

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
app = FastAPI()
security = HTTPBasic()

llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY")
    )
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


# ── Message model (one turn in history) ──────────────────────────
class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str

# ── Updated request body ──────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []   # ← new, defaults to empty list
                                  # so old callers don't break

# ─── ENDPOINTS ────────────────────────────────────────────────
@app.get("/login")
def login(user=Depends(authenticate)):
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}

def rewrite_query(query: str, history: list[Message]) -> str:
    """
    If there's no history, the query is already standalone — return as is.
    If there IS history, ask the LLM to rewrite the query so it makes
    sense without needing to read the conversation.
    """
    if not history:
        return query

    # Build a readable transcript of the last 6 turns (3 exchanges)
    # We cap it — sending 50 turns of history to rewrite a query is wasteful
    recent = history[-6:]
    transcript = "\n".join(
        f"{msg.role.upper()}: {msg.content}"
        for msg in recent
    )

    rewrite_prompt = [
        ("system",
         "You are a query rewriter. Given a conversation and a follow-up question, "
         "rewrite the follow-up into a single standalone question that contains all "
         "necessary context from the conversation. "
         "Output ONLY the rewritten question. No explanation, no prefix, no quotes."),
        ("human",
         f"Conversation:\n{transcript}\n\n"
         f"Follow-up question: {query}\n\n"
         f"Standalone question:"),
    ]

    result = llm.invoke(rewrite_prompt)
    rewritten = result.content.strip()

    # Safety net — if rewriter returns something empty, fall back to original
    return rewritten if rewritten else query

@app.post("/chat")
def chat(body: ChatRequest, user=Depends(authenticate)):
    role      = user["role"]
    query     = body.message
    history = body.history 
    allowed   = ROLE_DEPT_MAP.get(role)   # None = no restriction

    input_check = check_input(query)
    if not input_check.passed:
        return {
            "answer":  input_check.reason,
            "sources": [],
            "blocked": True,           # flag so Streamlit can style it differently
        }
    # ── Rewrite query using history ───────────────────────────────
    standalone_query = rewrite_query(query, history)

    # Build ChromaDB filter
    if allowed is None:
        # c-level: no filter, search all documents
        results = vector_store.similarity_search(standalone_query, k=5)        
    elif len(allowed) == 1:
        # Single department: simple equality filter
        results = vector_store.similarity_search(
            standalone_query, k=5,
            filter={"department": allowed[0]}
        )
    else:
        # Multiple departments: $in operator
        results = vector_store.similarity_search(
            standalone_query, k=5,
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

    history_text = ""                                                      
    if history:                                                            
        recent = history[-6:]                                              
        history_text = "\n".join(                                         
            f"{msg.role.upper()}: {msg.content}"                          
            for msg in recent                                             
        )

    # Prompt
    messages = [
    ("system",
     "You are a company assistant. Answer ONLY using the provided context. "
     "If the answer is not in the context, say 'I don't know'. "
     f"You have access to {role} data only. If asked about other departments, say you don't have access. "
     "Be concise and factual. Greetings like hi/hello can be responded to normally."),
    ("human", 
    f"Context:\n{context}\n\n"
    f"Conversation so far:\n{history_text}\n\n"
    f"Question: {query}"),
    ]

    response = llm.invoke(messages)
    raw_answer = response.content

    # ── OUTPUT GUARDRAIL ──────────────────────────────────────────
    output_check = check_output(raw_answer)
    final_answer = output_check.cleaned_text   # redacted if PII found, original otherwise

    return {
        "answer":   final_answer,
        "sources":  sources,
        "blocked":  False,
        "redacted": not output_check.passed,   # True if something was redacted
    }

