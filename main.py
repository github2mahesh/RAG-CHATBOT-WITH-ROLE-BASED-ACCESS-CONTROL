from langchain_community.document_loaders import UnstructuredMarkdownLoader, CSVLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

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

query = "What is the revenue growth?"

results = vector_store.similarity_search(
    query,
    k=5,
    filter={"department": "finance"}  
)

context = "\n\n".join([doc.page_content for doc in results])

messages = [
    (
        "system",
        "Answer the question using only the provided context. say I dont know if dont know the answer",
    ),
    ("human", f"Context:\n{context}\n\nQuestion: {query}"),
]

response = llm.invoke(messages)

print(response.content)
