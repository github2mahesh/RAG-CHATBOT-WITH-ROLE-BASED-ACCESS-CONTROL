# This script rebuilds chroma_db from the data/ folder.
# Run once locally and in CI/CD pipeline before starting the server.

from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

ROOT = Path(__file__).parent.parent
base_path = str(ROOT / "data")
persist_dir = str(ROOT / "chroma_db")

print("Building vector database from data/ folder...")

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

            all_docs.extend(docs)
            print(f"  Loaded {file} ({dept})")

# Split documents
headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

final_chunks = []
for doc in all_docs:
    if doc.metadata["source_file"].endswith(".csv"):
        final_chunks.append(doc)
    else:
        splits = markdown_splitter.split_text(doc.page_content)
        for split in splits:
            split.metadata.update(doc.metadata)
        split_chunks = text_splitter.split_documents(splits)
        final_chunks.extend(split_chunks)

print(f"Total chunks: {len(final_chunks)}")

# Create embeddings and vector store
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)

print("Creating vector store...")
vector_store = Chroma.from_documents(
    documents=final_chunks,
    embedding=embeddings,
    persist_directory=persist_dir
)

print(f"Done. Vector store saved to {persist_dir}/")