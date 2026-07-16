from dotenv import load_dotenv
import os
from src.helper import load_pdf_from_directory, filter_to_minimal_docs, text_split, create_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
print("1. Loading PDFs...")


extracted_data = load_pdf_from_directory(
    "/Users/mosalah/Build-a-Complete-Medical-Chatbot-/data"
)
print("Pages loaded:", len(extracted_data))
print("2. Filtering documents...")

minimal_docs = filter_to_minimal_docs(extracted_data)
print("Minimal documents:", len(minimal_docs))

text_chunk = text_split(minimal_docs)

print("3. Splitting documents...")


embeddings = create_embeddings()


pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

# Hämta alla befintliga index
existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embeddings,
    index_name=index_name
)

from dotenv import load_dotenv
import os
from src.helper import (
    load_pdf_from_directory,
    filter_to_minimal_docs,
    text_split,
    create_embeddings,
)
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY was not found in the .env file")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


print("1. Loading PDFs...")

extracted_data = load_pdf_from_directory(
    "/Users/mosalah/Build-a-Complete-Medical-Chatbot-/data"
)

print("Pages loaded:", len(extracted_data))


print("2. Filtering documents...")

minimal_docs = filter_to_minimal_docs(extracted_data)

print("Minimal documents:", len(minimal_docs))


print("3. Splitting documents...")

text_chunk = text_split(minimal_docs)

print("Text chunks:", len(text_chunk))


print("4. Loading embedding model...")

embeddings = create_embeddings()

print("Embedding model loaded")


pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

existing_indexes = pc.list_indexes().names()

import time

index_name = "medical-chatbot"

existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    print("Creating Pinecone index...")

    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    time.sleep(10)
else:
    print("Pinecone index already exists.")

index = pc.Index(index_name)

print("7. Upload completed")


print("8. Checking Pinecone index...")

stats = index.describe_index_stats()

print(stats)

print("6. Uploading chunks to Pinecone...")

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embeddings,
    index_name=index_name
)

print("7. Upload completed")


print("8. Checking Pinecone index...")

stats = index.describe_index_stats()

print(stats)