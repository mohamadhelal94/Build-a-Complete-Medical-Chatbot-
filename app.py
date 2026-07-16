from flask import Flask, render_template, request
from dotenv import load_dotenv
from src.helper import create_embeddings
from src.prompt import system_prompt
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import (create_stuff_documents_chain,)
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

# Create app 
app = Flask(__name__)


# Check Pinecone API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY was not found in the .env file"
    )

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# Load the same local embedding model used in store_index.py
embeddings = create_embeddings()


# Connect to the existing Pinecone index.

docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings,
)

# create a retriever to fitch the top 3 most similar documents from the Pinecone index
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)


# initialize local Ollama model
chat_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

# Define the system and user prompts for the chatbot
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# create the doc question-answer chain
question_answer_chain = create_stuff_documents_chain(
    chat_model,
    prompt,
)

# Combine the retriever and question-answer chain into a RAG chain
rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain,
)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_message = request.form.get("msg", "").strip()

    if not user_message:
        return "Please enter a medical question.", 400

    try:
        response = rag_chain.invoke({"input": user_message})

        return response["answer"]

    except Exception as error:
        app.logger.exception("Chatbot error")
        return f"An error occurred: {error}", 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True,
    )