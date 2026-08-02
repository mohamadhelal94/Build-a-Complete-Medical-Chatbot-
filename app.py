import os

from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.helper import create_embeddings
from src.prompt import system_prompt


load_dotenv()

app = Flask(__name__)


# Check required API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY was not found in the .env file"
    )

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY was not found in the .env file"
    )


# Load the same embedding model used in store_index.py
embeddings = create_embeddings()


# Connect to the existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings,
)


# Retrieve the top 3 most similar document chunks
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)


# Initialize OpenAI model
chat_model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
    max_tokens=300,
    timeout=30,
    max_retries=2,
)


# Define the system and user prompts
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# Create the document question-answer chain
question_answer_chain = create_stuff_documents_chain(
    chat_model,
    prompt,
)


# Combine the retriever and answer chain into a RAG chain
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
        response = rag_chain.invoke(
            {"input": user_message}
        )

        return response["answer"]

    except Exception:
        app.logger.exception("Chatbot error")

        return (
            "The medical assistant is temporarily unavailable. "
            "Please try again.",
            500,
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True,
    )