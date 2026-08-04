# 🩺 Medical Chatbot using RAG, GPT-4o mini, LangChain & Pinecone

A production-ready AI-powered Medical Chatbot built with **Retrieval-Augmented Generation (RAG)**. The application retrieves relevant medical knowledge from a Pinecone vector database and generates accurate, context-aware responses using **OpenAI GPT-4o mini**.

The project is fully containerized with Docker and deployed on **AWS EC2** using **Amazon ECR** and **GitHub Actions** for automated CI/CD.

---

## 🚀 Live Architecture

```text
                User
                  │
                  ▼
          Flask Web Application
                  │
                  ▼
          LangChain RAG Pipeline
                  │
      ┌───────────┴───────────┐
      │                       │
      ▼                       ▼
Pinecone Vector DB      GPT-4o mini
(HuggingFace Embeddings)   (OpenAI)
      │                       │
      └───────────┬───────────┘
                  ▼
          Medical Response
```

---

# ✨ Features

- AI-powered medical question answering
- Retrieval-Augmented Generation (RAG)
- GPT-4o mini integration
- Pinecone vector database
- Hugging Face sentence-transformer embeddings
- Semantic similarity search
- LangChain retrieval pipeline
- Flask web interface
- Docker containerization
- AWS EC2 deployment
- Amazon ECR container registry
- GitHub Actions CI/CD
- Production deployment with Gunicorn

---

# 🛠 Tech Stack

### Backend

- Python 3.10
- Flask
- Gunicorn

### AI / LLM

- OpenAI GPT-4o mini
- LangChain
- Hugging Face Embeddings
- sentence-transformers/all-MiniLM-L6-v2

### Vector Database

- Pinecone

### Cloud

- AWS EC2
- Amazon ECR

### DevOps

- Docker
- GitHub Actions

---

# 📂 Project Structure

```
Build-a-Complete-Medical-Chatbot
│
├── app.py
├── store_index.py
├── requirements.txt
├── Dockerfile
├── .github/
│   └── workflows/
├── src/
│   ├── helper.py
│   └── prompt.py
├── templates/
│   └── chat.html
├── static/
│   └── style.css
├── data/
└── README.md
```

---

# ⚙️ Installation

## Clone the repository

```bash
git clone https://github.com/mohamadhelal94/Build-a-Complete-Medical-Chatbot-.git
```

```bash
cd Build-a-Complete-Medical-Chatbot-
```

---

## Create Conda Environment

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Create a `.env` file

```env
PINECONE_API_KEY=your_pinecone_api_key

OPENAI_API_KEY=your_openai_api_key

OPENAI_MODEL=gpt-4o-mini
```

---

## Build the Vector Database

```bash
python store_index.py
```

---

## Run the Application

```bash
python app.py
```

Open:

```
http://localhost:8000
```

---

# 🧠 How the RAG Pipeline Works

1. User submits a medical question.
2. The question is converted into embeddings.
3. Pinecone retrieves the most relevant document chunks.
4. Retrieved context is combined with the user question.
5. GPT-4o mini generates a grounded response.
6. The chatbot returns the final answer.

---

# ☁ AWS Deployment

The project is deployed using Docker on AWS.

Deployment pipeline:

```
GitHub
    │
    ▼
GitHub Actions
    │
    ▼
Docker Build
    │
    ▼
Amazon ECR
    │
    ▼
AWS EC2
    │
    ▼
Gunicorn + Flask
```

---

# 🔄 CI/CD

GitHub Actions automatically:

- Builds the Docker image
- Pushes the image to Amazon ECR
- Deploys the latest version to AWS EC2

---

# 📈 Performance

The chatbot was optimized by replacing a local CPU-hosted model with GPT-4o mini.

| Version | Average Response Time |
|----------|----------------------:|
| Ollama (CPU) | 20–40 seconds |
| GPT-4o mini | ~1.8 seconds |

---

# 📸 Screenshots

You can add screenshots here after deployment.

Example:

```
screenshots/home.png

screenshots/chat.png
```

---

# 🔮 Future Improvements

- Conversation memory
- Source citations
- Streaming responses
- Chat history
- User authentication
- Dark mode
- Better UI/UX
- Multi-language support

---

# 📚 Technologies Used

- Python
- Flask
- LangChain
- OpenAI GPT-4o mini
- Hugging Face
- Pinecone
- Docker
- AWS EC2
- Amazon ECR
- GitHub Actions

---

# 👨‍💻 Author

**Mohamad Abou Helal**

GitHub:

https://github.com/mohamadhelal94

LinkedIn:

https://www.linkedin.com/in/mohamad-abou-helal/
