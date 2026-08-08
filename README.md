# 🩺 AI Medical Chatbot (RAG) | GPT-4o mini, LangChain, Pinecone & AWS

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o_mini-black)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-blueviolet)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![AWS](https://img.shields.io/badge/AWS-EC2_|_ECR-orange)

---

An AI-powered multilingual medical information chatbot built using **Retrieval-Augmented Generation (RAG)**.

The application retrieves relevant medical information from a **Pinecone vector database** built from curated **MedlinePlus-based medical content** and generates context-aware responses using **OpenAI GPT-4o mini**.

The chatbot supports:

- Multi-turn conversations
- Structured symptom extraction
- Medical risk assessment
- Automatic detection of new medical complaints
- Intelligent follow-up questions
- Multilingual conversations
- Source-grounded medical responses

The project is containerized with Docker and deployed using **AWS EC2**, **Amazon ECR**, and **GitHub Actions** for automated CI/CD.

---

# 🚀 Architecture

```text
                           User
                             │
                             ▼
                    Flask Web Application
                             │
                             ▼
                  Conversation Management
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    Structured Symptom Extraction   New Complaint Detection
              │                             │
              └──────────────┬──────────────┘
                             ▼
                 Medical Risk Assessment
                             │
                  ┌──────────┴──────────┐
                  │                     │
          Urgent Medical Case     Non-Urgent Case
                  │                     │
                  ▼                     ▼
     Immediate Safety Guidance   History-Aware RAG
                                        │
                              ┌─────────┴─────────┐
                              ▼                   ▼
                   Pinecone Vector DB      GPT-4o mini
                 (Medical Knowledge)        (OpenAI)
                              │                   │
                              └─────────┬─────────┘
                                        ▼
                         Medical Response + Sources
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
- Structured symptom extraction
- Multi-turn medical conversations
- Automatic detection of new medical complaints
- Medical risk assessment
- Emergency symptom detection
- Intelligent follow-up questions
- Context-aware conversation history
- Retrieval query translation into English
- Intelligent document reranking
- Medical source citations
- Multilingual support
- Flask web interface
- Docker containerization
- AWS EC2 deployment
- Amazon ECR
- GitHub Actions CI/CD
- Gunicorn production server

---

# 🎯 Current Capabilities

The chatbot currently supports:

- Retrieval-Augmented Generation (RAG)
- Context-aware medical conversations
- Structured symptom extraction
- Medical risk assessment
- Emergency symptom prioritization
- Automatic follow-up questions
- Retrieval query translation
- Intelligent document reranking
- Source-grounded medical responses
- Multilingual conversations (English, Swedish, Spanish and Arabic)

---

# 🌍 Supported Languages

The chatbot automatically understands and responds in:

- 🇺🇸 English
- 🇸🇪 Swedish
- 🇪🇸 Spanish
- 🇸🇦 Arabic

Medical retrieval is performed in English, while responses are generated in the user's preferred language.

---

# 🛠 Tech Stack

## Backend

- Python 3.10
- Flask
- Gunicorn

## AI / LLM

- OpenAI GPT-4o mini
- LangChain
- Pydantic
- Hugging Face Embeddings
- sentence-transformers/all-MiniLM-L6-v2

## Vector Database

- Pinecone

## Cloud

- AWS EC2
- Amazon ECR

## DevOps

- Docker
- GitHub Actions

---

# 📂 Project Structure

```text
Build-a-Complete-Medical-Chatbot
│
├── app.py
├── store_index.py
├── requirements.txt
├── Dockerfile
├── README.md
│
├── src/
│   ├── helper.py
│   └── prompt.py
│
├── templates/
│   └── chat.html
│
├── static/
│   └── style.css
│
├── scripts/
│   ├── parse_medlineplus.py
│   ├── build_topic_catalog.py
│   ├── map_topics_to_medlineplus.py
│   └── build_curated_documents.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── curated/
│
└── .github/
    └── workflows/
```

---

# ⚙️ Installation

## Clone Repository

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
conda activate medibot
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Create Environment File

Create a `.env` file:

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

# 🧠 AI Pipeline

```text
User Question
        │
        ▼
Conversation Analysis
        │
        ▼
Structured Symptom Extraction
        │
        ▼
Medical Risk Assessment
        │
        ▼
Medical Topic Detection
        │
        ▼
Query Translation (English)
        │
        ▼
Pinecone Retrieval
        │
        ▼
Document Reranking
        │
        ▼
History-Aware RAG
        │
        ▼
GPT-4o mini
        │
        ▼
Medical Response + Source Citations
```

---

# 📚 Medical Knowledge Base

The chatbot retrieves information from a curated medical knowledge base built using **MedlinePlus-based medical content**.

The pipeline:

- Processes medical documents
- Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Stores embeddings in Pinecone
- Retrieves semantically relevant medical information
- Grounds responses using retrieved sources

---

# ☁️ AWS Deployment

The application is containerized with Docker and deployed on AWS using Amazon EC2.

Deployment pipeline:

```text
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
Gunicorn
   │
   ▼
Flask Application
```

---

# 🔄 CI/CD

GitHub Actions automatically:

- Builds the Docker image
- Pushes the image to Amazon ECR
- Deploys the latest version to AWS EC2

---

# ⚡ Performance

| Version | Average Response Time |
|----------|----------------------:|
| Ollama (CPU) | 20–40 seconds |
| GPT-4o mini | ~1–2 seconds |

> Response times are approximate observations from local testing and may vary depending on API latency, network conditions, and retrieval complexity.

---

# 🔮 Future Improvements

- Voice input
- Streaming responses
- Medical image analysis
- PDF medical report analysis
- User authentication
- Persistent conversation history
- Improved UI/UX
- Dark mode
- Physician dashboard

---

# ⚠️ Medical Disclaimer

This project is intended for educational purposes only.

It provides **general medical information** and **does not diagnose diseases**, replace professional medical advice, or substitute consultation with a qualified healthcare provider.

For medical emergencies, users should immediately contact their local emergency services or seek urgent medical care.

---

# 📚 Technologies Used

- Python
- Flask
- LangChain
- OpenAI GPT-4o mini
- Hugging Face
- Pinecone
- Pydantic
- Docker
- AWS EC2
- Amazon ECR
- GitHub Actions

---

# 📄 License

This project is released for educational purposes.

---

# 👨‍💻 Author

**Mohamad Abou Helal**

GitHub:  
https://github.com/mohamadhelal94

LinkedIn:  
https://www.linkedin.com/in/mohamad-abou-helal/
