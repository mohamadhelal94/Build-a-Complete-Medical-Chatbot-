# 🩺 AI Medical Chatbot using RAG, GPT-4o mini, LangChain & Pinecone

An AI-powered multilingual Medical Chatbot built using **Retrieval-Augmented Generation (RAG)**. The application retrieves trusted medical information from a Pinecone vector database and generates context-aware responses using **OpenAI GPT-4o mini**.

The chatbot supports **multi-turn conversations**, **structured symptom extraction**, **medical risk assessment**, **automatic detection of new medical complaints**, and **multilingual interaction** while grounding responses in retrieved medical knowledge.

The project is containerized with Docker and can be deployed on **AWS EC2** using **Amazon ECR** and **GitHub Actions** for automated CI/CD.

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
                             ▼
               History-Aware RAG Retrieval
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
      Pinecone Vector Database       GPT-4o mini
      (Medical Knowledge Base)         (OpenAI)
               │                           │
               └─────────────┬─────────────┘
                             ▼
            Grounded Medical Response + Sources
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
- Context-aware conversation memory
- Retrieval query translation into English
- Automatic reranking of retrieved documents
- Medical source citations
- Multilingual support
- Flask web interface
- Docker containerization
- AWS EC2 deployment
- Amazon ECR
- GitHub Actions CI/CD
- Gunicorn production deployment

---

# 🌍 Supported Languages

The chatbot automatically understands and responds in:

- 🇺🇸 English
- 🇸🇪 Swedish
- 🇪🇸 Spanish
- 🇸🇦 Arabic

Medical retrieval is performed in English while responses are generated in the user's preferred language.

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
│   └── processed/
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

Open your browser:

```
http://localhost:8000
```

---

# 🧠 AI Pipeline

The chatbot follows a multi-stage AI pipeline:

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

# 🧠 Intelligent Medical Features

The chatbot includes several AI-powered capabilities:

- Structured symptom extraction
- Multi-turn conversations
- Automatic follow-up question generation
- Detection of new medical complaints
- Medical risk assessment
- Emergency symptom detection
- Retrieval query translation
- Context-aware RAG retrieval
- Intelligent document reranking
- Medical source attribution
- Multilingual conversations

---

# ☁ AWS Deployment

The project is deployed using Docker on AWS.

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

- Build the Docker image
- Push the image to Amazon ECR
- Deploy the latest version to AWS EC2

---

# 📈 Performance

| Version | Average Response Time |
|---------|----------------------:|
| Ollama (CPU) | 20–40 seconds |
| GPT-4o mini | ~1–2 seconds |

---

# 📸 Screenshots

Example screenshots:

```
screenshots/home.png

screenshots/chat.png

screenshots/emergency.png
```

---

# 🔮 Future Improvements

- Voice input
- Streaming responses
- Medical image analysis
- PDF medical report analysis
- User authentication
- Conversation persistence
- Better UI/UX
- Dark mode
- Physician dashboard

---

# ⚠️ Medical Disclaimer

This chatbot is designed to provide **general medical information and educational guidance only**.

It **does not diagnose diseases**, replace professional medical advice, or substitute consultation with a qualified healthcare provider.

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

# 👨‍💻 Author

**Mohamad Abou Helal**

GitHub

https://github.com/mohamadhelal94

LinkedIn

https://www.linkedin.com/in/mohamad-abou-helal/
