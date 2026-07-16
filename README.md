# Build a Complete Medical Chatbot
# Build-a-Complete-Medical-Chatbot-with-RAG-Ollama-LangChain-Pinecone-Flask

# How to run?

### STEPS:

Clone the repository

```bash
git clone https://github.com/mohamadhelal94/Build-a-Complete-Medical-Chatbot-.git
```

### STEP 01 - Create a Conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```

---

### STEP 02 - Install the requirements

```bash
pip install -r requirements.txt
```

---

### STEP 03 - Install Ollama

Download Ollama from:

https://ollama.com/download

After installation, pull the model:

```bash
ollama pull llama3.2:3b
```

Verify the model:

```bash
ollama list
```

---

### STEP 04 - Create a `.env` file in the root directory

```ini
PINECONE_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

### STEP 05 - Store embeddings into Pinecone

Run this command only once (or whenever you update the PDF):

```bash
python store_index.py
```

---

### STEP 06 - Run the Flask application

```bash
python app.py
```

Now open your browser and navigate to:

```text
http://127.0.0.1:8000
```

---

# Project Structure

```text
Build-a-Complete-Medical-Chatbot-/
│
├── app.py
├── store_index.py
├── requirements.txt
├── templates/
│   └── chat.html
├── static/
│   └── style.css
├── src/
│   ├── helper.py
│   └── prompt.py
├── research/
│   └── trials.ipynb
├── data/
│   └── Medical_book.pdf
└── README.md
```

---

### Tech Stack Used

- Python
- Flask
- LangChain
- Ollama (Llama 3.2)
- Hugging Face Embeddings
- Pinecone
- Sentence Transformers

---

# Features

- Medical Question Answering using RAG
- Local Large Language Model (Ollama)
- Hugging Face Embeddings
- Pinecone Vector Database
- Flask Web Interface
- LangChain Retrieval Pipeline
- No OpenAI API Required

---

# Example Questions

- What is Acromegaly?
- What are the symptoms of diabetes?
- Explain hypertension.
- What causes asthma?
- What is chronic kidney disease?

---

# AWS CICD Deployment with GitHub Actions

## 1. Login to AWS Console

---

## 2. Create IAM User for Deployment

### Required Access

- Amazon EC2
- Amazon ECR (Elastic Container Registry)

---

### Deployment Workflow

1. Build Docker image
2. Push Docker image to Amazon ECR
3. Launch EC2 Instance
4. Pull Docker image from ECR
5. Run the Docker container

---

### Required IAM Policies

- AmazonEC2ContainerRegistryFullAccess
- AmazonEC2FullAccess

---

## 3. Create an Amazon ECR Repository

Save the repository URI.

Example:

```text
xxxxxxxxxxxx.dkr.ecr.us-east-1.amazonaws.com/medical-chatbot
```

---

## 4. Create an EC2 Instance (Ubuntu)

---

## 5. Install Docker on EC2

```bash
sudo apt-get update -y
sudo apt-get upgrade -y

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

sudo usermod -aG docker ubuntu
newgrp docker
```

---

## 6. Configure EC2 as a Self-hosted GitHub Runner

GitHub Repository

```
Settings
    → Actions
        → Runners
            → New Self-hosted Runner
```

Follow the commands provided by GitHub.

---

## 7. Configure GitHub Secrets

Add the following repository secrets:

- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_DEFAULT_REGION
- ECR_REPOSITORY
- PINECONE_API_KEY

---

# Future Improvements

- Conversation Memory
- Source Citation
- Chat History
- User Authentication
- Multi-PDF Support
- Streaming Responses
- Docker Compose
- Kubernetes Deployment

---

# License

This project is for educational purposes.
