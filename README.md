#  Shakers Case Study: Retrieval-Augmented Generation (RAG) Pipeline

##  Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that delivers precise, context-aware conversational responses by combining a vector-based retrieval mechanism with LLM-powered generation.

The architecture is modular, graph-based, and allows conditional routing of user queries through various processing stages — including recommendation, intent classification, sentiment analysis, and fallback safety.

---

##  Ingestion System

###  Purpose

The ingestion system processes and imports external documents into a vector store, enabling efficient semantic search during question answering or recommendation.

###  How It Works

1. **Data Loading**
   Loads documents (Markdown) from a local path or API.

2. **Text Processing**
   Cleans and splits documents into manageable chunks.

3. **Embedding Generation**
   Converts each chunk into high-dimensional vectors using a pretrained embedder.

4. **Metadata Preparation**
   Attaches metadata such as source filename, document title, etc.

5. **Vector Store Insertion**
   Stores embeddings and metadata in a vector DB like **Qdrant** for fast semantic search.

---

##  Architecture

The system is built around a **graph-based state machine**, where each node performs a specific operation (e.g., detecting malicious input, intent classification, generating replies). The central idea is to **augment LLM answers with retrieved knowledge** from your own content.

![RAG Architecture](https://github.com/daniabellan/shakers_case_study/rag_architecture.png)

---

##  Core Components

### 1. **Malicious Query Detector**
- Flags unsafe content.
- Routes flagged queries to a fallback response.

### 2. **Intent Detector**
Classifies user intent:
- Direct
- Ambiguous
- Out of Scope

Routes queries accordingly.

### 3. **Ambiguous Question Handler**
- Clarifies vague queries before proceeding.

### 4. **Out-of-Scope Handler**
- Gracefully responds to unsupported topics.

### 5. **Message Saver**
- Stores all messages with metadata: user ID, sentiment, timestamps, etc.

### 6. **Sentiment Detector**
- Uses the LLM to classify the tone of the message.
- Sentiment influences future system behavior.

### 7. **QA Reply Generator**
- Retrieves relevant docs.
- Combines retrieval + LLM to craft grounded answers.
- Adds **personalized recommendations**.

### 8. **Unsafe Fallback**
- Returns a safe response for flagged input.

### 9. **Final Metrics Logger**
- Logs latency, success rates, and quality indicators.

---

##  User Profile Building

###  Purpose
Capture user interests and preferences based on their recent interactions to **personalize future recommendations**.

###  How It Works

1. Collects the most recent `n` user messages (e.g., 10).
2. Embeds each message into a vector.
3. Assigns **higher weights to more recent messages**.
4. Computes a **weighted average** to form a **profile vector**.
5. If no messages exist, returns a zero vector as fallback.

This profile vector is used to **adjust the semantic search**, tailoring results to the user’s interests.

---

##  Personalized Resource Recommendation

###  Purpose
Recommend relevant documents based on the user’s question and their profile context, ensuring **diversity and freshness**.

###  Process

1. **Embed the Question**
   Generate a vector for the current user query.

2. **Combine with Profile**
   Blend question vector with the profile vector (`alpha` controls weight).

3. **Search Vector Store**
   Retrieve top-K matching documents from Qdrant.

4. **Filter for Diversity**
   Use cosine similarity to avoid recommending redundant results.

5. **Exclude Seen Docs**
   Avoid recommending documents the user has already viewed.

6. **LLM Explanation**
   Use a prompt template to ask the LLM to explain how each doc answers the user's question.

7. **Log Metrics**
   Capture latency and token usage info for observability.

---

##  Setup & Execution Guide

### 1. Create Development Environment

Use `conda.yaml` to set up a reproducible dev environment.

```bash
conda env create -f conda.yaml
conda activate case_study_dev
```
###  Start Required Services

Make sure Docker is running. Then:

###  Qdrant (Vector Store)

```bash
cd docker/qdrant
docker compose up -d
```

### PostgreSQL (for chat history)

```bash
cd docker/postgresql
docker compose up -d
```

---

## 3. Run the FastAPI Backend

Start the backend server (in a new terminal):

```bash
cd shakers_case_study/
uvicorn shakers_case_study.app.backend.main:app --reload --port 8000
```

 API Docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 4. Run the Ingestion Script

In a second terminal, run:

```bash
python shakers_case_study/rag/pipelines/ingestion/run_ingestion.py
```

This will:
- Load source docs
- Split them
- Embed them
- Store in Qdrant with metadata

Make sure that you ingest documents first before doing a query.

---

## 5. Launch the Chatbot Interface (Streamlit)

In a third terminal, run:

```bash
streamlit run shakers_case_study/app/main.py --server.headless true
```

 Visit the chatbot at: [http://localhost:8501](http://localhost:8501)


##  Document Topics & User Profiles

The ingestion system processes a collection of documents that cover different functional areas of the platform. These documents are categorized into three main themes:

###  Onboarding
These documents help users get started with the platform, tailored for different roles:
- `01_getting_started.md`
- `02_onboarding_learners.md`
- `03_onboarding_instructors.md`
- `04_onboarding_admins.md`
- `05_faq_and_troubleshooting.md`

###  Payment
These documents explain the platform's billing, subscriptions, and payment processes:
- `06_payment_methods.md`
- `07_subscription_plans.md`
- `08_billing_and_invoices.md`
- `09_refunds_and_cancellations.md`
- `10_payment_faq.md`

###  Course Types
These describe the types of learning experiences offered:
- `11_course_types_overview.md`
- `12_self_paced_courses.md`
- `13_instructor_led_courses.md`
- `14_certification_programs.md`
- `15_corporate_learning_paths.md`

---

##  User Profiles

Based on user interactions with the topics above, the system dynamically generates **user profiles** to tailor recommendations and answers.

Each profile is built by analyzing recent message history, embedding the content, and computing a weighted profile vector that captures the user's interests (e.g., onboarding, payments, certification programs, etc.).

 You can view the predefined user profiles in the following file:

```bash
shakers_case_study/rag/pipelines/rag/profiles.json
```

These profiles are used to:
- Personalize resource recommendations
- Adjust LLM responses based on user preferences
- Improve context-awareness across conversations

These profiles have been created using the multiple queries in the json file.

## Secure Deployment via Cloudflare Tunnel

To securely expose the local application (backend API and/or chatbot UI) to the internet without deploying to a public server, I integrated a **Cloudflare Tunnel**.

### Why Cloudflare Tunnel?
- Exposes local services to the web **without exposing your IP**
- Adds **automatic HTTPS** encryption
- No need to open ports on your router or deploy to a cloud host
- Ideal for demos, testing, or remote access

### How It Works
- The backend (FastAPI) and frontend (Streamlit) can be tunneled using a Cloudflare-provided subdomain.
- A Cloudflare Tunnel agent (`cloudflared`) is run locally to establish a secure outbound connection to Cloudflare's edge network.
- The services can then be accessed publicly via a secure `https://<custom-subdomain>.trycloudflare.com` URL or via a custom domain if configured.

> This approach was used to make the solution accessible externally in a secure and controlled manner during development and testing.


## Pending Improvements & Known Limitations

Some intended features were not fully implemented due to time constraints. These areas have been identified for future development to improve robustness, usability, and deployment readiness:

###  Full Dockerization
-  A unified `docker-compose` setup for the entire pipeline (backend, Qdrant, PostgreSQL, and frontend) was not completed.
-  Although services are runnable individually, integrating all components into a single, plug-and-play Docker environment was postponed due to the time required for:
     - Proper environment variable and secret configuration
     - Adjusting internal URLs and network access between services
     - Pipeline initialization setup

###  Unit Testing
-  No unit or integration tests have been implemented.
-  Adding tests for ingestion, retrieval logic, and the backend API would greatly enhance code reliability and facilitate refactoring.

###  Explicit Feedback Collection
-  UI elements like “Helpful / Not Helpful” buttons were planned to gather user feedback on LLM responses.
-  This feedback would be useful for refining future responses and closing the human-in-the-loop evaluation loop.

###  Context Compression
-  Techniques to compress or summarize long user history into a compact representation were not implemented.
-  These are essential for scaling to longer conversations and reducing token usage while preserving relevance.

---

These areas are important for making the system production-ready and will be prioritized in future iterations.
