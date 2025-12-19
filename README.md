# 🧠 End-to-End RAG Chatbot with Topic-Aware Retrieval

> An interactive **Retrieval-Augmented Generation (RAG)** chatbot that combines classical information retrieval, machine learning–based topic classification, and a modern web UI built with **Streamlit**.

---

## ✨ Overview

This repository contains the full pipeline for an **end-to-end RAG chatbot** where users can:

- 💬 Chat with an AI agent conversationally  
- 🔍 Retrieve **topic-specific information** backed by Wikipedia sources  
- 🎯 Refine retrieval results by **manually adjusting topics**

The system blends **traditional IR techniques** with **probabilistic classification** to provide transparent, controllable document retrieval.

## How It Works

- Wikipedia pages are **scraped, summarized, and indexed** using an **inverted index**
- Each page is categorized into **1 of 10 predefined topics**
- Documents are retrieved using the **Okapi BM25 scoring method**

---

## 🏗️ Chat Flow

```text
User Query
   │
   ▼
Naive Bayes Topic Classifier
   │
   ├─► Top-3 Predicted Topics
   │
   ▼
User Topic Refinement (Add / Remove Topics)
   │
   ▼
BM25 Document Retrieval (Topic-Filtered, Top-5 Documents)
   │
   ▼
LLM Response Generation
```

## Running the Webapp

```bash
# Clone the repository
git clone https://github.com/kn58buff/project3-IR-chatbot.git
cd project3-IR-chatbot

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run src/app.py
```

## 🖼️ Sample Screenshots
<img width="1527" height="292" alt="image" src="https://github.com/user-attachments/assets/47c83154-6423-4743-a42a-8212dfab9000" />

<img width="1521" height="380" alt="image" src="https://github.com/user-attachments/assets/587ae914-9a3d-4d86-90a8-9c0e999245f6" />

<img width="1528" height="804" alt="image" src="https://github.com/user-attachments/assets/0d4f2855-b428-4467-b613-82758644b4ba" />

<img width="1920" height="895" alt="image" src="https://github.com/user-attachments/assets/a93f6b58-77f6-4fe4-8d90-d63150603cae" />


## [🎥 Demo Video](https://www.youtube.com/watch?v=Gc9p9zfdhXY)

## 🛠️ Technologies Used
* Python
* Webscraping
* RAG/LLM/Prompt Engineering
* Information Retrieval
