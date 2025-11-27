import streamlit as st
import pandas as pd
from indexer import Indexer
import query_processor as qp
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.title("Search Page")

@st.cache_resource
def load_resources():
    indexer = Indexer.load("./inverted_index.pkl")
    scorer = qp.QueryProcessor(indexer)

    return indexer, scorer

@st.cache_data
def load_data():
    df = pd.read_csv("./wikipedia_scraped_data.csv")
    return df

indexer, scorer = load_resources()
data = load_data()

class DocumentRetriever(BaseRetriever):
    scorer

    def _get_relevant_documents(self, query):
        topics = self.scorer._classify_query(query)

        docs = self.scorer.retrieve_rel_docs(query, topics = topics)

        langchain_docs = []
        for page_id in docs.keys():
            langchain_docs.append(
                Document(
                    page_content = data.loc[data["page_id"] == page_id].iloc[0]["summary"],
                    metadata = {"page_id": page_id, "topic": data.loc[data["page_id"] == page_id].iloc[0]["topic"]}
                )
            )
        return langchain_docs

llm = CTransformers(model = "hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF", model_type = "llama")

retriever = DocumentRetriever(scorer = scorer)

qa = RetrievalQA.from_chain_type(llm = llm, retriever = retriever, chain_type = "stuff")

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template = """
You are a friendly AI assistant that chats naturally but can answer fact-based questions by retrieving information when needed.

User question:
{question}

Here are the retrieved documents:
{context}

Please write a natural-sounding answer that summarizes the information. Then list the sources as a list of page IDs.

Format:
[Your summary]

Sources: [comma-separated list of page_IDs]
"""
)

qa.combine_documents_chain.llm_chain.prompt = prompt

def search_mode(q):
    keywords = ["explain", "what are", "what is", "information", "tell me about", "tell me more", "details", "retrieve"]

    return any(k in q.lower() for k in keywords)

def bot_response(msg):
    if search_mode(msg):
        return qa.run(msg)
    else:
        return llm(msg)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

cbox = st.chat_input("Start chatting...")

if cbox:
    st.chat_message("user").markdown(cbox)

    st.session_state.messages.append({"role": "user", "content": cbox})

    response = bot_response(cbox)

    st.chat_message("assistant").markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})