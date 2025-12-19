import streamlit as st
import pandas as pd
from indexer import Indexer
import query_processor as qp
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from typing import Any, List
from langchain_ollama.llms import OllamaLLM
import time

st.title("Chatbot and IR System")

@st.cache_resource
def load_resources():
    indexer = Indexer.load("./inverted_index.pkl")
    scorer = qp.QueryProcessor(indexer)

    return indexer, scorer

@st.cache_resource
def load_llm():
    model_name = "llama3.2"

    llm = OllamaLLM(model = model_name, temperature = 0.5)
    return llm

@st.cache_data
def load_data():
    df = pd.read_csv("./wikipedia_scraped_data.csv")
    return df

llm = load_llm()
indexer, scorer = load_resources()
data = load_data()

topics = sorted(data["topic"].unique())

class DocumentRetriever(BaseRetriever):
    scorer: Any
    data: Any

    def _get_relevant_documents(self, query):
        topics = self.scorer._classify_query(query)

        docs = self.scorer.retrieve_rel_docs(query, topics = topics)
        available_ids = set(self.data["page_id"].astype(str).tolist())

        results = []
        for page_id, score in docs.items():
            pid = str(page_id)
            if pid not in available_ids:
                continue
            
            row = self.data.loc[self.data["page_id"].astype(str) == pid]

            if row.empty:
                continue

            r = row.iloc[0]

            results.append(
                Document(
                    page_content = r["summary"],
                    metadata = {"page_id": pid, "topic": r["topic"], "RSVBM25": score}
                )
            )
        return results

chat_or_ir = PromptTemplate.from_template("""
You are a classifier. Determine if the user wants:
- "chitchat": conversational, jokes, feelings, casual talk
- "retrieval": factual, answer needs external info or knowledge

User message: {input}

Return ONLY one word: CHITCHAT or RETRIEVAL
"""
)

retriever = DocumentRetriever(scorer = scorer, data = data)

split_chain = chat_or_ir | llm | StrOutputParser()

chitchat_prompt = ChatPromptTemplate.from_template("""
You are a friendly conversational AI that chats naturally. Conversation should be built upon the past exchanges.

This is the conversation so far: {chat_history}
                                               
The user just said: {input}

Respond in a friendly and engaging manner that fits the conversation context.
""")

def run_chitchat(msg):
    chat_history = "\n".join(f"{message['role']}: {message['content']}" for message in st.session_state.messages)

    prompt = chitchat_prompt.format_messages(input=msg, chat_history=chat_history)

    return llm.invoke(prompt)

chitchat_chain = chitchat_prompt | llm

rag_prompt = ChatPromptTemplate.from_template("""
You are an expert that answers questions ONLY using the information in the context.
                                              
All responses should sound NATURAL and not "machine-like". In addition to answering the question, you should also provide a cohesive general summary of all the documents retrieved, even if they did not contribute to answering the question. Provide your answer to the question within the document summary.
                                              
If the answer is not contained in the context, say: 'I don't know based on the provided documents.
                                              
Each document is labeled like [DOC 123].
In your answer AND the summary, after each claim, include a citation like (source: DOC 123).
Cite ONLY document IDs that appear in the context.

                                              
                                              
CONTEXT: {context}

QUESTION: {question}
""")

def make_rag_chain(llm, retriever):
    def rag_chain(inputs):
        query = inputs["input"]

        docs = retriever._get_relevant_documents(query)
        context = "\n\n".join([f"[DOC {d.metadata['page_id']}] {d.page_content}" for d in docs])

        prompt = rag_prompt.format_messages(question=query, context=context)

        print(prompt)
        response = llm.invoke(prompt)
        if hasattr(response, "to_string"):
            response = response.to_string()
        
        sources = [{"page_id": d.metadata.get("page_id")} for d in docs]
        return {"answer": response, "sources": sources}
    return rag_chain

def search_mode(msg):
    facts_keywords = ["who", "what", "when", "where", "how", "explain", "define", "calculate"]
    if any(msg.lower().startswith(k) for k in facts_keywords):
        return "retrieval"
    else:
        return "chitchat"

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "query_log" not in st.session_state:
        st.session_state.query_log = []

    if "await_query" not in st.session_state:
        st.session_state.await_query = None

    if "await_pred_topics" not in st.session_state:
        st.session_state.await_pred_topics = []

    if "await_selected_topics" not in st.session_state:
        st.session_state.await_selected_topics = []

    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    
    return True

initialize_session_state()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("Sources:"):
                for src in message["sources"]:
                    st.markdown(f"- **Page ID:** {src['page_id']}")

cbox = st.chat_input("Start chatting...")

if cbox:
    st.chat_message("user").markdown(cbox)

    st.session_state.messages.append({"role": "user", "content": cbox})

    route = split_chain.invoke({"input": cbox}).lower()
    print(route)

    st.session_state.start_time = time.time()

    #with st.chat_message("assistant"):
    if route == "chitchat":
        response = run_chitchat(cbox)

        st.chat_message("assistant").write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        end_time = time.time()

        elapsed_time = end_time - st.session_state.start_time

        st.session_state.start_time = None

        st.session_state.query_log.append({
        "query": cbox,
        "bot_response": response,
        "intent": "chitchat",
        "timestamp": pd.Timestamp.now(),
        "processing_time": elapsed_time
        })
    else:
        st.session_state.await_query = cbox                
        detected_topics = set(scorer._classify_query(cbox))

        st.session_state.await_pred_topics = list(detected_topics)
        st.session_state.await_selected_topics = detected_topics.copy()

        st.info(f"Detected topics: {', '.join(detected_topics)}")

        st.rerun()
if st.session_state.await_query:
    st.info("Adjust topics as needed and confirm, detected topics are pre-selected.")

    selected = st.multiselect(
        "Add or remove topics:",
        options = topics,
        default = st.session_state.await_selected_topics,
        key = "topic_selector"
    )

    st.session_state.await_selected_topics = selected

    if st.button("Run Query", key = "run_query_button"):
    #combined_topics = list(set(st.session_state.selected_topics) & detected_topics)
        filtered_data = data[data["topic"].isin(st.session_state.await_selected_topics)]
        retriever = DocumentRetriever(scorer = scorer, data = filtered_data)

        rag_chain = make_rag_chain(llm, retriever)

        response = rag_chain({"input": st.session_state.await_query})

        with st.chat_message("assistant"):
            st.write(response["answer"])

        with st.expander("Sources:"):
            for src in response["sources"]:
                st.markdown(f"- **Page ID:** {src['page_id']}")

        st.session_state.messages.append({"role": "assistant", "content": response["answer"],"sources": response["sources"]})
        
        end_time = time.time()
        elapsed_time = end_time - st.session_state.start_time

        st.session_state.query_log.append({
        "query": st.session_state.await_query,
        "intent": "retrieval",
        "bot_response": response["answer"],
        "detected_topics": st.session_state.await_pred_topics,
        "selected_topics": st.session_state.await_selected_topics,
        "relevance_scores": [d.metadata.get("RSVBM25") for d in retriever._get_relevant_documents(st.session_state.await_query)],
        "timestamp": pd.Timestamp.now(),
        "processing_time": elapsed_time
        })

        st.session_state.await_query = None
        st.session_state.await_pred_topics = []
        st.session_state.await_selected_topics = []
        st.session_state.start_time = None

        st.rerun()
