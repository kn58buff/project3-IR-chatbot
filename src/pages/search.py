import streamlit as st
import pandas as pd
from indexer import Indexer
import query_processor as qp
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.llms import CTransformers
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from typing import Any, List


st.title("Search Page")

@st.cache_resource
def load_resources():
    indexer = Indexer.load("./inverted_index.pkl")
    scorer = qp.QueryProcessor(indexer)

    return indexer, scorer

@st.cache_resource
def load_llm():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    pipe = pipeline(
        "text-generation",
        model = model_name,
        max_new_tokens = 512,
        temperature = 0.1
    )

    llm = HuggingFacePipeline(pipeline = pipe)
    return llm

@st.cache_data
def load_data():
    df = pd.read_csv("./wikipedia_scraped_data.csv")
    return df

llm = load_llm()
indexer, scorer = load_resources()
data = load_data()

class DocumentRetriever(BaseRetriever):
    scorer: Any
    data: Any

    def _get_relevant_documents(self, query):
        topics = self.scorer._classify_query(query)

        docs = self.scorer.retrieve_rel_docs(query, topics = topics)

        results = []
        for page_id in docs.keys():
            row = self.data.loc[self.data["page_id"] == page_id].iloc[0]
            results.append(
                Document(
                    page_content = row["summary"],
                    metadata = {"page_id": page_id, "topic": row["topic"]}
                )
            )
        return results

chat_or_ir = PromptTemplate.from_template("""
You are a classifier. Determine if the user wants:
- "chitchat": conversational, jokes, feelings, casual talk
- "retrieval": factual, answer needs external info or knowledge

User message: {input}

Return ONLY one word: "chitchat" or "retrieval".
"""
)

retriever = DocumentRetriever(scorer = scorer, data = data)

split_chain = chat_or_ir | llm | StrOutputParser()

chitchat_prompt = PromptTemplate.from_template("""
You are a friendly conversational AI that chats naturally.
Keep replies casual and engaging.
                                               
User: {input}
Assistant:
""")

chitchat_chain = chitchat_prompt | llm

rag_prompt = ChatPromptTemplate.from_template("""
Use ONLY the provided context to answer the question.

Context:
{context}

Question: {input}

Answer:
""")

def make_rag_chain(llm, retriever):
    def rag_chain(inputs):
        query = inputs["input"]

        docs = retriever._get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = rag_prompt.format(input=query, context=context)

        response = llm.invoke(prompt)
        if hasattr(response, "to_string"):
            response = response.to_string()
        return response

    return rag_chain

def search_mode(msg):
    facts_keywords = ["who", "what", "when", "where", "how", "explain", "define", "calculate"]
    if any(msg.lower().startswith(k) for k in facts_keywords):
        return "retrieval"
    else:
        return "chitchat"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

cbox = st.chat_input("Start chatting...")

if cbox:
    st.chat_message("user").markdown(cbox)

    st.session_state.messages.append({"role": "user", "content": cbox})

    with st.chat_message("assistant"):
        route = search_mode(cbox)
        print(route)
        rag_chain = make_rag_chain(llm, retriever)
        if route == "chitchat":
            response = llm.invoke(cbox)
        else:
            response = rag_chain({"input": cbox})

        st.chat_message("assistant").write(response)

        #st.session_state.messages.append({"role": "assistant", "content": response})