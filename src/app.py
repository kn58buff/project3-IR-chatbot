import streamlit as st
import pandas as pd

st.set_page_config(page_title = "Wikipedia IR", layout="wide")

st.title("Wikipedia Search System and Chatbot")

st.markdown("---")
st.subheader("Pages")
st.markdown("""
- **Search:** Chat with a bot or ask it a question to retrieve relevant Wikipedia articles.
- **Visualizations:** View statistics and logs of previous queries
""")