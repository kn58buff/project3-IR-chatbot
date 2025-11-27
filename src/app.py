"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title = "Wikipedia IR", layout="wide")

st.title("Wikipedia Search System")
st.write("Navigate between **Search** and **Visualizations**")

st.markdown("---")
st.subheader("Pages")
st.markdown("""
- **Search:** Enter a query
- **Visualizations:** View statistics and logs of previous queries
""")