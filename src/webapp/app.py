"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd

# pages
main_page = st.Page("main_page.py", title="Main Page", icon="🏠")
viz_page = st.Page("viz_page.py", title="Visualization", icon="📊")

# nav
pg = st.navigation([main_page, viz_page])

pg.run()