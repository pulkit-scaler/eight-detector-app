import time

import streamlit as st


@st.cache_data
def load_data():
    time.sleep(3)
    return "Data loaded"


st.title("Caching Demo — stable key")

if st.button("Load Data"):
    st.write(load_data())
